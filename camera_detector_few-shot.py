#!/usr/bin/env python3
"""
手机摄像头检测项目
使用 Qwen3-VL 模型，通过 MaaS OpenAI 兼容 API，批量检测图像中是否存在摄像头模组

错误处理策略：
  - 限流 (429)         : 等待 60 秒后重试
  - 认证失败 (401/403) : 等待 50 秒后重试
  - 网络超时           : 指数退避重试
  - 其他服务端错误     : 指数退避重试
"""

import os
import sys
import json
import time
import base64
import logging
import argparse
from datetime import datetime
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] 缺少依赖：请先运行 pip install openai")
    sys.exit(1)

# ─────────────────────────────────────────────
# 配置区域（请填写以下三项）
# ─────────────────────────────────────────────
API_KEY   = "XX"   # 替换为你的 API Key
BASE_URL  = "XX"   # 替换为你的 MaaS API Base URL，例如 https://xxx/v1
MODEL     = "XX"   # 替换为模型名称，例如 qwen3-vl

# ── 重试参数 ──
WAIT_RATE_LIMIT  = 60   # 限流 (429) 等待秒数
WAIT_AUTH_ERROR  = 50   # 认证失败 (401/403) 等待秒数
BACKOFF_BASE     = 5    # 指数退避初始等待秒数（实际等待 = BACKOFF_BASE * 2^n）
BACKOFF_MAX      = 120  # 指数退避上限秒数


# ─────────────────────────────────────────────
# 日志配置
# ─────────────────────────────────────────────
def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("CameraDetector")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────
# 图像工具
# ─────────────────────────────────────────────
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

def collect_images(directory: str) -> list[Path]:
    base = Path(directory)
    if not base.exists():
        raise FileNotFoundError(f"目录不存在：{directory}")
    return sorted(
        p for p in base.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTS
    )


def encode_image_base64(image_path: Path) -> tuple[str, str]:
    media_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png",  ".bmp": "image/bmp",
        ".webp": "image/webp", ".tiff": "image/tiff",
    }
    media_type = media_map.get(image_path.suffix.lower(), "image/jpeg")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return b64, media_type


# ─────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """
你是一个专业的摄影设备识别视觉分析系统。
你的任务是：严谨判断画面中是否存在手机（smartphone）或运动相机（action camera），并输出最终判定结果。
请注意：目标设备可能藏在画面某个角落，请仔细搜索整张图片。

【手机判定条件】：
条件A：外形轮廓 —— 长方形薄片状物体。
条件B：屏幕特征 —— 玻璃/金属质感的平整表面，有镜面反光或发光显示区域。
条件C：摄像头模组 —— 关键特征！物体表面存在圆形镜头开孔、摄像头凸起、多摄模组或闪光灯。
条件D：手持姿态 —— 有手部明确握持该物体，手指呈持机姿势。
条件E：细节特征 —— 可见手机品牌logo、充电口、音量键等细节。

【运动相机判定条件】：
条件F：外形轮廓 —— 小巧方块状或圆筒状机身，体积明显小于单反相机。
条件G：镜头特征 —— 关键特征！正面具有大口径鱼眼或广角镜头，镜头通常凸出机身，圆形开孔明显。
条件H：机身特征 —— 坚固防水外壳（常见黑色/彩色塑料），可见GoPro、DJI Osmo Action、Insta360等品牌标识或其风格。
条件I：安装配件 —— 机身底部或侧面可见卡扣、固定夹、自拍杆连接口等配件。
条件J：使用姿态 —— 固定在头盔/车把/胸前支架，或手持自拍杆拍摄。

【判定逻辑（二元分类）】：
判定为"是"（YES）——满足以下任一：
  手机：识别到【条件C：摄像头模组】且伴随A/B/D/E中任意一项；或同时满足A、B、D、E四项中的三项以上且视觉证据极度明确。
  运动相机：识别到【条件G：镜头特征】且伴随F/H/I/J中任意一项；或同时满足F、H、I、J四项中的三项以上且视觉证据极度明确。

判定为"否"（NO）——满足以下任一：
  手机：未识别到条件C，且满足A/B/D/E不足三项；或画面模糊无法确认核心特征。
  运动相机：未识别到条件G，且满足F/H/I/J不足三项；或画面模糊无法确认核心特征。
  以上两类目标均未识别到。

【排除对象】：
- 记事本/书本、钱包、充电宝、遥控器（无镜头）。
- 平板电脑（尺寸过大且无手机特征）。
- 单反/微单相机（体积过大，机身风格与运动相机明显不同）。
- 监控摄像头、行车记录仪（固定安装，无手持/运动特征）。

只返回 JSON，不返回任何其他文字。
"""

USER_PROMPT = """
请分析图片，判断是否存在手机或运动相机。
严格按照以下 JSON 格式返回：
{
  "is_target": "YES" 或 "NO",
  "target_type": "手机" 或 "运动相机" 或 "两者都有" 或 "无",
  "confidence": 0到100的整数,
  "matched_conditions": ["条件A", "条件C"],
  "camera_detected": true或false,
  "target_location": "物体位置描述，若无则填null",
  "key_evidence": "最关键的视觉证据说明",
  "exclusion_reason": "若判定为NO，请说明理由；否则填null"
}
【判定流程】：
1. 分别检查手机条件（A-E）和运动相机条件（F-J），记录到 matched_conditions。
2. 执行最终判定：
   - 手机：(条件C=true 且 匹配总数>=2) OR (匹配总数>=3 且 证据确凿) -> is_target="YES", target_type="手机"
   - 运动相机：(条件G=true 且 匹配总数>=2) OR (匹配总数>=3 且 证据确凿) -> is_target="YES", target_type="运动相机"
   - 两者均检测到 -> is_target="YES", target_type="两者都有"
   - 其余所有情况 -> is_target="NO", target_type="无"
3. 填写剩余字段。
【注意】：
- 你的判定结果必须非黑即白。
- 只要无法百分之百确定，请倾向于判定为"NO"以降低误报。
- 手机不限于iPhone、Xiaomi、Huawei、OPPO、vivo等品牌。
- 运动相机不限于GoPro，DJI Osmo Action、Insta360、Sony FDR等均在判定范围内。
- 摄像头/镜头（条件C/G）是各自判定为YES的核心权重。
"""

# ─────────────────────────────────────────────
# Few-shot 示例（引导 32B 模型理解判定逻辑和输出格式）
# 每个示例由：用户侧"图片描述文本" + 助手侧"标准 JSON 回复" 构成
# 真实推理时，示例轮次在 messages 最前面插入，最后一轮才是真实图片
# ─────────────────────────────────────────────
FEW_SHOT_EXAMPLES = [
    # ── 正例 1：手机（背面多摄，手持） ──
    {
        "user_desc": (
            "图片内容描述：一只手握持一部现代智能手机，手机背面朝上，"
            "可见右上角有三个圆形摄像头镜头排列成L形，旁边有一个小型闪光灯，"
            "机身为黑色玻璃背板，边框为金属材质，整体为长方形薄片状。"
        ),
        "assistant_json": {
            "is_target": "YES",
            "target_type": "手机",
            "confidence": 97,
            "matched_conditions": ["条件A", "条件C", "条件D", "条件E"],
            "camera_detected": True,
            "target_location": "画面中央，手持状态，背面朝向镜头",
            "key_evidence": "背面三摄模组+闪光灯清晰可见，手指呈持机姿势，金属边框长方形机身",
            "exclusion_reason": None
        }
    },
    # ── 正例 2：运动相机（GoPro 风格，固定在头盔） ──
    {
        "user_desc": (
            "图片内容描述：一名骑手佩戴头盔，头盔正前方固定了一个小型方块状设备，"
            "设备正面有一个大口径圆形广角镜头明显凸出，机身为黑色防水塑料外壳，"
            "底部可见金属卡扣固定件，整体体积约为火柴盒大小。"
        ),
        "assistant_json": {
            "is_target": "YES",
            "target_type": "运动相机",
            "confidence": 95,
            "matched_conditions": ["条件F", "条件G", "条件H", "条件I", "条件J"],
            "camera_detected": True,
            "target_location": "头盔正前方，卡扣固定",
            "key_evidence": "正面大口径广角镜头凸出明显，方块状防水机身，头盔固定卡扣，符合GoPro风格",
            "exclusion_reason": None
        }
    },
    # ── 正例 3：两者都有 ──
    {
        "user_desc": (
            "图片内容描述：桌面上放置了两个设备：左侧是一部屏幕朝上的智能手机，"
            "可见发光屏幕和顶部摄像头开孔；右侧是一个小型方块状运动相机，"
            "正面有圆形广角镜头，机身侧面有固定卡扣，黑色防水外壳。"
        ),
        "assistant_json": {
            "is_target": "YES",
            "target_type": "两者都有",
            "confidence": 96,
            "matched_conditions": ["条件A", "条件B", "条件C", "条件F", "条件G", "条件I"],
            "camera_detected": True,
            "target_location": "桌面左侧为手机，右侧为运动相机",
            "key_evidence": "手机屏幕发光+前置摄像孔；运动相机广角镜头+防水机身+固定卡扣，两者同时存在",
            "exclusion_reason": None
        }
    },
    # ── 负例 1：充电宝（易混淆） ──
    {
        "user_desc": (
            "图片内容描述：桌面上放置一个长方形黑色物体，表面平整光滑，"
            "一侧有两个USB接口和一排LED指示灯，无任何镜头或摄像头开孔，"
            "无屏幕，无品牌logo，整体外观与充电宝高度一致。"
        ),
        "assistant_json": {
            "is_target": "NO",
            "target_type": "无",
            "confidence": 92,
            "matched_conditions": ["条件A"],
            "camera_detected": False,
            "target_location": None,
            "key_evidence": None,
            "exclusion_reason": "物体为充电宝：有USB口和LED灯但无摄像头镜头、无屏幕、无持机姿势，排除手机和运动相机"
        }
    },
    # ── 负例 2：画面模糊，无法确认 ──
    {
        "user_desc": (
            "图片内容描述：画面整体严重模糊，可隐约看到桌面上有一个深色矩形物体，"
            "但无法分辨是否有摄像头、屏幕或其他细节特征，整体视觉信息不足。"
        ),
        "assistant_json": {
            "is_target": "NO",
            "target_type": "无",
            "confidence": 30,
            "matched_conditions": [],
            "camera_detected": False,
            "target_location": None,
            "key_evidence": None,
            "exclusion_reason": "画面严重模糊，无法确认摄像头模组或任何关键特征，按规则倾向判定为NO"
        }
    },
]


def build_few_shot_messages(system_prompt: str, image_b64: str, media_type: str) -> list[dict]:
    """
    构建包含 few-shot 示例的完整 messages 列表：
      [system]
      [user: 示例1描述] -> [assistant: 示例1 JSON]
      [user: 示例2描述] -> [assistant: 示例2 JSON]
      ...
      [user: 真实图片 + USER_PROMPT]  ← 最后才是真实推理请求
    """
    import json as _json
    messages = [{"role": "system", "content": system_prompt}]

    for ex in FEW_SHOT_EXAMPLES:
        # 用户侧：文字描述图片内容（代替真实图片）
        messages.append({
            "role": "user",
            "content": ex["user_desc"] + "\n\n" + USER_PROMPT
        })
        # 助手侧：标准 JSON 回复
        messages.append({
            "role": "assistant",
            "content": _json.dumps(ex["assistant_json"], ensure_ascii=False)
        })

    # 最后一轮：真实图片
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{image_b64}"},
            },
            {"type": "text", "text": USER_PROMPT},
        ],
    })
    return messages


# ─────────────────────────────────────────────
# 错误分类
# ─────────────────────────────────────────────
def _classify_error(e: Exception) -> str:
    """
    将异常归类为以下四类之一：
      rate_limit  - 限流 (429)
      auth_error  - 认证失败 (401/403)
      timeout     - 网络超时
      server_error - 其他服务端/未知错误
    """
    err_str  = str(e).lower()
    err_type = type(e).__name__

    # 限流
    if (
        "429" in err_str
        or "rate limit" in err_str
        or "rate_limit" in err_str
        or "too many requests" in err_str
        or "RateLimitError" in err_type
    ):
        return "rate_limit"

    # 认证失败
    if (
        "401" in err_str
        or "403" in err_str
        or "unauthorized" in err_str
        or "forbidden" in err_str
        or "authentication" in err_str
        or "AuthenticationError" in err_type
        or "PermissionDeniedError" in err_type
    ):
        return "auth_error"

    # 网络超时
    if (
        "timeout" in err_str
        or "timed out" in err_str
        or "APITimeoutError" in err_type
        or "ConnectTimeout" in err_type
        or "ReadTimeout" in err_type
    ):
        return "timeout"

    # 其他服务端错误
    return "server_error"


def _backoff_wait(attempt: int) -> float:
    """计算指数退避等待时间：BACKOFF_BASE * 2^(attempt-1)，上限 BACKOFF_MAX"""
    return min(BACKOFF_BASE * (2 ** (attempt - 1)), BACKOFF_MAX)


# ─────────────────────────────────────────────
# 检测核心（含完整错误重试策略）
# ─────────────────────────────────────────────
def detect_camera(
    client: OpenAI,
    model: str,
    image_path: Path,
    logger: logging.Logger,
) -> dict:
    """
    调用 Qwen3-VL API 分析单张图片，返回检测结果与耗时。
    错误处理：
      - 限流 (429)         → 固定等待 60 秒重试
      - 认证失败 (401/403) → 固定等待 50 秒重试
      - 网络超时           → 指数退避重试
      - 其他服务端错误     → 指数退避重试
    """
    result = {
        "file":        str(image_path),
        "detected":    None,      # True=检测到目标, False=未检测到
        "target_type": "无",      # 手机 / 运动相机 / 两者都有 / 无
        "confidence":  0,         # 0-100
        "reason":      "",
        "raw_response": "",
        "error":       None,
        "elapsed_sec": 0.0,
    }

    b64_data, media_type = encode_image_base64(image_path)
    attempt = 0

    while True:
        attempt += 1
        t_start = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=build_few_shot_messages(SYSTEM_PROMPT, b64_data, media_type),
                max_tokens=512,  # few-shot 轮次增多，适当加大 token 上限
                temperature=0,
            )

            elapsed = time.perf_counter() - t_start
            result["elapsed_sec"] = elapsed

            raw = response.choices[0].message.content.strip()
            result["raw_response"] = raw

            # 解析 JSON 输出（去除可能的 markdown 代码块包裹）
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            parsed = json.loads(clean)
            result["detected"]     = parsed.get("is_target", "NO").upper() == "YES"
            result["target_type"]  = parsed.get("target_type", "无")
            result["confidence"]   = parsed.get("confidence", 0)
            result["reason"]       = (
                parsed.get("key_evidence")
                or parsed.get("exclusion_reason")
                or ""
            )
            return result  # ✅ 成功

        except Exception as e:
            elapsed  = time.perf_counter() - t_start
            category = _classify_error(e)

            if category == "rate_limit":
                wait = WAIT_RATE_LIMIT
                label = f"限流 (429)"
            elif category == "auth_error":
                wait = WAIT_AUTH_ERROR
                label = f"认证失败 (401/403)"
            elif category == "timeout":
                wait = _backoff_wait(attempt)
                label = f"网络超时 [指数退避第 {attempt} 次]"
            else:
                wait = _backoff_wait(attempt)
                label = f"服务端错误 [指数退避第 {attempt} 次]"

            logger.warning(
                f"  ⚠️  {label} | 文件: {image_path.name} "
                f"| 耗时: {elapsed:.1f}s | 等待 {wait:.0f}s 后重试..."
                f"\n      错误详情: {e}"
            )
            time.sleep(wait)
            logger.info(f"      ▶️  重试中（第 {attempt + 1} 次尝试）...")


# ─────────────────────────────────────────────
# 日志输出
# ─────────────────────────────────────────────
def log_result(result: dict, idx: int, total: int, logger: logging.Logger):
    fname  = Path(result["file"]).name
    prefix = f"[{idx}/{total}] {fname}"

    if result["error"]:
        logger.error(
            f"{prefix} | 调用失败 | 耗时: {result['elapsed_sec']:.2f}s"
            f" | 错误: {result['error']}"
        )
        return

    tag = "✅ 有" if result["detected"] else "❌ 没有"
    logger.info(
        f"{prefix} | {tag}"
        f" | 类型: {result['target_type']}"
        f" | 置信度: {result['confidence']}"
        f" | 耗时: {result['elapsed_sec']:.2f}s"
        f" | 原因: {result['reason']}"
    )


def log_summary(results: list[dict], total_elapsed: float, logger: logging.Logger):
    total     = len(results)
    detected  = sum(1 for r in results if r["detected"] is True)
    not_found = sum(1 for r in results if r["detected"] is False)
    failed    = sum(1 for r in results if r["error"] is not None)
    api_times = [r["elapsed_sec"] for r in results if r["error"] is None]
    avg_api   = (sum(api_times) / len(api_times)) if api_times else 0.0

    logger.info("=" * 60)
    logger.info("【检测汇总报告】")
    logger.info(f"  总图片数量      : {total} 张")
    logger.info(f"  检测到目标      : {detected} 张")
    logger.info(f"  未检测到目标    : {not_found} 张")
    logger.info(f"  调用失败        : {failed} 张")
    logger.info(f"  总耗时          : {total_elapsed:.2f}s（含所有等待重试）")
    logger.info(f"  平均每张 API 耗时: {avg_api:.2f}s（仅计成功请求，不含等待）")
    logger.info("=" * 60)

    if detected > 0:
        logger.info("检测到目标的文件列表：")
        for r in results:
            if r["detected"] is True:
                logger.info(f"  - {r['file']} [{r['target_type']}]")

    if failed > 0:
        logger.warning("调用失败的文件列表：")
        for r in results:
            if r["error"]:
                logger.warning(f"  - {r['file']} | {r['error']}")


# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="使用 Qwen3-VL 批量检测图像中的手机摄像头模组"
    )
    parser.add_argument("--image-dir", "-d", required=True, help="待检测图片所在目录路径")
    parser.add_argument("--log-file",  "-l", default=None,  help="日志文件路径（可选）")
    args = parser.parse_args()

    log_file = args.log_file or f"camera_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger   = setup_logger(log_file)

    logger.info("=" * 60)
    logger.info("手机摄像头检测系统启动")
    logger.info(f"  模型            : {MODEL}")
    logger.info(f"  API Base        : {BASE_URL}")
    logger.info(f"  图片目录        : {args.image_dir}")
    logger.info(f"  限流等待        : {WAIT_RATE_LIMIT}s")
    logger.info(f"  认证失败等待    : {WAIT_AUTH_ERROR}s")
    logger.info(f"  指数退避初始值  : {BACKOFF_BASE}s（上限 {BACKOFF_MAX}s）")
    logger.info(f"  日志文件        : {log_file}")
    logger.info("=" * 60)

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    try:
        images = collect_images(args.image_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    if not images:
        logger.warning(f"目录 {args.image_dir} 下未找到任何支持的图片文件")
        sys.exit(0)

    logger.info(f"共发现 {len(images)} 张图片，开始批量检测...\n")

    wall_start = time.perf_counter()
    results    = []

    for idx, img_path in enumerate(images, start=1):
        logger.info(f"[{idx}/{len(images)}] 正在分析: {img_path.name}")
        result = detect_camera(client, MODEL, img_path, logger)
        log_result(result, idx, len(images), logger)
        results.append(result)

    total_elapsed = time.perf_counter() - wall_start
    log_summary(results, total_elapsed, logger)
    logger.info(f"检测完成，日志已保存至: {log_file}")


if __name__ == "__main__":
    main()
