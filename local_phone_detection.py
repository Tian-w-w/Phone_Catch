"""
手机识别脚本 —— 基于本地 Ollama 部署的 Qwen3-VL
输出格式：YES（存在手机）/ NO（不存在手机）
结果写入日志文件，便于后期复盘
"""

import os
import base64
import logging
import httpx
import json
import re
from pathlib import Path

# ──────────────────────────────────────────────
# 配置区（按需修改）
# ──────────────────────────────────────────────
OLLAMA_BASE_URL = "https://10.19.205.173:11434/v1"
MODEL_NAME      = "qwen3-vl:32b"
LOG_FILE        = "phone_recognition.log"
VERIFY_SSL      = False   # 自签证书设为 False

# ──────────────────────────────────────────────
# 提示词
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """
你是一个专业的手机识别视觉分析系统。
你的任务是：严谨判断画面中是否存在手机（smartphone），并输出最终判定结果。
请注意：手机可能藏在某个位置。
【手机判定条件】：
条件A：外形轮廓 —— 长方形薄片状物体。
条件B：屏幕特征 —— 玻璃/金属质感的平整表面，有镜面反光或发光显示区域。
条件C：摄像头模组 —— 关键特征！物体表面存在圆形镜头开孔、摄像头凸起、多摄模组或闪光灯。
条件D：手持姿态 —— 有手部明确握持该物体，手指呈持机姿势。
条件E：细节特征 —— 可见手机品牌logo、充电口、音量键等细节。
【判定逻辑（二元分类）】：
判定为"是"（YES）：
  - 必须识别到【条件C：摄像头模组】。只要看到摄像头相关模组，且伴随 A/B/D/E 中任意一项，即判定为存在手机。
  - 或者：虽然没看到摄像头，但同时满足 A、B、D、E 四项中的三项以上，且视觉证据极度明确。
判定为"否"（NO）：
  - 未能识别到摄像头模组（条件C不满足）。
  - 虽然疑似长方形物体，但特征不足三项（如：只是个充电宝、记事本或钱包）。
  - 画面模糊导致无法确认核心特征。
【排除对象】：
- 记事本/书本、钱包、充电宝、遥控器、平板电脑（尺寸过大且无手机特征）。
只返回 JSON，不返回任何其他文字。
"""

DETECTION_PROMPT = """
请分析图片，判断是否存在手机。
严格按照以下 JSON 格式返回：
{
  "is_phone": "YES" 或 "NO",
  "confidence": 0到100的整数,
  "matched_conditions": ["条件A", "条件C"],
  "camera_detected": true或false,
  "phone_location": "物体位置描述，若无则填null",
  "key_evidence": "最关键的视觉证据说明",
  "exclusion_reason": "若判定为NO，请说明理由；否则填null"
}
【判定流程】：
1. 检查五个条件（A-E），记录到 matched_conditions。
2. 执行最终判定：
   - 若 (条件C = true 且 匹配总数 >= 2) OR (匹配总数 >= 3 且 证据确凿) -> "is_phone": "YES"
   - 其余所有情况（特征模糊、仅有轮廓、疑似但无摄像头等） -> "is_phone": "NO"
3. 填写剩余字段。
【注意】：
- 你的判定结果必须非黑即白。
- 只要无法百分之百确定是手机，请倾向于判定为 "NO" 以降低误报。
- 手机摄像头可能藏在图片中某个地方，请仔细判断。
- 手机不限于iphone、xiaomi、huawei、oppo、vivo等等
- 摄像头（条件C）是判定为 "YES" 的核心权重。
"""

# ──────────────────────────────────────────────
# 日志配置
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────
def encode_image(image_path: str) -> tuple[str, str]:
    """将图片编码为 base64，并推断 MIME 类型。"""
    suffix = Path(image_path).suffix.lower()
    mime_map = {
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png":  "image/png",
        ".gif":  "image/gif",
        ".webp": "image/webp",
        ".bmp":  "image/bmp",
    }
    mime = mime_map.get(suffix, "image/jpeg")
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return data, mime


def parse_result(raw: str) -> tuple[str, dict | None]:
    """
    从模型输出中解析 JSON，提取 is_phone 字段。
    返回 ("YES"/"NO", 完整json字典或None)
    """
    # 去掉可能的 markdown 代码块标记
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        data = json.loads(cleaned)
        is_phone = str(data.get("is_phone", "NO")).strip().upper()
        answer = "YES" if is_phone == "YES" else "NO"
        return answer, data
    except json.JSONDecodeError:
        # 容错：直接在原始文本里找 YES/NO
        upper = raw.upper()
        answer = "YES" if '"IS_PHONE": "YES"' in upper or "'IS_PHONE': 'YES'" in upper else "NO"
        logger.warning(f"JSON 解析失败，降级文本匹配，原始输出：{raw[:200]}")
        return answer, None


def detect_phone(image_path: str) -> str:
    """
    调用 Qwen3-VL 判断图片中是否存在手机。
    返回 "YES" 或 "NO"。
    """
    b64, mime = encode_image(image_path)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.strip(),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                    {
                        "type": "text",
                        "text": DETECTION_PROMPT.strip(),
                    },
                ],
            },
        ],
        "temperature": 0,
        "max_tokens": 300,   # JSON 字段约 200 token 即可
        "stream": False,
    }

    url = f"{OLLAMA_BASE_URL.rstrip('/')}/chat/completions"

    with httpx.Client(verify=VERIFY_SSL, timeout=120) as client:
        resp = client.post(
            url,
            headers={"Content-Type": "application/json"},
            content=json.dumps(payload),
        )
        resp.raise_for_status()

    raw = resp.json()["choices"][0]["message"]["content"].strip()
    answer, _ = parse_result(raw)
    return answer


# ──────────────────────────────────────────────
# 单张图片
# ──────────────────────────────────────────────
def run_single(image_path: str, label: str | None = None) -> str:
    """
    识别单张图片，记录日志，返回 YES/NO。

    label：可选，预期正确答案（YES/NO），用于日志标注对错。
    """
    result = detect_phone(image_path)

    if label is not None:
        match  = (result == label.strip().upper())
        status = "✓" if match else "✗"
        logger.info(f"{status} | 文件={image_path} | 预测={result} | 标注={label.upper()}")
    else:
        logger.info(f"  | 文件={image_path} | 预测={result}")

    return result


# ──────────────────────────────────────────────
# 批量识别
# ──────────────────────────────────────────────
def run_batch(tasks: list[dict]) -> dict:
    """
    批量运行手机识别，统计准确率。

    tasks 格式：
    [
        {"image": "path/to/img.jpg", "label": "YES"},  # label 可省略
        {"image": "path/to/img2.jpg", "label": "NO"},
        ...
    ]

    返回统计字典：{"total": N, "correct": N, "accuracy": 0.xx}
    """
    logger.info("=" * 60)
    logger.info(f"批量任务开始 | 共 {len(tasks)} 张 | 模型：{MODEL_NAME}")
    logger.info("=" * 60)

    total, correct = len(tasks), 0
    has_label = any("label" in t for t in tasks)

    for idx, task in enumerate(tasks, 1):
        image_path = task["image"]
        label      = task.get("label", None)

        try:
            result = detect_phone(image_path)
        except Exception as e:
            logger.error(f"[{idx:04d}] 文件={image_path} | 错误：{e}")
            continue

        if label is not None:
            match  = (result == label.strip().upper())
            correct += int(match)
            status = "✓" if match else "✗"
            logger.info(
                f"[{idx:04d}] {status} | 文件={image_path} "
                f"| 预测={result} | 标注={label.upper()}"
            )
        else:
            logger.info(f"[{idx:04d}]   | 文件={image_path} | 预测={result}")

    # 汇总
    if has_label:
        acc = correct / total * 100
        logger.info("-" * 60)
        logger.info(f"准确率：{correct}/{total}  ({acc:.1f}%)")

    logger.info("=" * 60)
    return {"total": total, "correct": correct, "accuracy": correct / total if total else 0}


# ──────────────────────────────────────────────
# 示例入口
# ──────────────────────────────────────────────
if __name__ == "__main__":

    # —— 示例 1：单张图片 ——
    # result = run_single("test.jpg", label="YES")
    # print("识别结果:", result)

    # —— 示例 2：批量任务 ——
    tasks = [
        {"image": "images/phone_001.jpg", "label": "YES"},
        {"image": "images/no_phone_001.jpg", "label": "NO"},
        {"image": "images/phone_002.png", "label": "YES"},
        {"image": "images/unknown.jpg"},          # 无标注，仅看预测
    ]

    stats = run_batch(tasks)
    print(f"\n最终准确率：{stats['correct']}/{stats['total']}")
