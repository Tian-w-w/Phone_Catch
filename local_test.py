"""
手机识别脚本 —— 扫描文件夹，批量识别，结果写入日志
用法：python phone_detection.py --folder /path/to/images
      python phone_detection.py --folder /path/to/images --workers 4
"""

import argparse
import base64
import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

# ──────────────────────────────────────────────
# 配置区
# ──────────────────────────────────────────────
OLLAMA_BASE_URL = "https://10.19.205.173:11434/v1"
MODEL_NAME      = "qwen3-vl:32b"
LOG_FILE        = "phone_detection.log"
VERIFY_SSL      = False

SUPPORTED_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}

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
# 日志
# ──────────────────────────────────────────────
def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("phone_det")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


logger = setup_logger(LOG_FILE)

# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────
def collect_images(folder: str) -> list[Path]:
    """扫描单层文件夹，返回所有支持格式的图片路径（已排序）。"""
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise NotADirectoryError(f"路径不存在或不是文件夹：{folder}")
    images = sorted(
        p for p in folder_path.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )
    return images


def encode_image(image_path: Path) -> tuple[str, str]:
    mime_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png",  ".gif":  "image/gif",
        ".webp": "image/webp", ".bmp": "image/bmp",
    }
    mime = mime_map.get(image_path.suffix.lower(), "image/jpeg")
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return data, mime


def parse_result(raw: str) -> str:
    """从模型 JSON 输出中提取 is_phone，返回 YES/NO。"""
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        data = json.loads(cleaned)
        val  = str(data.get("is_phone", "NO")).strip().upper()
        return "YES" if val == "YES" else "NO"
    except json.JSONDecodeError:
        upper = raw.upper()
        if '"IS_PHONE": "YES"' in upper or "'IS_PHONE': 'YES'" in upper:
            return "YES"
        logger.warning(f"JSON 解析失败，降级匹配，原始输出：{raw[:200]}")
        return "NO"


def detect_phone(image_path: Path) -> str:
    """调用 Qwen3-VL 判断单张图片是否含手机，返回 YES/NO。"""
    b64, mime = encode_image(image_path)
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                    {"type": "text", "text": DETECTION_PROMPT.strip()},
                ],
            },
        ],
        "temperature": 0,
        "max_tokens": 300,
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
    return parse_result(raw)


# ──────────────────────────────────────────────
# 批量处理（支持多线程）
# ──────────────────────────────────────────────
def run_folder(folder: str, workers: int = 1) -> None:
    """
    扫描文件夹，对所有图片跑手机识别，结果写入日志。

    folder  : 图片文件夹路径
    workers : 并发线程数（默认 1，显存充足可适当提高）
    """
    images = collect_images(folder)
    total  = len(images)

    if total == 0:
        logger.warning(f"文件夹 [{folder}] 中未找到任何支持的图片，退出。")
        return

    logger.info("=" * 60)
    logger.info(f"文件夹：{folder}")
    logger.info(f"图片总数：{total}  |  并发线程：{workers}  |  模型：{MODEL_NAME}")
    logger.info("=" * 60)

    yes_count = 0
    err_count = 0

    def _process(idx_path: tuple[int, Path]) -> tuple[int, Path, str]:
        idx, path = idx_path
        result = detect_phone(path)
        return idx, path, result

    # 用字典保存 future -> (idx, path)，按完成顺序打印
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process, (idx, p)): (idx, p)
            for idx, p in enumerate(images, 1)
        }
        for future in as_completed(futures):
            idx, path = futures[future]
            try:
                _, _, result = future.result()
            except Exception as e:
                logger.error(f"[{idx:05d}/{total}] ERROR  {path.name}  |  {e}")
                err_count += 1
                continue

            yes_count += int(result == "YES")
            logger.info(f"[{idx:05d}/{total}] {result:<3}  {path.name}")

    # 汇总行
    no_count = total - yes_count - err_count
    logger.info("-" * 60)
    logger.info(
        f"完成 | 共 {total} 张 | "
        f"YES={yes_count}  NO={no_count}  ERROR={err_count}"
    )
    logger.info("=" * 60)


# ──────────────────────────────────────────────
# 命令行入口
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="手机识别 —— 批量文件夹模式")
    parser.add_argument(
        "--folder", "-f",
        required=True,
        help="图片文件夹路径，例如：/data/images 或 D:\\dataset\\photos",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="并发线程数（默认 1，显存充足可适当提高，建议不超过 4）",
    )
    args = parser.parse_args()
    run_folder(folder=args.folder, workers=args.workers)
