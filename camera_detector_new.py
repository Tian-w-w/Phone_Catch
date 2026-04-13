#!/usr/bin/env python3
"""
手机摄像头检测项目
使用 Qwen3-VL 模型，通过 MaaS OpenAI 兼容 API，批量检测图像中是否存在摄像头模组
遇到限流（429）时自动等待并重试，直到成功为止
"""

import os
import sys
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

# 触发限流后的等待时间（秒），与平台每分钟 10 次限额对齐
RATE_LIMIT_WAIT = 60


# ─────────────────────────────────────────────
# 日志配置
# ─────────────────────────────────────────────
def setup_logger(log_file: str = None) -> logging.Logger:
    """配置日志：同时输出到控制台和文件"""
    logger = logging.getLogger("CameraDetector")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────
# 图像工具
# ─────────────────────────────────────────────
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

def collect_images(directory: str) -> list[Path]:
    """递归收集目录下所有支持的图片文件"""
    base = Path(directory)
    if not base.exists():
        raise FileNotFoundError(f"目录不存在：{directory}")
    return sorted(
        p for p in base.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTS
    )


def encode_image_base64(image_path: Path) -> tuple[str, str]:
    """将图片编码为 base64，返回 (base64_data, media_type)"""
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
SYSTEM_PROMPT = """你是一位专业的手机硬件视觉检测专家。
你的任务是分析图像，判断图中是否存在手机摄像头模组（Camera Module）。
摄像头模组的典型特征包括：
- 圆形或方形的镜头开孔
- 玻璃或塑料镜片（通常有反光）
- 金属或塑料摄像头环/保护框
- 多摄像头阵列排列
- 闪光灯、ToF 传感器等伴随组件

请严格按照以下格式输出结论，不要添加任何多余内容：
检测结论: <是/否>
置信度: <高/中/低>
原因: <简要说明，不超过50字>"""

USER_PROMPT = "请分析此图像，判断是否检测到手机摄像头模组。"


# ─────────────────────────────────────────────
# 限流判断
# ─────────────────────────────────────────────
def _is_rate_limit_error(e: Exception) -> bool:
    """判断异常是否为限流错误（HTTP 429 / RateLimitError）"""
    err_str = str(e)
    return (
        "429" in err_str
        or "rate limit" in err_str.lower()
        or "rate_limit" in err_str.lower()
        or "too many requests" in err_str.lower()
        or "RateLimitError" in type(e).__name__
    )


# ─────────────────────────────────────────────
# 检测核心（含自动限流重试）
# ─────────────────────────────────────────────
def detect_camera(
    client: OpenAI,
    model: str,
    image_path: Path,
    logger: logging.Logger,
) -> dict:
    """
    调用 Qwen3-VL API 分析单张图片。
    - 限流错误（429）：等待 RATE_LIMIT_WAIT 秒后无限重试，直到成功
    - 其他错误：记录后直接返回，不重试
    """
    result = {
        "file": str(image_path),
        "detected": None,
        "confidence": "未知",
        "reason": "",
        "raw_response": "",
        "error": None,
    }

    b64_data, media_type = encode_image_base64(image_path)
    attempt = 0

    while True:
        attempt += 1
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{b64_data}"
                                },
                            },
                            {"type": "text", "text": USER_PROMPT},
                        ],
                    },
                ],
                max_tokens=256,
                temperature=0,
            )

            raw = response.choices[0].message.content.strip()
            result["raw_response"] = raw

            # 解析模型结构化输出
            lines = {
                line.split(":")[0].strip(): line.split(":", 1)[1].strip()
                for line in raw.splitlines()
                if ":" in line
            }
            conclusion = lines.get("检测结论", "").strip()
            result["detected"]   = "是" in conclusion
            result["confidence"] = lines.get("置信度", "未知")
            result["reason"]     = lines.get("原因", raw[:80])
            return result  # 成功，退出

        except Exception as e:
            if _is_rate_limit_error(e):
                # 限流：打印倒计时日志，等待后重试
                logger.warning(
                    f"  ⚠️  触发限流（第 {attempt} 次尝试）"
                    f" | 文件: {image_path.name}"
                    f" | 等待 {RATE_LIMIT_WAIT} 秒后重试..."
                )
                for remaining in range(RATE_LIMIT_WAIT, 0, -10):
                    logger.info(f"      ⏳ 还需等待 {remaining} 秒...")
                    time.sleep(min(10, remaining))
                logger.info("      ▶️  等待结束，重新发起请求")
                # 继续 while 循环重试
            else:
                # 非限流错误，不重试
                result["error"] = str(e)
                return result


# ─────────────────────────────────────────────
# 日志输出
# ─────────────────────────────────────────────
def log_result(result: dict, idx: int, total: int, logger: logging.Logger):
    """将单条检测结果写入日志"""
    fname  = Path(result["file"]).name
    prefix = f"[{idx}/{total}] {fname}"

    if result["error"]:
        logger.error(f"{prefix} | 调用失败 | 错误: {result['error']}")
        return

    tag = "✅ 发现摄像头模组" if result["detected"] else "❌ 未发现摄像头模组"
    logger.info(
        f"{prefix} | {tag} | "
        f"置信度: {result['confidence']} | "
        f"原因: {result['reason']}"
    )


def log_summary(results: list[dict], logger: logging.Logger):
    """输出批量检测汇总"""
    total     = len(results)
    detected  = sum(1 for r in results if r["detected"] is True)
    not_found = sum(1 for r in results if r["detected"] is False)
    failed    = sum(1 for r in results if r["error"] is not None)

    logger.info("=" * 60)
    logger.info("【检测汇总报告】")
    logger.info(f"  总图片数量  : {total}")
    logger.info(f"  发现摄像头  : {detected} 张")
    logger.info(f"  未发现摄像头: {not_found} 张")
    logger.info(f"  调用失败    : {failed} 张")
    logger.info("=" * 60)

    if detected > 0:
        logger.info("发现摄像头的文件列表：")
        for r in results:
            if r["detected"] is True:
                logger.info(f"  - {r['file']}")

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
    logger = setup_logger(log_file)

    logger.info("=" * 60)
    logger.info("手机摄像头检测系统启动")
    logger.info(f"  模型        : {MODEL}")
    logger.info(f"  API Base    : {BASE_URL}")
    logger.info(f"  图片目录    : {args.image_dir}")
    logger.info(f"  限流等待    : {RATE_LIMIT_WAIT} 秒（触发后自动重试）")
    logger.info(f"  日志文件    : {log_file}")
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

    results = []
    for idx, img_path in enumerate(images, start=1):
        logger.info(f"[{idx}/{len(images)}] 正在分析: {img_path.name}")
        result = detect_camera(client, MODEL, img_path, logger)
        log_result(result, idx, len(images), logger)
        results.append(result)

    log_summary(results, logger)
    logger.info(f"检测完成，日志已保存至: {log_file}")


if __name__ == "__main__":
    main()
