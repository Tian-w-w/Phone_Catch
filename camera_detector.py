#!/usr/bin/env python3
"""
手机摄像头检测项目
使用 Qwen3-VL 模型，通过 MaaS OpenAI 兼容 API，批量检测图像中是否存在摄像头模组
"""

import os
import sys
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

# ─────────────────────────────────────────────
# 日志配置
# ─────────────────────────────────────────────
def setup_logger(log_file: str = None) -> logging.Logger:
    """配置日志：同时输出到控制台和文件（可选）"""
    logger = logging.getLogger("CameraDetector")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 控制台 Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 文件 Handler（可选）
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
    images = sorted(
        p for p in base.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTS
    )
    return images


def encode_image_base64(image_path: Path) -> tuple[str, str]:
    """将图片编码为 base64，返回 (base64_data, media_type)"""
    suffix = image_path.suffix.lower()
    media_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
        ".tiff": "image/tiff",
    }
    media_type = media_map.get(suffix, "image/jpeg")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return b64, media_type


# ─────────────────────────────────────────────
# 检测核心
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


def detect_camera(
    client: OpenAI,
    model: str,
    image_path: Path,
    logger: logging.Logger,
) -> dict:
    """
    调用 Qwen3-VL API 分析单张图片
    返回结构：{
        "file": str,
        "detected": bool | None,  # True=有摄像头, False=无, None=解析失败
        "confidence": str,
        "reason": str,
        "raw_response": str,
        "error": str | None
    }
    """
    result = {
        "file": str(image_path),
        "detected": None,
        "confidence": "未知",
        "reason": "",
        "raw_response": "",
        "error": None,
    }

    try:
        b64_data, media_type = encode_image_base64(image_path)

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
            temperature=0,  # 检测任务使用确定性输出
        )

        raw = response.choices[0].message.content.strip()
        result["raw_response"] = raw

        # 解析模型输出
        lines = {
            line.split(":")[0].strip(): line.split(":", 1)[1].strip()
            for line in raw.splitlines()
            if ":" in line
        }
        conclusion = lines.get("检测结论", "").strip()
        result["detected"] = "是" in conclusion
        result["confidence"] = lines.get("置信度", "未知")
        result["reason"] = lines.get("原因", raw[:80])

    except Exception as e:
        result["error"] = str(e)

    return result


# ─────────────────────────────────────────────
# 日志输出
# ─────────────────────────────────────────────
def log_result(result: dict, idx: int, total: int, logger: logging.Logger):
    """将单条检测结果写入日志"""
    fname = Path(result["file"]).name
    prefix = f"[{idx}/{total}] {fname}"

    if result["error"]:
        logger.error(f"{prefix} | 调用失败 | 错误: {result['error']}")
        return

    detected_str = "✅ 发现摄像头模组" if result["detected"] else "❌ 未发现摄像头模组"
    logger.info(
        f"{prefix} | {detected_str} | "
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
    parser.add_argument(
        "--image-dir", "-d",
        required=True,
        help="待检测图片所在目录路径"
    )
    parser.add_argument(
        "--log-file", "-l",
        default=None,
        help="日志文件路径（可选，默认仅输出到控制台）"
    )
    args = parser.parse_args()

    # ── 初始化日志 ──
    log_file = args.log_file or f"camera_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logger(log_file)

    logger.info("=" * 60)
    logger.info("手机摄像头检测系统启动")
    logger.info(f"  模型      : {MODEL}")
    logger.info(f"  API Base  : {BASE_URL}")
    logger.info(f"  图片目录  : {args.image_dir}")
    logger.info(f"  日志文件  : {log_file}")
    logger.info("=" * 60)

    # ── 初始化 OpenAI 兼容客户端 ──
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    )

    # ── 收集图片 ──
    try:
        images = collect_images(args.image_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    if not images:
        logger.warning(f"目录 {args.image_dir} 下未找到任何支持的图片文件")
        sys.exit(0)

    logger.info(f"共发现 {len(images)} 张图片，开始批量检测...\n")

    # ── 批量检测 ──
    results = []
    for idx, img_path in enumerate(images, start=1):
        logger.info(f"[{idx}/{len(images)}] 正在分析: {img_path.name}")
        result = detect_camera(client, MODEL, img_path, logger)
        log_result(result, idx, len(images), logger)
        results.append(result)

    # ── 汇总报告 ──
    log_summary(results, logger)
    logger.info(f"检测完成，日志已保存至: {log_file}")


if __name__ == "__main__":
    main()
