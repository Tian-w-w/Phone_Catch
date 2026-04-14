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
SYSTEM_PROMPT = """你是一名视觉理解助手。
请观察图像，凭借你对现实世界的理解，判断图中是否存在手机或摄像头模组。
不需要依赖任何特定描述，直接根据你对图像内容的整体认知作出判断。

输出要求（严格按此格式，不得添加任何其他内容）：
检测结论: <有/没有>
置信度: <高/中/低>
原因: <用一句话说明你的判断依据，不超过30字>"""

USER_PROMPT = "请判断这张图片中是否存在手机或摄像头模组？"

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
        "detected":    None,
        "confidence":  "未知",
        "reason":      "",
        "raw_response": "",
        "error":       None,
        "elapsed_sec": 0.0,  # 本张图片实际 API 耗时（不含等待重试）
    }

    b64_data, media_type = encode_image_base64(image_path)
    attempt = 0

    while True:
        attempt += 1
        t_start = time.perf_counter()
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

            elapsed = time.perf_counter() - t_start
            result["elapsed_sec"] = elapsed

            raw = response.choices[0].message.content.strip()
            result["raw_response"] = raw

            # 解析结构化输出
            lines = {
                line.split(":")[0].strip(): line.split(":", 1)[1].strip()
                for line in raw.splitlines()
                if ":" in line
            }
            conclusion = lines.get("检测结论", "").strip()
            result["detected"]   = "是" in conclusion
            result["confidence"] = lines.get("置信度", "未知")
            result["reason"]     = lines.get("原因", raw[:80])
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

    tag = "✅ 发现摄像头模组" if result["detected"] else "❌ 未发现摄像头模组"
    logger.info(
        f"{prefix} | {tag}"
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
    logger.info(f"  发现摄像头      : {detected} 张")
    logger.info(f"  未发现摄像头    : {not_found} 张")
    logger.info(f"  调用失败        : {failed} 张")
    logger.info(f"  总耗时          : {total_elapsed:.2f}s（含所有等待重试）")
    logger.info(f"  平均每张 API 耗时: {avg_api:.2f}s（仅计成功请求，不含等待）")
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
