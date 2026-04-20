"""
图像数据集自动打标工具
调用本地 QWEN3-VL 模型对数据集图片进行多维度内容识别与标注
"""

import os
import json
import base64
import time
import argparse
from pathlib import Path
from collections import defaultdict
from openai import OpenAI

# ─────────────────────────────────────────────
# 配置区
# ─────────────────────────────────────────────
BASE_URL = "http://10.19.205.173:11434/v1"
API_KEY  = "ollama"          # Ollama 本地服务固定填 "ollama"
MODEL    = "qwen3-vl"        # 与 ollama list 中的模型名保持一致

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}

# 打标维度 & 提示词
SYSTEM_PROMPT = """你是一个专业的图像内容分析助手。
请严格按照 JSON 格式返回分析结果，不要输出任何额外文字。"""

USER_PROMPT = """请仔细分析这张图片，并返回如下 JSON 结构（所有字段必填，不确定时填 "unknown"）：

{
  "devices": ["画面中出现的设备列表，如 smartphone、tablet、laptop、camera 等"],
  "lighting": "整体光线条件，从以下选项中选一个: bright_natural / dim_indoor / artificial_light / backlit / mixed / unknown",
  "hand_occlusion": "手部是否遮挡设备，从以下选项中选一个: none / partial / heavy / unknown",
  "device_orientation": "手持设备的方向，从以下选项中选一个: portrait / landscape / flat / no_device / unknown",
  "scene": "场景类型，从以下选项中选一个: indoor / outdoor / studio / unknown",
  "num_hands": "画面中手的数量，整数，0 表示没有手",
  "image_quality": "图片质量，从以下选项中选一个: clear / blurry / noisy / overexposed / underexposed / unknown",
  "extra_notes": "其他值得记录的信息（简短），没有则填空字符串"
}

只返回 JSON，不要有任何解释或 markdown 代码块。"""


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def encode_image(image_path: str) -> str:
    """将图片编码为 base64 字符串"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_mime(ext: str) -> str:
    mapping = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png",  ".bmp": "image/bmp",
        ".webp": "image/webp", ".gif": "image/gif",
    }
    return mapping.get(ext.lower(), "image/jpeg")


def call_model(client: OpenAI, image_path: str, retries: int = 3) -> dict:
    """调用 QWEN3-VL 模型，返回解析后的 JSON 标注结果"""
    ext = Path(image_path).suffix.lower()
    b64  = encode_image(image_path)
    mime = get_image_mime(ext)

    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime};base64,{b64}"
                                },
                            },
                            {"type": "text", "text": USER_PROMPT},
                        ],
                    },
                ],
                temperature=0.1,
                max_tokens=512,
            )

            raw = response.choices[0].message.content.strip()

            # 去掉可能存在的 markdown 围栏
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            parsed = json.loads(raw)
            return {"status": "ok", "labels": parsed}

        except json.JSONDecodeError as e:
            print(f"  [!] JSON 解析失败 (attempt {attempt}): {e} | raw={raw[:120]}")
        except Exception as e:
            print(f"  [!] 请求异常 (attempt {attempt}): {e}")

        if attempt < retries:
            time.sleep(2 ** attempt)   # 指数退避

    return {"status": "error", "labels": {}, "error": "max retries exceeded"}


def collect_images(folder: str) -> list[str]:
    """递归收集文件夹中所有支持的图片路径"""
    folder_path = Path(folder)
    images = []
    for ext in SUPPORTED_EXTS:
        images.extend(folder_path.rglob(f"*{ext}"))
        images.extend(folder_path.rglob(f"*{ext.upper()}"))
    return sorted(set(str(p) for p in images))


def summarize_labels(results: list[dict]) -> dict:
    """
    统计每个标签维度下各类别出现次数。
    仅统计 status=ok 的记录。
    """
    counters = defaultdict(lambda: defaultdict(int))

    scalar_fields = [
        "lighting", "hand_occlusion", "device_orientation",
        "scene", "image_quality"
    ]
    int_fields = ["num_hands"]

    for item in results:
        if item.get("status") != "ok":
            continue
        labels = item.get("labels", {})

        # 列表字段：devices
        for dev in labels.get("devices", []):
            counters["devices"][str(dev).lower()] += 1

        # 标量字段
        for field in scalar_fields:
            val = labels.get(field, "unknown")
            counters[field][str(val)] += 1

        # 数值字段
        for field in int_fields:
            val = labels.get(field, "unknown")
            try:
                key = str(int(val))
            except (ValueError, TypeError):
                key = "unknown"
            counters[field][key] += 1

    # 转为普通 dict，并按数量排序
    summary = {}
    for field, counts in counters.items():
        summary[field] = dict(
            sorted(counts.items(), key=lambda x: x[1], reverse=True)
        )
    return summary


def print_summary(summary: dict):
    """终端打印汇总表"""
    print("\n" + "=" * 60)
    print("  标签类别统计汇总")
    print("=" * 60)
    for field, counts in summary.items():
        print(f"\n【{field}】")
        for val, cnt in counts.items():
            bar = "█" * min(cnt, 40)
            print(f"  {val:<25} {cnt:>5}  {bar}")
    print("=" * 60)


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="QWEN3-VL 图像数据集自动打标工具")
    parser.add_argument(
        "--dataset", "-d",
        required=True,
        help="数据集图片根目录，例如 ./dataset"
    )
    parser.add_argument(
        "--output", "-o",
        default="labels_output.json",
        help="输出 JSON 文件路径（默认 labels_output.json）"
    )
    parser.add_argument(
        "--max", "-n",
        type=int,
        default=0,
        help="最多处理 N 张图片（0 表示全部）"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="每张图片之间的间隔秒数（避免过载，默认 0.5s）"
    )
    args = parser.parse_args()

    # 初始化客户端
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    # 收集图片
    all_images = collect_images(args.dataset)
    if not all_images:
        print(f"[错误] 在 {args.dataset} 中未找到任何支持的图片文件。")
        return

    target = all_images if args.max == 0 else all_images[: args.max]
    print(f"[信息] 共发现 {len(all_images)} 张图片，本次处理 {len(target)} 张")

    # 逐张打标
    results = []
    for i, img_path in enumerate(target, 1):
        rel_path = os.path.relpath(img_path, args.dataset)
        print(f"[{i:>4}/{len(target)}] {rel_path}")

        result = call_model(client, img_path)
        results.append({
            "image_path": img_path,
            "relative_path": rel_path,
            **result,
        })

        # 实时保存（防崩溃丢数据）
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(
                {"results": results, "total": len(results)},
                f, ensure_ascii=False, indent=2
            )

        if result["status"] == "ok":
            print(f"       ✓ devices={result['labels'].get('devices')}  "
                  f"lighting={result['labels'].get('lighting')}  "
                  f"orientation={result['labels'].get('device_orientation')}")
        else:
            print(f"       ✗ 标注失败")

        if args.delay > 0 and i < len(target):
            time.sleep(args.delay)

    # 汇总统计
    summary = summarize_labels(results)
    print_summary(summary)

    # 将汇总也写入 JSON
    output_data = {
        "total_images": len(target),
        "success_count": sum(1 for r in results if r["status"] == "ok"),
        "error_count": sum(1 for r in results if r["status"] != "ok"),
        "label_summary": summary,
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n[完成] 结果已保存至 {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
