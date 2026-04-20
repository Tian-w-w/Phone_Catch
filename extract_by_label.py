"""
基于打标结果的数据抽取工具
读取 label_dataset.py 输出的 JSON，按条件过滤图片并拷贝到目标文件夹
"""

import json
import shutil
import argparse
from pathlib import Path
from collections import defaultdict


# ─────────────────────────────────────────────
# 过滤条件（按需修改）
# ─────────────────────────────────────────────
DEFAULT_FILTERS = {
    # 只保留光线为 bright_natural 或 artificial_light 的图片
    "lighting": ["bright_natural", "artificial_light"],

    # 只保留手部遮挡为 none 或 partial 的图片
    "hand_occlusion": ["none", "partial"],

    # 只保留竖屏手持设备的图片
    "device_orientation": ["portrait"],

    # 只保留包含 smartphone 或 tablet 的图片（设备列表 any match）
    "devices_include_any": ["smartphone", "tablet"],

    # 图片质量必须是 clear
    "image_quality": ["clear"],
}


def load_results(json_path: str) -> list[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("results", [])


def matches_filters(labels: dict, filters: dict) -> bool:
    """判断一张图片的标注是否满足所有过滤条件"""

    # 标量字段匹配
    for field in ["lighting", "hand_occlusion", "device_orientation", "image_quality"]:
        if field in filters:
            val = labels.get(field, "unknown")
            if val not in filters[field]:
                return False

    # devices any-match
    if "devices_include_any" in filters:
        devices = [d.lower() for d in labels.get("devices", [])]
        wanted  = [d.lower() for d in filters["devices_include_any"]]
        if not any(w in devices for w in wanted):
            return False

    # num_hands 范围（可选）
    if "num_hands_max" in filters:
        try:
            if int(labels.get("num_hands", 999)) > filters["num_hands_max"]:
                return False
        except (ValueError, TypeError):
            pass

    return True


def extract_images(
    json_path: str,
    output_dir: str,
    filters: dict,
    copy_files: bool = True,
    save_manifest: bool = True,
):
    results = load_results(json_path)
    print(f"[信息] 共加载 {len(results)} 条标注记录")

    matched = []
    for item in results:
        if item.get("status") != "ok":
            continue
        if matches_filters(item.get("labels", {}), filters):
            matched.append(item)

    print(f"[信息] 符合过滤条件的图片: {len(matched)} 张")

    if copy_files and matched:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for item in matched:
            src = Path(item["image_path"])
            if src.exists():
                dst = out / src.name
                # 避免同名覆盖
                if dst.exists():
                    dst = out / f"{src.stem}_{src.stat().st_ino}{src.suffix}"
                shutil.copy2(src, dst)
        print(f"[信息] 图片已拷贝至 {Path(output_dir).resolve()}")

    if save_manifest:
        manifest_path = Path(output_dir) / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                {"filters": filters, "matched_count": len(matched), "items": matched},
                f, ensure_ascii=False, indent=2,
            )
        print(f"[信息] 清单已保存至 {manifest_path.resolve()}")

    # 打印简单统计
    print_filter_stats(matched)
    return matched


def print_filter_stats(matched: list[dict]):
    if not matched:
        print("[!] 没有任何图片通过过滤条件。")
        return

    counters = defaultdict(lambda: defaultdict(int))
    for item in matched:
        lb = item.get("labels", {})
        for field in ["lighting", "hand_occlusion", "device_orientation",
                      "scene", "image_quality"]:
            counters[field][lb.get(field, "unknown")] += 1
        for dev in lb.get("devices", []):
            counters["devices"][dev.lower()] += 1

    print("\n" + "─" * 50)
    print("  过滤后数据分布")
    print("─" * 50)
    for field, counts in counters.items():
        print(f"\n【{field}】")
        for val, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {val:<25} {cnt}")


# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="基于标注结果的图片抽取工具")
    parser.add_argument("--labels", "-l", required=True,
                        help="label_dataset.py 输出的 JSON 文件")
    parser.add_argument("--output", "-o", default="./extracted",
                        help="抽取结果输出目录（默认 ./extracted）")
    parser.add_argument("--no-copy", action="store_true",
                        help="只生成 manifest，不拷贝图片文件")
    args = parser.parse_args()

    # ── 在这里按需自定义你的过滤条件 ──
    filters = DEFAULT_FILTERS

    extract_images(
        json_path=args.labels,
        output_dir=args.output,
        filters=filters,
        copy_files=not args.no_copy,
        save_manifest=True,
    )


if __name__ == "__main__":
    main()
