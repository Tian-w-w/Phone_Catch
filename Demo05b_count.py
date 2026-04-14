#!/usr/bin/env python3
"""
检测指标评估脚本
从日志文件中解析检测结果，结合图片文件名的真实标签，
计算准确率、精确率、召回率、F1-Score

真实标签规则：
  - 文件名以 True  开头（如 True1.jpg   ~ True500.jpg）  → 正例（有摄像头）
  - 文件名以 False 开头（如 False1.jpg  ~ False49.jpg）  → 负例（无摄像头）
"""

import re
import sys
import argparse
from pathlib import Path


# ─────────────────────────────────────────────
# 日志解析
# ─────────────────────────────────────────────
# 匹配形如：[1/549] True1.jpg | 有 | ...
#       或：[1/549] False1.jpg | 没有 | ...
LOG_PATTERN = re.compile(
    r"\[\d+/\d+\]\s+(?P<filename>\S+?)\s+\|"
    r"\s*(?P<result>有|没有)\s*\|"
)


def parse_log(log_path: str) -> list[dict]:
    """
    解析日志文件，返回每条记录：
      { "filename": str, "predicted": bool }
    """
    records = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            m = LOG_PATTERN.search(line)
            if m:
                records.append({
                    "filename":  m.group("filename"),
                    "predicted": m.group("result") == "有",
                })
    return records


# ─────────────────────────────────────────────
# 真实标签推断
# ─────────────────────────────────────────────
def get_ground_truth(filename: str) -> bool | None:
    """
    根据文件名前缀判断真实标签：
      True*  → True  （正例，有摄像头）
      False* → False （负例，无摄像头）
      其他   → None  （无法判断，跳过）
    """
    stem = Path(filename).stem  # 去掉扩展名
    if stem.lower().startswith("true"):
        return True
    if stem.lower().startswith("false"):
        return False
    return None


# ─────────────────────────────────────────────
# 指标计算
# ─────────────────────────────────────────────
def compute_metrics(records: list[dict]) -> dict:
    """
    计算混淆矩阵及四项指标。
    正例定义：有摄像头（predicted=True / label=True）
    """
    TP = FP = TN = FN = 0
    skipped   = []
    evaluated = []

    for r in records:
        label = get_ground_truth(r["filename"])
        if label is None:
            skipped.append(r["filename"])
            continue

        pred = r["predicted"]
        evaluated.append({**r, "label": label})

        if label and pred:
            TP += 1
        elif not label and pred:
            FP += 1
        elif not label and not pred:
            TN += 1
        else:  # label and not pred
            FN += 1

    total     = TP + FP + TN + FN
    accuracy  = (TP + TN) / total        if total          else 0.0
    precision = TP / (TP + FP)           if (TP + FP)      else 0.0
    recall    = TP / (TP + FN)           if (TP + FN)      else 0.0
    f1        = (2 * precision * recall) / (precision + recall) \
                if (precision + recall)  else 0.0

    return {
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "total":     total,
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "skipped":   skipped,
        "evaluated": evaluated,
    }


# ─────────────────────────────────────────────
# 报告输出
# ─────────────────────────────────────────────
def print_report(m: dict):
    TP, FP, TN, FN = m["TP"], m["FP"], m["TN"], m["FN"]

    print()
    print("=" * 55)
    print("           检测模型评估报告")
    print("=" * 55)

    # 混淆矩阵
    print("\n【混淆矩阵】")
    print(f"  {'':20s}  预测:有摄像头  预测:无摄像头")
    print(f"  {'真实:有摄像头':20s}  {TP:>12d}  {FN:>12d}   (共 {TP+FN} 张)")
    print(f"  {'真实:无摄像头':20s}  {FP:>12d}  {TN:>12d}   (共 {FP+TN} 张)")

    # 四项指标
    print(f"\n【评估指标】（共评估 {m['total']} 张）")
    print(f"  准确率  Accuracy  : {m['accuracy']  * 100:.2f}%"
          f"  ({TP + TN}/{m['total']} 预测正确)")
    print(f"  精确率  Precision : {m['precision'] * 100:.2f}%"
          f"  ({TP}/{TP + FP} 预测为正例中真正有摄像头)")
    print(f"  召回率  Recall    : {m['recall']    * 100:.2f}%"
          f"  ({TP}/{TP + FN} 真实正例中被正确找到)")
    print(f"  F1 Score          : {m['f1']        * 100:.2f}%")

    # 错误样本
    fp_files = [r["filename"] for r in m["evaluated"] if not r["label"] and r["predicted"]]
    fn_files = [r["filename"] for r in m["evaluated"] if r["label"]     and not r["predicted"]]

    print(f"\n【误报 FP {len(fp_files)} 张】（真实无摄像头，但预测为有）")
    if fp_files:
        for f in fp_files:
            print(f"  - {f}")
    else:
        print("  （无）")

    print(f"\n【漏报 FN {len(fn_files)} 张】（真实有摄像头，但预测为无）")
    if fn_files:
        for f in fn_files:
            print(f"  - {f}")
    else:
        print("  （无）")

    if m["skipped"]:
        print(f"\n【跳过 {len(m['skipped'])} 条】（文件名不符合 True*/False* 规则）")
        for f in m["skipped"]:
            print(f"  - {f}")

    print("\n" + "=" * 55)


# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="从检测日志计算准确率/精确率/召回率/F1-Score"
    )
    parser.add_argument(
        "--log-file", "-l",
        required=True,
        help="camera_detector.py 生成的日志文件路径"
    )
    args = parser.parse_args()

    if not Path(args.log_file).exists():
        print(f"[ERROR] 日志文件不存在：{args.log_file}")
        sys.exit(1)

    print(f"正在解析日志：{args.log_file}")
    records = parse_log(args.log_file)

    if not records:
        print("[ERROR] 日志中未找到任何检测记录，请确认日志格式是否正确")
        sys.exit(1)

    print(f"共解析到 {len(records)} 条检测记录")

    metrics = compute_metrics(records)
    print_report(metrics)


if __name__ == "__main__":
    main()
