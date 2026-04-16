"""
日志统计分析脚本
- 从检测日志中解析每张图片的文件名与模型结果
- 根据文件名前缀（True/False）推断人工标签
- 计算准确率、精确率、召回率、F1 值
- 将分类错误的样本输出到控制台
"""

import re
import sys
from pathlib import Path

# ─────────────────────────── 配置 ───────────────────────────
# 默认读取最新的日志文件；也可在命令行传入日志路径
#   python analyze_log.py detection_20260416_112556.log
LOG_PATH = sys.argv[1] if len(sys.argv) > 1 else None

# ANSI 颜色
WHITE  = "\033[97m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def c(text, color): return f"{color}{text}{RESET}"

# ─────────────────────────── 自动定位日志 ───────────────────────────

def find_latest_log() -> Path:
    logs = sorted(Path(".").glob("detection_*.log"), key=lambda p: p.stat().st_mtime)
    if not logs:
        print(c("✗ 当前目录未找到 detection_*.log 文件，请手动指定路径。", RED))
        sys.exit(1)
    return logs[-1]

# ─────────────────────────── 解析日志 ───────────────────────────

# 匹配 "[N/543] 正在检测: path\Filename.jpg"
RE_FILE   = re.compile(r"\[\d+/\d+\] 正在检测: .+?([^/\\]+\.(?:jpg|jpeg|png|bmp|webp))", re.IGNORECASE)
# 匹配 "结果: YES" 或 "结果: NO"
RE_RESULT = re.compile(r"结果:\s*(YES|NO)", re.IGNORECASE)

def parse_log(log_path: Path) -> list[dict]:
    """
    逐行扫描日志，将每张图片的文件名与模型结果配对。
    返回列表：[{"filename": str, "pred": bool, "gt": bool}, ...]
    """
    records = []
    current_file = None

    with open(log_path, encoding="utf-8") as f:
        for line in f:
            fm = RE_FILE.search(line)
            if fm:
                current_file = fm.group(1)
                continue

            rm = RE_RESULT.search(line)
            if rm and current_file:
                pred = rm.group(1).upper() == "YES"

                name_lower = current_file.lower()
                if name_lower.startswith("true"):
                    gt = True
                elif name_lower.startswith("false"):
                    gt = False
                else:
                    # 无法从文件名推断标签，跳过
                    current_file = None
                    continue

                records.append({
                    "filename": current_file,
                    "pred": pred,
                    "gt": gt,
                })
                current_file = None   # 等待下一条

    return records

# ─────────────────────────── 指标计算 ───────────────────────────

def compute_metrics(records: list[dict]) -> dict:
    TP = sum(1 for r in records if     r["gt"] and     r["pred"])
    TN = sum(1 for r in records if not r["gt"] and not r["pred"])
    FP = sum(1 for r in records if not r["gt"] and     r["pred"])
    FN = sum(1 for r in records if     r["gt"] and not r["pred"])

    total    = TP + TN + FP + FN
    accuracy  = (TP + TN) / total        if total          else 0.0
    precision = TP / (TP + FP)           if (TP + FP)      else 0.0
    recall    = TP / (TP + FN)           if (TP + FN)      else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return dict(TP=TP, TN=TN, FP=FP, FN=FN,
                total=total, accuracy=accuracy,
                precision=precision, recall=recall, f1=f1)

# ─────────────────────────── 输出 ───────────────────────────

def print_report(records: list[dict], metrics: dict, log_path: Path):
    sep  = c("─" * 55, WHITE)
    sep2 = c("═" * 55, WHITE)

    print(f"\n{sep2}")
    print(c(f"  日志分析报告   {log_path.name}", BOLD + WHITE))
    print(sep2)

    # ── 混淆矩阵 ──
    print(c("\n  混淆矩阵", CYAN + BOLD))
    print(c(f"  {'':12} {'预测 YES':>10} {'预测 NO':>10}", WHITE))
    print(c(f"  {'标签 True':12} {metrics['TP']:>10} {metrics['FN']:>10}", WHITE))
    print(c(f"  {'标签 False':12} {metrics['FP']:>10} {metrics['TN']:>10}", WHITE))

    # ── 指标 ──
    print(c("\n  评估指标", CYAN + BOLD))
    total = metrics['total']
    print(c(f"  样本总数  : {total}  (True={metrics['TP']+metrics['FN']}  False={metrics['TN']+metrics['FP']})", WHITE))
    print(c(f"  准确率    : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)", GREEN))
    print(c(f"  精确率    : {metrics['precision']:.4f}  ({metrics['precision']*100:.2f}%)", GREEN))
    print(c(f"  召回率    : {metrics['recall']:.4f}  ({metrics['recall']*100:.2f}%)", GREEN))
    print(c(f"  F1 值     : {metrics['f1']:.4f}  ({metrics['f1']*100:.2f}%)", GREEN))

    # ── 错误样本 ──
    errors = [r for r in records if r["gt"] != r["pred"]]
    print(c(f"\n  分类错误样本（共 {len(errors)} 张）", YELLOW + BOLD))
    if not errors:
        print(c("  ✓ 无错误样本！", GREEN))
    else:
        print(sep)
        fp_list = [r for r in errors if not r["gt"] and r["pred"]]   # 误报
        fn_list = [r for r in errors if     r["gt"] and not r["pred"]]  # 漏报

        if fp_list:
            print(c(f"\n  ▶ 误报（False Positive）—— 标签为 False，模型判 YES  共 {len(fp_list)} 张", RED))
            for r in fp_list:
                print(c(f"    ✗  {r['filename']}", RED))

        if fn_list:
            print(c(f"\n  ▶ 漏报（False Negative）—— 标签为 True，模型判 NO   共 {len(fn_list)} 张", YELLOW))
            for r in fn_list:
                print(c(f"    ✗  {r['filename']}", YELLOW))

    print(f"\n{sep2}\n")

# ─────────────────────────── 入口 ───────────────────────────

if __name__ == "__main__":
    log_path = Path(LOG_PATH) if LOG_PATH else find_latest_log()
    if not log_path.exists():
        print(c(f"✗ 日志文件不存在: {log_path}", RED))
        sys.exit(1)

    print(c(f"正在解析日志: {log_path}", WHITE))
    records = parse_log(log_path)

    if not records:
        print(c("✗ 未解析到任何有效记录，请确认日志格式是否匹配。", RED))
        sys.exit(1)

    metrics = compute_metrics(records)
    print_report(records, metrics, log_path)
