"""
摄像设备检测脚本
- 支持手机、运动相机等摄像设备的识别
- 基于 OpenAI 兼容接口（MaaS API）
- 内置限流重试、few-shot 策略、日志输出
"""

import os
import base64
import time
import logging
import json
from pathlib import Path
from datetime import datetime
from openai import OpenAI

# ─────────────────────────── 抑制 HTTP 请求日志 ───────────────────────────
# openai SDK 底层使用 httpx，会打印 "HTTP Request: POST ..." 这类日志，统一屏蔽
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# ─────────────────────────── 配置区 ───────────────────────────

API_KEY    = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"   # ← 替换为你的 API Key
BASE_URL   = "https://api.openai.com/v1"              # ← 替换为你的 MaaS 端点
MODEL      = "gpt-4o"                                 # ← 替换为你的视觉模型名称

# 输入：数据集根目录（支持 jpg/jpeg/png/bmp/webp）
DATASET_DIR = "./dataset"

# 输出：日志文件
LOG_FILE   = f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# 限流：每分钟最多调用次数
RATE_LIMIT_PER_MIN = 10
# 触发限流后等待时间（秒）
RATE_LIMIT_WAIT    = 65
# 非限流错误的最大重试次数
MAX_RETRIES        = 3
# 非限流错误的重试间隔（秒）
RETRY_WAIT         = 5

# ─────────────────────────── Few-shot 样本 ───────────────────────────
# 负样本图片路径（你已提供的两张办公室截图），用于教模型区分误报
# 如果你想在 few-shot 中加入正样本，同样添加到 POSITIVE_EXAMPLES 中
NEGATIVE_EXAMPLE_PATHS = [
    "/mnt/user-data/uploads/1776308101763_5cd54cf22064432efc82e89299d4044a.jpg",
    "/mnt/user-data/uploads/1776308106269_9b7d51f3002a7e6c62f8143dcb530546.jpg",
]

# ─────────────────────────── 日志配置 ───────────────────────────
# 控制台：所有级别统一白色（去除终端默认红色 ERROR/WARNING 着色）
# 文件：保留完整级别标签，便于后期排查

class PlainWhiteFormatter(logging.Formatter):
    """所有级别均以白色纯文本输出，避免终端 ANSI 红色干扰。"""
    WHITE  = "\033[97m"
    RESET  = "\033[0m"
    FMT    = "%(asctime)s [%(levelname)s] %(message)s"

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        return f"{self.WHITE}{msg}{self.RESET}"


_file_handler    = logging.FileHandler(LOG_FILE, encoding="utf-8")
_file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

_console_handler = logging.StreamHandler()
_console_handler.setFormatter(PlainWhiteFormatter("%(asctime)s [%(levelname)s] %(message)s"))

logging.basicConfig(level=logging.INFO, handlers=[_file_handler, _console_handler])
logger = logging.getLogger(__name__)

# ─────────────────────────── 工具函数 ───────────────────────────

def encode_image(image_path: str) -> tuple[str, str]:
    """将图片编码为 base64，返回 (base64字符串, mime_type)"""
    suffix = Path(image_path).suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png",  ".bmp": "image/bmp",
        ".webp": "image/webp",
    }
    mime = mime_map.get(suffix, "image/jpeg")
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8"), mime


def build_few_shot_messages(negative_paths: list[str]) -> list[dict]:
    """
    构建 few-shot 上下文：
    将负样本图片作为历史对话注入，让模型学会不对这类场景误报。
    """
    messages = []

    # System prompt
    messages.append({
        "role": "system",
        "content": (
            "你是一个专业的图像安全审查助手，负责判断图片中是否出现了手机或运动相机等摄像设备。\n\n"
            "【判定规则】\n"
            "1. 只要图片中出现以下任意一种设备，即判定为 YES：\n"
            "   - 手机（智能手机）\n"
            "   - 运动相机（如 GoPro、Insta360、大疆 Action 等）\n"
            "   - 其他手持摄像设备\n"
            "2. 手机判定细则：\n"
            "   - 若检测到摄像头模组，且满足以下至少一条其他特征，即判定为手机：\n"
            "     矩形屏幕、品牌 Logo、手机边框/机身、手机壳、正在被人手持操作\n"
            "   - 若未检测到摄像头模组，则需同时满足上述所有其他特征才能判定。\n"
            "3. 运动相机判定细则：\n"
            "   - 只要检测到摄像头模组/镜头组件，即可判定为 YES。\n"
            "4. 负样本说明（不应触发 YES 的场景）：\n"
            "   - 普通办公室场景：电脑、键盘、显示器、耳机线、文件夹等\n"
            "   - 手持非摄像设备（遥控器、硬盘、数据线等小型电子元件）\n"
            "   - 人脸本身、眼镜、耳机\n\n"
            "【输出格式】严格按以下 JSON 输出，不得有多余内容：\n"
            '{"result": "YES" 或 "NO", "confidence": 0~1 的浮点数, '
            '"evidence": ["证据1", "证据2", ...], "reasoning": "简短推理过程"}'
        ),
    })

    # Few-shot 负样本示例
    for idx, img_path in enumerate(negative_paths):
        if not os.path.exists(img_path):
            logger.warning(f"Few-shot 图片不存在，跳过: {img_path}")
            continue

        b64, mime = encode_image(img_path)
        # 用户侧：提供负样本图片
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"},
                },
                {"type": "text", "text": "请判断该图片中是否存在手机或运动相机等摄像设备？"},
            ],
        })
        # 助手侧：给出正确的负样本答案
        neg_answer = json.dumps(
            {
                "result": "NO",
                "confidence": 0.97,
                "evidence": ["图中为普通办公室场景", "手持物品为非摄像电子元件或摄像头配件，但无完整手机或相机"],
                "reasoning": "虽然图中人物手持小型电子设备，但未见完整手机屏幕、机身或运动相机镜头模组，不满足判定条件。",
            },
            ensure_ascii=False,
        )
        messages.append({"role": "assistant", "content": neg_answer})

    return messages


# ─────────────────────────── 限流调用器 ───────────────────────────

class RateLimitedClient:
    def __init__(self, api_key: str, base_url: str, limit_per_min: int):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.limit  = limit_per_min
        self._calls: list[float] = []   # 记录每次调用时间戳

    def _wait_if_needed(self):
        """如果最近 60 秒内调用次数已达上限，则等待"""
        now = time.time()
        # 清理 60 秒以前的记录
        self._calls = [t for t in self._calls if now - t < 60]
        if len(self._calls) >= self.limit:
            oldest = self._calls[0]
            wait   = 60 - (now - oldest) + 1   # 多等 1 秒缓冲
            logger.warning(f"达到速率限制（{self.limit} 次/分钟），等待 {wait:.1f} 秒...")
            time.sleep(wait)
            # 再次清理
            now = time.time()
            self._calls = [t for t in self._calls if now - t < 60]

    def chat_complete(self, messages: list[dict], **kwargs) -> str:
        """带限流 + 重试的 chat completion，返回 content 字符串"""
        for attempt in range(1, MAX_RETRIES + 2):   # +2 保证至少尝试一次
            self._wait_if_needed()
            try:
                resp = self.client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    max_tokens=512,
                    temperature=0,
                    **kwargs,
                )
                self._calls.append(time.time())
                return resp.choices[0].message.content.strip()

            except Exception as e:
                err_str = str(e).lower()
                # 识别限流错误
                if "rate limit" in err_str or "429" in err_str or "too many" in err_str:
                    logger.warning(f"API 限流错误，等待 {RATE_LIMIT_WAIT} 秒后重试... ({attempt})")
                    time.sleep(RATE_LIMIT_WAIT)
                elif attempt <= MAX_RETRIES:
                    logger.warning(f"API 调用失败: {e}，{RETRY_WAIT} 秒后重试... ({attempt}/{MAX_RETRIES})")
                    time.sleep(RETRY_WAIT)
                else:
                    logger.error(f"API 调用最终失败: {e}")
                    raise

        raise RuntimeError("超出最大重试次数")


# ─────────────────────────── 检测主逻辑 ───────────────────────────

def detect_image(client: RateLimitedClient, image_path: str, few_shot_messages: list[dict]) -> dict:
    """对单张图片执行检测，返回结构化结果"""
    b64, mime = encode_image(image_path)

    # 在 few-shot 历史后追加当前待检测图片
    messages = few_shot_messages + [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"},
                },
                {"type": "text", "text": "请判断该图片中是否存在手机或运动相机等摄像设备？"},
            ],
        }
    ]

    raw = client.chat_complete(messages)

    # 解析 JSON
    try:
        # 容错：模型有时会包裹在 ```json ... ```
        clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        result = json.loads(clean)
    except json.JSONDecodeError:
        logger.warning(f"JSON 解析失败，原始输出: {raw}")
        result = {"result": "PARSE_ERROR", "raw": raw}

    return result


def run_dataset(dataset_dir: str):
    """遍历数据集目录，对每张图片执行检测"""
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted(
        p for p in Path(dataset_dir).rglob("*") if p.suffix.lower() in supported
    )

    if not image_paths:
        logger.error(f"数据集目录中未找到图片: {dataset_dir}")
        return

    logger.info(f"共找到 {len(image_paths)} 张图片，开始检测...")

    # 初始化客户端 & few-shot
    api_client        = RateLimitedClient(API_KEY, BASE_URL, RATE_LIMIT_PER_MIN)
    few_shot_messages = build_few_shot_messages(NEGATIVE_EXAMPLE_PATHS)
    logger.info(f"Few-shot 构建完成，共注入 {len(NEGATIVE_EXAMPLE_PATHS)} 个负样本示例")

    summary    = {"YES": 0, "NO": 0, "ERROR": 0}
    img_times: list[float] = []          # 每张图片耗时记录
    total_start = time.time()            # 总计时开始

    for idx, img_path in enumerate(image_paths, 1):
        logger.info(f"[{idx}/{len(image_paths)}] 正在检测: {img_path}")
        img_start = time.time()          # 单张计时开始
        try:
            result     = detect_image(api_client, str(img_path), few_shot_messages)
            img_cost   = time.time() - img_start
            img_times.append(img_cost)

            verdict    = result.get("result", "ERROR")
            confidence = result.get("confidence", "N/A")
            evidence   = result.get("evidence", [])
            reasoning  = result.get("reasoning", "")

            if verdict == "YES":
                summary["YES"] += 1
            elif verdict == "NO":
                summary["NO"] += 1
            else:
                summary["ERROR"] += 1

            logger.info(
                f"  结果: {verdict} | 置信度: {confidence} | 耗时: {img_cost:.2f}s\n"
                f"  证据: {'; '.join(evidence)}\n"
                f"  推理: {reasoning}"
            )

        except Exception as e:
            img_cost = time.time() - img_start
            img_times.append(img_cost)
            summary["ERROR"] += 1
            logger.error(f"  检测异常: {e} | 耗时: {img_cost:.2f}s")

    # ── 汇总统计 ──
    total_cost = time.time() - total_start
    avg_cost   = (sum(img_times) / len(img_times)) if img_times else 0
    max_cost   = max(img_times) if img_times else 0
    min_cost   = min(img_times) if img_times else 0

    logger.info(
        f"\n{'='*55}\n"
        f"检测完成！\n"
        f"  检测到摄像设备（YES）: {summary['YES']}\n"
        f"  未检测到（NO）      : {summary['NO']}\n"
        f"  异常/解析错误       : {summary['ERROR']}\n"
        f"\n── 时间开销分析 ──\n"
        f"  总耗时  : {total_cost:.2f}s\n"
        f"  平均耗时: {avg_cost:.2f}s / 张\n"
        f"  最快    : {min_cost:.2f}s\n"
        f"  最慢    : {max_cost:.2f}s\n"
        f"\n日志已保存至: {LOG_FILE}\n"
        f"{'='*55}"
    )


# ─────────────────────────── 入口 ───────────────────────────

if __name__ == "__main__":
    run_dataset(DATASET_DIR)
