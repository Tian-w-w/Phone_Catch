 # -*- coding: utf-8 -*-
  """
  maas_device_detector.py
  需求：
  1) 调用 MaaS 视觉模型
  2) API 配置写死在代码中
  3) 扫描文件夹图片样本
  4) 判断图片中是否有 手机 或 运动相机
  5) 输出结果到日志（JSON Lines）
  """

  import base64
  import json
  import mimetypes
  import re
  import time
  from datetime import datetime
  from pathlib import Path

  import requests

  # =========================
  # 1) 硬编码配置（按需修改）
  # =========================
  API_URL = "https://your-maas-endpoint/v1/chat/completions"  # 例如 OpenAI兼容网关地址
  API_KEY = "your_api_key_here"
  MODEL = "your_vision_model_name"

  INPUT_DIR = r"./samples"          # 样本图片目录
  LOG_FILE = r"./detect_result.log" # 输出日志(JSON Lines)

  # 可识别的图片后缀
  IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

  # 模型提示词：强制返回 JSON，便于解析和审计
  SYSTEM_PROMPT = (
      "你是图像审核二分类器。"
      "任务：判断图片中是否出现“手机”或“运动相机（如 GoPro、DJI Action、insta360 等可用于拍摄的视频设备）”。"
      "只输出 JSON，不要输出任何额外文本。"
      'JSON 格式固定为：{"has_device": true/false, "device_types": ["phone"|"action_camera"], "confidence": 0~1,
  "reason": "不超过20字"}'
  )

  USER_PROMPT = "请判断这张图中是否包含手机或运动相机。"


  def image_to_data_url(image_path: Path) -> str:
      mime, _ = mimetypes.guess_type(str(image_path))
      if not mime:
          mime = "application/octet-stream"
      b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
      return f"data:{mime};base64,{b64}"


  def extract_json(text: str) -> dict:
      """
      尝试从模型返回中提取 JSON。
      支持：
      - 纯 JSON
      - ```json ... ``` 包裹
      """
      text = text.strip()

      # 去掉 markdown 代码块
      fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S | re.I)
      if fence_match:
          text = fence_match.group(1).strip()

      # 直接解析
      try:
          obj = json.loads(text)
          if isinstance(obj, dict):
              return obj
      except json.JSONDecodeError:
          pass

      # 再尝试抽取第一个 {...}
      brace_match = re.search(r"\{.*\}", text, re.S)
      if brace_match:
          obj = json.loads(brace_match.group(0))
          if isinstance(obj, dict):
              return obj

      raise ValueError(f"无法解析模型返回为JSON: {text[:200]}")


  def call_maas_vision(image_path: Path, timeout: int = 60) -> dict:
      data_url = image_to_data_url(image_path)

      headers = {
          "Authorization": f"Bearer {API_KEY}",
          "Content-Type": "application/json",
      }

      payload = {
          "model": MODEL,
          "temperature": 0,
          "max_tokens": 200,
          "messages": [
              {"role": "system", "content": SYSTEM_PROMPT},
              {
                  "role": "user",
                  "content": [
                      {"type": "text", "text": USER_PROMPT},
                      {"type": "image_url", "image_url": {"url": data_url}},
                  ],
              },
          ],
      }

      resp = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
      resp.raise_for_status()
      result = resp.json()

      # OpenAI 兼容结构：choices[0].message.content
      content = result["choices"][0]["message"]["content"]

      # 某些网关可能返回 content 为 list，这里统一转字符串
      if isinstance(content, list):
          content = "".join(
              x.get("text", "") if isinstance(x, dict) else str(x) for x in content
          )

      parsed = extract_json(content)

      # 兜底字段规范化
      has_device = bool(parsed.get("has_device", False))
      device_types = parsed.get("device_types", [])
      if not isinstance(device_types, list):
          device_types = []
      confidence = parsed.get("confidence", 0)
      reason = str(parsed.get("reason", ""))

      return {
          "has_device": has_device,
          "device_types": device_types,
          "confidence": confidence,
          "reason": reason,
          "raw_content": content,  # 审计保留原始输出
      }


  def append_log(record: dict):
      with open(LOG_FILE, "a", encoding="utf-8") as f:
          f.write(json.dumps(record, ensure_ascii=False) + "\n")


  def iter_images(folder: Path):
      for p in folder.rglob("*"):
          if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
              yield p


  def main():
      input_dir = Path(INPUT_DIR)
      if not input_dir.exists():
          raise FileNotFoundError(f"输入目录不存在: {input_dir.resolve()}")

      # 写入任务开始标记
      append_log({
          "event": "start",
          "time": datetime.now().isoformat(timespec="seconds"),
          "input_dir": str(input_dir.resolve()),
          "model": MODEL,
      })

      total = 0
      ok = 0
      has_device_count = 0

      for img_path in iter_images(input_dir):
          total += 1
          t0 = time.time()

          try:
              result = call_maas_vision(img_path)
              cost_ms = int((time.time() - t0) * 1000)
              ok += 1
              if result["has_device"]:
                  has_device_count += 1

              record = {
                  "event": "result",
                  "time": datetime.now().isoformat(timespec="seconds"),
                  "file": str(img_path.resolve()),
                  "has_device": result["has_device"],
                  "device_types": result["device_types"],
                  "confidence": result["confidence"],
                  "reason": result["reason"],
                  "latency_ms": cost_ms,
                  "raw_content": result["raw_content"],
              }
              append_log(record)
              print(f"[OK] {img_path.name} -> has_device={result['has_device']} types={result['device_types']}")

          except Exception as e:
              record = {
                  "event": "error",
                  "time": datetime.now().isoformat(timespec="seconds"),
                  "file": str(img_path.resolve()),
                  "error": str(e),
              }
              append_log(record)
              print(f"[ERR] {img_path.name} -> {e}")

      # 写入任务结束标记
      append_log({
          "event": "finish",
          "time": datetime.now().isoformat(timespec="seconds"),
          "total": total,
          "success": ok,
          "has_device_count": has_device_count,
      })

      print(f"\n完成：total={total}, success={ok}, has_device={has_device_count}")
      print(f"日志文件：{Path(LOG_FILE).resolve()}")


  if __name__ == "__main__":
      main()