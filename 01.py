"""
防偷拍检测系统 - Web API 服务端
提供 HTTP 接口，可配合前端或移动端使用
"""

import os
import base64
import json
import io
import re
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI
from PIL import Image

# 依赖: pip install flask flask-cors openai pillow

app = Flask(__name__)
CORS(app)

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "YOUR_DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-vl-plus"

client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)

SYSTEM_PROMPT = """你是一个专业的安防视觉分析系统，负责检测画面中是否存在手机偷拍行为。
偷拍特征：手机摄像头朝向他人隐私部位、隐藏设备拍摄、异常持机角度等。
请严格以 JSON 格式返回分析结果。"""

DETECTION_PROMPT = """分析此图片是否存在手机偷拍行为，返回严格 JSON（无其他文字）：
{
  "is_spying": true或false,
  "confidence": 0-100整数,
  "risk_level": "HIGH"/"MEDIUM"/"LOW"/"NONE",
  "evidence": ["证据1"],
  "description": "描述（50字内）",
  "suggestion": "建议措施"
}"""


def analyze_image(image_data: bytes) -> dict:
    """调用 qwen3-vl-plus 分析图片"""
    b64 = base64.b64encode(image_data).decode("utf-8")

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": DETECTION_PROMPT}
                ]
            }
        ],
        max_tokens=512,
        temperature=0.1
    )

    raw = response.choices[0].message.content.strip()
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            result["timestamp"] = datetime.now().isoformat()
            return result
        except Exception:
            pass

    return {
        "is_spying": False, "confidence": 0,
        "risk_level": "NONE", "evidence": [],
        "description": "解析失败", "suggestion": "重试",
        "timestamp": datetime.now().isoformat()
    }


@app.route("/detect", methods=["POST"])
def detect():
    """
    POST /detect
    Body: multipart/form-data with 'image' file
       OR JSON { "image_base64": "..." }
    """
    try:
        if request.files.get("image"):
            img_file = request.files["image"]
            img_bytes = img_file.read()
            # 压缩处理
            pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            pil.thumbnail((640, 480))
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=85)
            img_bytes = buf.getvalue()

        elif request.json and request.json.get("image_base64"):
            img_bytes = base64.b64decode(request.json["image_base64"])

        else:
            return jsonify({"error": "请提供 image 文件或 image_base64 字段"}), 400

        result = analyze_image(img_bytes)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME,
                    "time": datetime.now().isoformat()})


if __name__ == "__main__":
    print("🛡️  防偷拍检测 API 服务启动中...")
    print(f"   模型: {MODEL_NAME}")
    print(f"   接口: POST http://localhost:5000/detect")
    app.run(host="0.0.0.0", port=5000, debug=False)
