import os
import json
import shutil
import base64
import requests
from PIL import Image
import imagehash
import torch
import clip
import numpy as np
from tqdm import tqdm

# =========================
# 配置
# =========================
INPUT_DIR = "data/raw"
OUTPUT_DIR = "output"
RESULT_JSON = "result.json"

PHASH_THRESHOLD = 8
CLIP_THRESHOLD = 0.95

BASE_URL = "http://10.19.205.173:11434/v1/chat/completions"
MODEL_NAME = "qwen3-vl"

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 初始化 CLIP
# =========================
model, preprocess = clip.load("ViT-B/32", device=device)

# =========================
# 工具函数
# =========================
def get_all_images(folder):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(exts)
    ]

# =========================
# Step1: pHash
# =========================
def compute_phash(img_path):
    img = Image.open(img_path).convert("RGB")
    return imagehash.phash(img)

def phash_dedup(paths):
    unique = []
    hashes = []

    for p in tqdm(paths, desc="pHash去重"):
        try:
            h = compute_phash(p)
        except:
            continue

        dup = False
        for eh in hashes:
            if abs(h - eh) <= PHASH_THRESHOLD:
                dup = True
                break

        if not dup:
            unique.append(p)
            hashes.append(h)

    return unique

# =========================
# Step2: CLIP
# =========================
def get_clip_emb(img_path):
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(image)
        emb /= emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]

def clip_dedup(paths):
    unique = []
    embs = []

    for p in tqdm(paths, desc="CLIP去重"):
        try:
            emb = get_clip_emb(p)
        except:
            continue

        dup = False
        for e in embs:
            sim = np.dot(emb, e)
            if sim > CLIP_THRESHOLD:
                dup = True
                break

        if not dup:
            unique.append(p)
            embs.append(emb)

    return unique

# =========================
# Step3: Qwen3-VL 分类
# =========================
def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def classify_image(path):
    img_base64 = encode_image(path)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "你是一个严格分类器，只输出 phone / action_camera / none"
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "图中是否包含手机或运动相机？"},
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{img_base64}"
                    }
                ]
            }
        ],
        "temperature": 0
    }

    try:
        resp = requests.post(BASE_URL, json=payload, timeout=30)
        result = resp.json()
        text = result["choices"][0]["message"]["content"].strip().lower()

        if "phone" in text:
            return "phone"
        elif "action_camera" in text:
            return "action_camera"
        else:
            return "none"

    except Exception as e:
        print(f"分类失败: {path}")
        return "error"

# =========================
# Step4: 保存结果
# =========================
def save_result(img_path, label):
    dst_dir = os.path.join(OUTPUT_DIR, label)
    os.makedirs(dst_dir, exist_ok=True)

    shutil.copy(img_path, os.path.join(dst_dir, os.path.basename(img_path)))

# =========================
# 主流程
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("读取图片...")
    images = get_all_images(INPUT_DIR)

    # Step1
    images = phash_dedup(images)

    # Step2
    images = clip_dedup(images)

    # Step3
    results = []

    for img in tqdm(images, desc="Qwen分类"):
        label = classify_image(img)

        save_result(img, label)

        results.append({
            "image": img,
            "label": label
        })

    # Step4
    with open(RESULT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print("完成！")

# =========================
# 入口
# =========================
if __name__ == "__main__":
    main()