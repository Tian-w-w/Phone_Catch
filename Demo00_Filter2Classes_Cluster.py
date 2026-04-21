import os
import shutil
from PIL import Image
import imagehash
import torch
import clip
import numpy as np
from tqdm import tqdm
import hdbscan

# =========================
# 配置
# =========================
INPUT_DIR = "data/raw"
OUTPUT_DIR = "clusters_output"

PHASH_THRESHOLD = 8

# HDBSCAN参数（可调）
MIN_CLUSTER_SIZE = 10
MIN_SAMPLES = 5

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 初始化 CLIP
# =========================
model, preprocess = clip.load("ViT-B/32", device=device)

# =========================
# 获取图片
# =========================
def get_images(folder):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(exts)
    ]

# =========================
# Step1: pHash 去重
# =========================
def phash_dedup(paths):
    unique = []
    hashes = []

    for p in tqdm(paths, desc="pHash去重"):
        try:
            img = Image.open(p).convert("RGB")
            h = imagehash.phash(img)
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
# Step2: CLIP embedding
# =========================
def get_clip_embeddings(paths, batch_size=32):
    all_embeddings = []

    for i in tqdm(range(0, len(paths), batch_size), desc="CLIP编码"):
        batch_paths = paths[i:i+batch_size]
        images = []

        for p in batch_paths:
            try:
                img = preprocess(Image.open(p)).unsqueeze(0)
                images.append(img)
            except:
                continue

        if not images:
            continue

        images = torch.cat(images).to(device)

        with torch.no_grad():
            emb = model.encode_image(images)
            emb /= emb.norm(dim=-1, keepdim=True)

        all_embeddings.append(emb.cpu().numpy())

    return np.vstack(all_embeddings)

# =========================
# Step3: HDBSCAN 聚类
# =========================
def cluster_embeddings(embeddings):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        metric='euclidean'
    )
    labels = clusterer.fit_predict(embeddings)
    return labels

# =========================
# Step4: 保存分类结果
# =========================
def save_clusters(paths, labels):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cluster_map = {}

    for path, label in zip(paths, labels):
        if label == -1:
            cluster_name = "noise"
        else:
            cluster_name = f"类别{label+1}"

        if cluster_name not in cluster_map:
            cluster_map[cluster_name] = []
        cluster_map[cluster_name].append(path)

    # 保存图片
    for cluster_name, imgs in cluster_map.items():
        dst_dir = os.path.join(OUTPUT_DIR, cluster_name)
        os.makedirs(dst_dir, exist_ok=True)

        for img_path in imgs:
            try:
                shutil.copy(img_path, os.path.join(dst_dir, os.path.basename(img_path)))
            except:
                pass

    print(f"共生成 {len(cluster_map)} 个类别（含 noise）")

# =========================
# 主流程
# =========================
def main():
    print("读取图片...")
    paths = get_images(INPUT_DIR)

    # Step1 去重
    paths = phash_dedup(paths)

    # Step2 CLIP
    embeddings = get_clip_embeddings(paths)

    # Step3 聚类
    labels = cluster_embeddings(embeddings)

    # Step4 保存
    save_clusters(paths, labels)

    print("完成！")

# =========================
if __name__ == "__main__":
    main()