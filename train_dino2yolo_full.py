import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics import YOLO


# ==========================================
# 第一部分：特征对齐与蒸馏 Loss 模块
# ==========================================

class FeatureProjector(nn.Module):
    """
    将 YOLO 的特征通道 (例如 512) 映射到 DINOv2 的通道 (768)
    并进行空间维度缩放对齐
    """

    def __init__(self, yolo_channels, dino_channels=768):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Conv2d(yolo_channels, dino_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(dino_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, yolo_feat, target_shape):
        x = self.projector(yolo_feat)
        # 强制将 YOLO 特征图缩放到与 DINO 目标特征图一致的大小
        x = F.interpolate(x, size=target_shape, mode='bilinear', align_corners=False)
        return x


def cosine_loss(yolo_feat, dino_feat):
    """余弦相似度损失：让特征在方向上趋同"""
    # yolo_feat, dino_feat: [B, C, H, W]
    sim = F.cosine_similarity(yolo_feat, dino_feat, dim=1)
    return (1.0 - sim).mean()


# ==========================================
# 第二部分：自定义训练器逻辑 (核心)
# ==========================================

class BvnDistillTrainer(DetectionTrainer):
    def __init__(self, overrides=None, _callbacks=None):
        super().__init__(overrides=overrides, _callbacks=_callbacks)
        self.yolo_feats = None  # 暂存 Hook 抓取的特征
        self.dino_model = None

    def build_model(self):
        """初始化 YOLO 模型并挂载辅助模块"""
        model = super().build_model()

        # 1. 给 YOLO 的 Detect 层 (最后一层) 注册 Hook，抓取输入特征 P3, P4, P5
        model.model[-1].register_forward_hook(self._feature_hook)

        # 2. 初始化 Projector (针对 YOLOv8s 的 P5 层，通道是 512)
        # 如果你后续想换 YOLOv8n，这里要把 512 改成 256
        model.projector = FeatureProjector(yolo_channels=512, dino_channels=768)

        return model

    def _feature_hook(self, module, input, output):
        """Hook 函数：拦截前向传播时的特征图"""
        self.yolo_feats = input[0]  # 这里拿到的是 [P3, P4, P5] 的列表

    def train_step(self):
        """单步训练逻辑：计算双重 Loss 并回传梯度"""
        self.optimizer.zero_grad()
        device = self.batch['img'].device

        # 延迟加载 Teacher，节省启动时的显存开销
        if self.dino_model is None:
            print(">>> 正在加载 Teacher: DINOv2-ViT-Base...")
            self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device).eval()
            for p in self.dino_model.parameters():
                p.requires_grad = False
            self.model.projector.to(device)

        # 使用 AMP 混合精度，加快 24G 显卡训练速度
        with torch.cuda.amp.autocast(self.amp):
            # 1. 运行 YOLO 前向传播 (触发 Hook)
            # loss_standard 包含原生的 Box, Cls, DFL 损失
            loss_standard, loss_items = self.model(self.batch)

            # 2. 提取 Teacher (DINO) 特征
            images = self.batch['img']
            with torch.no_grad():
                # DINOv2 输出 Patch Tokens: [B, N, 768]
                dino_out = self.dino_model.forward_features(images)
                patch_tokens = dino_out['x_norm_patchtokens']
                B, N, C = patch_tokens.shape
                H, W = images.shape[2:]
                h_feat, w_feat = H // 14, W // 14  # DINO 默认 stride=14
                dino_target = patch_tokens.reshape(B, h_feat, w_feat, C).permute(0, 3, 1, 2)

            # 3. 计算蒸馏 Loss (针对 P5 特征)
            yolo_p5 = self.yolo_feats[2]  # 提取 YOLO 最深层的语义特征
            yolo_aligned = self.model.projector(yolo_p5, dino_target.shape[2:])
            loss_distill = cosine_loss(yolo_aligned, dino_target)

            # 4. 权重融合 (alpha=2.0 是一个稳健的经验值)
            alpha = 2.0
            total_loss = loss_standard + alpha * loss_distill

        # 反向传播：梯度将通过 total_loss 流向 YOLO Backbone
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # 更新进度条显示的 Loss
        self.loss = total_loss
        self.loss_items = loss_items


# ==========================================
# 第三部分：正式训练入口
# ==========================================

if __name__ == '__main__':
    # 配置参数
    # 24G 显存配置参考：YOLOv8s, Batch=32, imgsz=640
    train_args = {
        "model": "yolov8s.pt",  # 官方预训练权重作为起点
        "data": "bvn.yaml",  # 你的数据集配置
        "epochs": 100,  # 全量训练通常 100-200 轮
        "imgsz": 640,  # 输入尺寸
        "batch": 32,  # 24G 显存可以尝试 32 甚至 64
        "device": 0,  # 使用第一块显卡
        "workers": 8,  # 数据加载线程
        "name": "yolov8s_bvn_distill",  # 训练结果文件夹名称
        "exist_ok": True,  # 如果文件夹存在则覆盖
        "amp": True  # 开启混合精度
    }

    print("---  启动拍摄设备识别模型蒸馏训练 ---")
    trainer = BvnDistillTrainer(overrides=train_args)
    trainer.train()

    print("---  训练完成！模型已自动保存至 runs/detect/yolov8s_bvn_distill/weights/best.pt ---")