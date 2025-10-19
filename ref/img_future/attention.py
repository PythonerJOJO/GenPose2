import torch
import torch.nn as nn
import torch.nn.functional as F


class IndustrialAttentionFusion(nn.Module):
    def __init__(self, dim=384, num_patches=196, patch_size=16):
        super().__init__()
        self.dim = dim
        self.num_patches = num_patches  # 必须与实际patch数量一致（如14x14=196）
        self.patch_size = patch_size

        # 1. 层间注意力
        self.layer_attn = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, 1)
        )

        # 2. 空间几何注意力：修复相对位置索引范围
        h = w = int(num_patches**0.5)
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(h),
                torch.arange(w),
                indexing="ij",  # 明确索引模式，消除警告
            ),
            dim=-1,
        )
        coords = coords.reshape(-1, 2)  # (num_patches, 2)

        # 计算相对坐标的最大范围（用于设置embedding容量）
        max_rel = 2 * (h - 1)  # 相对坐标范围：-(h-1) ~ h-1，共2h-1种可能
        self.rel_pos_emb = nn.Embedding(max_rel * max_rel, dim // 4)  # 足够大的容量

        # 预计算相对坐标（偏移为非负索引）
        self.rel_coords = coords.unsqueeze(0) - coords.unsqueeze(
            1
        )  # (num_patches, num_patches, 2)
        self.rel_coords = self.rel_coords + (h - 1)  # 偏移至0~2h-2

        # 3. 边缘增强分支：确保卷积层输入兼容
        self.edge_guide = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, features) -> torch.Tensor:
        bs, num_patches, dim = features[0].shape
        h = w = int(num_patches**0.5)

        # 检查空间维度是否合法（至少为1x1，避免卷积报错）
        if h < 1 or w < 1:
            raise ValueError(
                f"Invalid spatial size: {h}x{w}, num_patches={num_patches}"
            )

        # 步骤1：层间注意力融合
        feats_stacked = torch.stack(features, dim=1)
        layer_attn_weights = self.layer_attn(feats_stacked.transpose(1, 2)).transpose(
            1, 2
        )
        layer_attn_weights = F.softmax(layer_attn_weights, dim=1)
        fused_feats = (feats_stacked * layer_attn_weights).sum(dim=1)

        # 步骤2：空间几何注意力（修复索引越界）
        if h > 1 and w > 1:  # 仅当空间尺寸足够时计算（避免1x1时无意义）
            # 计算相对位置索引（确保在embedding容量内）
            rel_pos_idx = (
                self.rel_coords[..., 0] * (2 * (h - 1) + 1) + self.rel_coords[..., 1]
            )
            rel_pos_idx = rel_pos_idx.clamp(
                0, self.rel_pos_emb.num_embeddings - 1
            )  # 截断到有效范围
            rel_pos_emb = self.rel_pos_emb(rel_pos_idx.to(fused_feats.device))

            # 空间注意力计算
            feat_geo = fused_feats[:, :, dim // 4 :]
            attn_spatial = torch.matmul(feat_geo, feat_geo.transpose(1, 2))
            attn_spatial = attn_spatial * rel_pos_emb.sum(dim=-1)
            attn_spatial = F.softmax(attn_spatial, dim=-1)
            geo_enhanced_feats = torch.matmul(attn_spatial, fused_feats)
        else:
            geo_enhanced_feats = 0  # 1x1时无需空间注意力

        # 步骤3：边缘增强（修复卷积输入形状）
        spatial_feat = fused_feats.transpose(1, 2).view(bs, dim, h, w)  # 确保形状正确
        edge_weight = self.edge_guide(spatial_feat).view(bs, 1, dim // 4)
        edge_enhanced_feats = fused_feats * torch.cat(
            [edge_weight] * 4, dim=-1
        )  # 更安全的广播方式

        # 最终融合
        final_feats = fused_feats + 0.2 * geo_enhanced_feats + 0.1 * edge_enhanced_feats
        return final_feats
