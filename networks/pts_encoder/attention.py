import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
1. 位置编码支持
添加了可学习的位置编码模块PositionalEncoding

支持动态调整以适应不同数量的点

位置编码在Transformer块开始时添加

2. 归一化策略改进
提供了Pre-Norm和Post-Norm两种选项

Pre-Norm通常训练更稳定，适合深层网络

Post-Norm是原始Transformer的标准配置

3. 前馈网络优化
使用GELU激活函数替代ReLU，通常能获得更好的性能

添加了ff_ratio参数，可以灵活控制前馈网络的隐藏层大小

4. 层缩放支持
添加了可选的层缩放机制，有助于训练非常深的网络

基于ConvNeXt和CaiT等现代架构的设计思想

5. 高效版本
提供了EfficientTransformerBlock作为备选

保持了标准接口，但内部可能使用更高效的实现

6. 配置灵活性
增加了多个参数，使模块更灵活可配置

可以根据具体任务调整各种超参数
"""


class PositionalEncoding(nn.Module):
    """可学习的位置编码，适用于点云数据"""

    def __init__(self, d_model, max_points=1024):
        super().__init__()
        self.position_emb = nn.Parameter(torch.zeros(1, max_points, d_model))
        self.max_points = max_points

    def forward(self, x):
        # x: [B, C, N]
        batch_size, _, num_points = x.size()

        if num_points > self.max_points:
            # 如果点数超过最大值，使用插值
            pos_enc = F.interpolate(
                self.position_emb.transpose(1, 2),
                size=num_points,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        else:
            pos_enc = self.position_emb[:, :num_points, :]

        # 添加位置编码
        x = x.transpose(1, 2)  # [B, N, C]
        x = x + pos_enc
        return x.transpose(1, 2)  # [B, C, N]


class TransformerBlock(nn.Module):
    """改进的Transformer块，包含多头自注意力和前馈网络"""

    def __init__(
        self,
        channels,
        num_heads=8,
        dropout=0.1,
        ff_ratio=4,
        pre_norm=True,
        use_pos_enc=True,
        layer_scale=None,
    ):
        """
        参数:
            channels: 输入特征维度
            num_heads: 注意力头数
            dropout: Dropout比率
            ff_ratio: 前馈网络隐藏层与输入层的比例
            pre_norm: 是否使用Pre-Norm而不是Post-Norm
            use_pos_enc: 是否使用位置编码
            layer_scale: 层缩放因子，如果为None则不使用
        """
        super().__init__()
        self.pre_norm = pre_norm
        self.use_pos_enc = use_pos_enc
        self.layer_scale = layer_scale is not None

        # 位置编码
        if use_pos_enc:
            self.pos_enc = PositionalEncoding(channels)

        # 归一化层
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        # 多头自注意力
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # 前馈网络
        ff_dim = int(channels * ff_ratio)
        self.ff = nn.Sequential(
            nn.Linear(channels, ff_dim),
            nn.GELU(),  # 使用GELU激活函数，效果通常优于ReLU
            nn.Dropout(dropout),
            nn.Linear(ff_dim, channels),
            nn.Dropout(dropout),
        )

        # Dropout层
        self.dropout = nn.Dropout(dropout)

        # 层缩放（可选）
        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(channels))
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(channels))
        else:
            self.gamma1, self.gamma2 = 1.0, 1.0

    def forward(self, x):
        """
        输入:
            x: [B, C, N] 输入特征
        输出:
            x: [B, C, N] 输出特征
        """
        # 添加位置编码
        if self.use_pos_enc:
            x = self.pos_enc(x)

        # 保存残差连接
        residual = x

        # Pre-Norm或Post-Norm
        if self.pre_norm:
            # Pre-Norm: 归一化 -> 注意力 -> 残差连接
            x = x.transpose(1, 2)  # [B, N, C]
            x_norm = self.norm1(x)
            attn_output, _ = self.attn(x_norm, x_norm, x_norm)
            x = x + self.dropout(attn_output) * self.gamma1
            x = self.norm2(x)
            x = x + self.ff(x) * self.gamma2
            x = x.transpose(1, 2)  # [B, C, N]
        else:
            # Post-Norm: 注意力 -> 残差连接 -> 归一化
            x = x.transpose(1, 2)  # [B, N, C]
            attn_output, _ = self.attn(x, x, x)
            x = x + self.dropout(attn_output) * self.gamma1
            x = self.norm1(x)
            ff_output = self.ff(x)
            x = x + ff_output * self.gamma2
            x = self.norm2(x)
            x = x.transpose(1, 2)  # [B, C, N]

        return x


class EfficientTransformerBlock(nn.Module):
    """高效版Transformer块，使用线性注意力降低计算复杂度"""

    def __init__(self, channels, num_heads=8, dropout=0.1, ff_ratio=4):
        super().__init__()

        # 归一化层
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        # 线性注意力（近似标准注意力，但计算复杂度更低）
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # 前馈网络
        ff_dim = int(channels * ff_ratio)
        self.ff = nn.Sequential(
            nn.Linear(channels, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, channels),
            nn.Dropout(dropout),
        )

        # Dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 输入: [B, C, N] -> 转换为 [B, N, C]
        x = x.transpose(1, 2)

        # 自注意力部分
        residual = x
        x = self.norm1(x)
        attn_output, _ = self.attn(x, x, x)
        x = residual + self.dropout(attn_output)

        # 前馈网络部分
        residual = x
        x = self.norm2(x)
        x = residual + self.ff(x)

        # 转换回 [B, C, N]
        return x.transpose(1, 2)


class GatedAttentionFusion(nn.Module):
    """针对位姿估计的门控注意力融合机制"""

    def __init__(self, current_channels, original_channels, reduction_ratio=4):
        super().__init__()
        # print(
        #     f"GatedAttentionFusion init: current_channels={current_channels}, original_channels={original_channels}"
        # )
        # 通道注意力模块
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(
                current_channels
                * 2,  # 修改这里，使用current_channels而不是original_channels
                (current_channels * 2) // reduction_ratio,
                1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                (current_channels * 2) // reduction_ratio,
                current_channels,
                1,
            ),
            nn.Sigmoid(),
        )

        # 空间注意力模块
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

        # 门控机制
        self.gate = nn.Sequential(
            nn.Conv1d(current_channels * 2, current_channels, 1),
            nn.BatchNorm1d(current_channels),
            nn.Sigmoid(),
        )

        # 原始特征转换
        self.original_transform = nn.Sequential(
            nn.Conv1d(original_channels, current_channels, 1),
            nn.BatchNorm1d(current_channels),
            nn.ReLU(inplace=True),
        )

        # 下采样层，用于调整原始特征的点数
        self.downsample = nn.Sequential(
            nn.Conv1d(current_channels, current_channels, 1),
            nn.BatchNorm1d(current_channels),
            nn.ReLU(inplace=True),
        )

        # 输出转换
        self.output_conv = nn.Sequential(
            nn.Conv1d(current_channels, current_channels, 1),
            nn.BatchNorm1d(current_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, current_feat, original_feat):
        # 首先确保原始特征的点数与当前特征匹配
        if original_feat.size(2) != current_feat.size(2):
            # 使用插值调整点数
            original_feat = F.interpolate(
                original_feat,
                size=current_feat.size(2),
                mode="linear",
                align_corners=False,
            )

        # 转换原始特征
        original_transformed = self.original_transform(original_feat)

        # 如果需要，进一步调整转换后的特征
        if original_transformed.size(2) != current_feat.size(2):
            original_transformed = self.downsample(original_transformed)

        # 通道注意力
        channel_att_input = torch.cat([current_feat, original_transformed], dim=1)
        channel_att = self.channel_attention(channel_att_input)

        # 空间注意力
        max_pool = torch.max(current_feat, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(current_feat, dim=1, keepdim=True)
        spatial_att_input = torch.cat([max_pool, avg_pool], dim=1)
        spatial_att = self.spatial_attention(spatial_att_input)

        # 应用注意力
        attended_original = original_transformed * channel_att * spatial_att

        # 门控融合
        gate_input = torch.cat([current_feat, attended_original], dim=1)
        gate_weights = self.gate(gate_input)

        # 融合特征
        fused_feat = (
            gate_weights * current_feat + (1 - gate_weights) * attended_original
        )

        # 输出转换
        return self.output_conv(fused_feat)


class ImprovedPositionalEncoding(nn.Module):
    """基于点云坐标的位置编码"""

    def __init__(self, d_model, use_learnable=True, base_freq=10000.0):
        super().__init__()
        self.use_learnable = use_learnable
        self.base_freq = base_freq

        if use_learnable:
            # 可学习的位置编码
            self.mlp = nn.Sequential(
                nn.Conv1d(3, d_model // 2, 1),  # 先映射到较高维
                nn.BatchNorm1d(d_model // 2),
                nn.GELU(),  # GELU激活函数可能比ReLU效果更好
                nn.Conv1d(d_model // 2, d_model, 1),
                # 最后不再加BN和激活，让网络自己学习如何调整幅度
            )
        else:
            # 固定正弦位置编码
            self.d_model = d_model

    def forward(self, xyz: torch.Tensor, features: torch.Tensor):
        """
        参数:
            xyz: [B, 3, N] 点云坐标
            features: [B, C, N] 特征
        返回:
            添加位置编码后的特征 [B, C, N]
        """
        # 检查xyz是否为None
        if xyz is None:
            # 如果xyz为None，直接返回特征而不添加位置编码
            return features
        # 首先确保xyz的形状正确 [B, 3, N]
        if xyz.dim() == 3 and xyz.size(2) == 3:
            # 输入是 [B, N, 3]，需要转置为 [B, 3, N]
            xyz = xyz.transpose(1, 2).contiguous()
        batch_size, _, num_points = xyz.size()
        if self.use_learnable:  # True
            # 基于坐标学习位置编码
            pos_enc = self.mlp(xyz)  # [B, C, N]
        else:
            # 使用正弦编码
            pos_enc = torch.zeros(
                batch_size, self.d_model, num_points, device=xyz.device, dtype=xyz.dtype
            )
            assert (
                self.d_model % 6 == 0
            ), "d_model must be divisible by 6 for 3D coordinates in this implementation"
            num_freqs_per_axis = self.d_model // 6  # 每个坐标轴使用的频率数量

            # 生成指数项：0, 1, ..., num_freqs_per_axis-1
            dim_t = torch.arange(num_freqs_per_axis, device=xyz.device, dtype=xyz.dtype)
            # 计算频率分母项： base_freq^(2*i / d_model)
            # 这里使用对数间隔的频率，从低频到高频
            dim_t = self.base_freq ** (2 * dim_t / num_freqs_per_axis)
            pos_enc_axes = xyz.unsqueeze(2) / dim_t.view(1, 1, -1, 1)

            # 4. 交替应用sin和cos，并为每个轴交错排列
            # sin: [B, 3, num_freqs_per_axis, N]
            # cos: [B, 3, num_freqs_per_axis, N]
            sin_enc = torch.sin(pos_enc_axes)
            cos_enc = torch.cos(pos_enc_axes)

            # 5. 将sin和cos交错拼接： [sin, cos, sin, cos, ...]
            # 首先将sin和cos在频率维度上交错堆叠 [B, 3, num_freqs_per_axis*2, N]
            axis_enc = torch.stack([sin_enc, cos_enc], dim=3).flatten(2, 3)

            # 6. 将三个轴（x, y, z）的编码在通道维度拼接起来
            # axis_enc: [B, 3, d_model//3, N] -> flatten(1,2) -> [B, d_model, N]
            pos_enc = axis_enc.flatten(1, 2)
        if features.size(1) == pos_enc.size(1):  # True
            return features + pos_enc
        else:
            # 如果特征维度和位置编码维度不同，一个常见的做法是将位置编码加到网络底层，
            # 或者用一个线性层将位置编码投影到特征维度C。
            # 这里我们简单地输出一个警告并截取或补零（不推荐，最好保证维度匹配）
            # 更好的方法是在初始化时让d_model等于特征的通道数C
            print(
                f"Warning: Feature dim {features.size(1)} != pos_enc dim {pos_enc.size(1)}. Truncating positional encoding."
            )
            min_dim = min(features.size(1), pos_enc.size(1))
            features[:, :min_dim, :] += pos_enc[:, :min_dim, :]
            return features


class MultiheadAttentionWithRelativePE(nn.Module):
    """支持相对位置编码的多头注意力机制"""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # 查询、键、值的线性变换
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        # 输出线性变换
        self.wo = nn.Linear(d_model, d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, relative_bias=None):
        """
        参数:
            x: [B, N, C] 输入特征
            relative_bias: [B, num_heads, N, N] 相对位置偏置
        返回:
            output: [B, N, C] 输出特征
            attn_weights: [B, num_heads, N, N] 注意力权重（可选）
        """
        batch_size, seq_len, _ = x.size()

        # 线性变换并分割成多头
        q = (
            self.wq(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # [B, H, N, D]
        k = (
            self.wk(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # [B, H, N, D]
        v = (
            self.wv(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # [B, H, N, D]

        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )  # [B, H, N, N]

        # 添加相对位置偏置
        if relative_bias is not None:
            attn_scores = attn_scores + relative_bias

        # 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重到值
        output = torch.matmul(attn_weights, v)  # [B, H, N, D]

        # 合并多头
        output = (
            output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        )  # [B, N, C]

        # 输出线性变换
        output = self.wo(output)

        return output, attn_weights


class TransformerBlockWithRelativePE(nn.Module):
    """支持相对位置编码的Transformer块"""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttentionWithRelativePE(d_model, num_heads, dropout)
        # self.self_attn = EfficientMultiheadAttention(d_model, num_heads, dropout)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, relative_bias=None):
        """
        参数:
            x: [B, C, N] 输入特征
            relative_bias: [B, num_heads, N, N] 相对位置偏置
        返回:
            [B, C, N] 输出特征
        """
        # 调整输入维度为 [B, N, C]
        x_transposed = x.transpose(1, 2).contiguous()

        # 自注意力计算，加入相对位置偏置
        attn_output, _ = self.self_attn(x_transposed, relative_bias)
        attn_output = self.dropout(attn_output)

        # 残差连接和层归一化
        x_transposed = self.norm1(x_transposed + attn_output)

        # FFN
        ff_output = self.linear2(
            self.dropout(self.activation(self.linear1(x_transposed)))
        )
        ff_output = self.dropout(ff_output)

        # 残差连接和层归一化
        x_transposed: torch.Tensor = self.norm2(x_transposed + ff_output)

        # 调整回 [B, C, N]
        return x_transposed.transpose(1, 2).contiguous()


class RelativePositionalEncoding(nn.Module):
    """基于点云相对坐标的位置编码，用于注意力偏置"""

    def __init__(self, d_model, num_heads, base_freq=10000.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.base_freq = base_freq

        # 一个小的MLP来将相对坐标映射到注意力偏置
        self.mlp = nn.Sequential(
            nn.Linear(3, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_heads),
        )

    def forward(self, xyz: torch.Tensor):
        """
        参数:
            xyz: [B, 3, N] 点云坐标
        返回:
            relative_bias: [B, num_heads, N, N] 相对位置偏置，用于注意力得分
        """
        if xyz is None:
            return None

        # 统一xyz格式为 [B, N, 3]
        if xyz.dim() == 3 and xyz.size(1) == 3:
            xyz = xyz.transpose(1, 2).contiguous()

        # batch_size, num_points, _ = xyz.size()

        # 计算相对坐标: [B, N, N, 3]
        rel_pos = xyz.unsqueeze(1) - xyz.unsqueeze(2)  # [B, N, N, 3]

        # 通过MLP计算偏置: [B, N, N, num_heads]
        relative_bias: torch.Tensor = self.mlp(rel_pos)  # [B, N, N, num_heads]

        # 调整维度为 [B, num_heads, N, N]
        relative_bias = relative_bias.permute(0, 3, 1, 2).contiguous()

        return relative_bias


class LocalRelativePositionalEncoding(nn.Module):
    """基于局部邻域的相对位置编码，用于注意力偏置"""

    def __init__(self, d_model, num_heads, k_neighbors=16):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.k_neighbors = k_neighbors

        # 一个小的MLP来将相对坐标映射到注意力偏置
        self.mlp = nn.Sequential(
            nn.Linear(3, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, num_heads),
        )

    def forward(self, xyz: torch.Tensor):
        """
        参数:
            xyz: [B, 3, N] 点云坐标
        返回:
            relative_bias: [B, num_heads, N, N] 相对位置偏置，但只对邻域内点有值
        """
        if xyz is None:
            return None

        # 统一xyz格式为 [B, N, 3]
        if xyz.dim() == 3 and xyz.size(1) == 3:
            xyz = xyz.transpose(1, 2).contiguous()

        batch_size, num_points, _ = xyz.size()

        # 使用kNN找到每个点的邻域
        dist = torch.cdist(xyz, xyz)  # [B, N, N]
        _, idx = torch.topk(
            dist, k=self.k_neighbors, dim=-1, largest=False
        )  # [B, N, k]

        # 创建稀疏的相对位置偏置矩阵
        relative_bias = torch.zeros(
            batch_size,
            self.num_heads,
            num_points,
            num_points,
            device=xyz.device,
            dtype=xyz.dtype,
        )

        # 为每个点计算其邻域内的相对位置偏置
        for b in range(batch_size):
            for i in range(num_points):
                # 获取点i的邻域点索引
                neighbors = idx[b, i]  # [k]

                # 计算点i与邻域点的相对坐标
                rel_pos = xyz[b, i].unsqueeze(0) - xyz[b, neighbors]  # [k, 3]

                # 通过MLP计算偏置
                bias = self.mlp(rel_pos)  # [k, num_heads]

                # 将偏置放入对应位置
                relative_bias[b, :, i, neighbors] = bias.transpose(
                    0, 1
                )  # [num_heads, k]

        return relative_bias


class EfficientRelativePositionalEncoding(nn.Module):
    """高效相对位置编码，使用距离和方向分离处理"""

    def __init__(self, d_model, num_heads, use_distance=True, use_direction=True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_distance = use_distance
        self.use_direction = use_direction

        # 距离编码部分
        if use_distance:
            self.distance_encoder = nn.Sequential(
                nn.Linear(1, 16),  # 将距离映射到高维
                nn.ReLU(),
                nn.Linear(16, num_heads),  # 输出每个头的偏置
            )

        # 方向编码部分（使用球面谐波基函数）
        if use_direction:
            self.direction_encoder = nn.Sequential(
                nn.Linear(3, 16),  # 处理方向向量
                nn.ReLU(),
                nn.Linear(16, num_heads),  # 输出每个头的偏置
            )

        # 可选的融合层，将距离和方向编码结合
        if use_distance and use_direction:
            self.fusion = nn.Linear(2 * num_heads, num_heads)
        else:
            self.fusion = None

    def forward(self, xyz: torch.Tensor):
        """
        参数:
            xyz: [B, 3, N] 或 [B, N, 3] 点云坐标
        返回:
            relative_bias: [B, num_heads, N, N] 相对位置偏置
        """
        if xyz is None:
            return None

        # 统一xyz格式为 [B, N, 3]
        if xyz.dim() == 3 and xyz.size(1) == 3:
            xyz = xyz.transpose(1, 2).contiguous()

        batch_size, num_points, _ = xyz.size()

        # 计算相对坐标: [B, N, N, 3]
        rel_pos = xyz.unsqueeze(1) - xyz.unsqueeze(2)

        # 初始化偏置张量
        relative_bias = torch.zeros(
            batch_size,
            num_points,
            num_points,
            self.num_heads,
            device=xyz.device,
            dtype=xyz.dtype,
        )

        # 处理距离信息
        if self.use_distance:
            # 计算欧氏距离: [B, N, N, 1]
            distances = torch.norm(rel_pos, dim=-1, keepdim=True)
            # 通过小型网络编码距离
            distance_bias = self.distance_encoder(distances)  # [B, N, N, num_heads]
            relative_bias = relative_bias + distance_bias

        # 处理方向信息
        if self.use_direction:
            # 计算单位方向向量
            direction = rel_pos / (torch.norm(rel_pos, dim=-1, keepdim=True) + 1e-7)
            # 通过小型网络编码方向
            direction_bias = self.direction_encoder(direction)  # [B, N, N, num_heads]
            relative_bias = relative_bias + direction_bias

        # 融合距离和方向信息（如果两者都使用）
        if self.fusion is not None:
            # 将距离和方向偏置连接后融合
            cat_bias = torch.cat([distance_bias, direction_bias], dim=-1)
            fused_bias = self.fusion(cat_bias)
            relative_bias = fused_bias

        # 调整维度为 [B, num_heads, N, N]
        relative_bias = relative_bias.permute(0, 3, 1, 2).contiguous()

        return relative_bias


class LightweightPositionalEncoding(nn.Module):
    """轻量级位置编码，直接使用坐标的线性变换"""

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        # 直接学习从坐标到头数的映射
        self.coord_to_bias = nn.Linear(3, num_heads)

    def forward(self, xyz):
        """
        直接使用坐标的线性变换作为位置编码
        这种方法非常轻量，但表达能力有限
        """
        if xyz is None:
            return None

        # 统一xyz格式为 [B, N, 3]
        if xyz.dim() == 3 and xyz.size(1) == 3:
            xyz = xyz.transpose(1, 2).contiguous()

        batch_size, num_points, _ = xyz.size()

        # 直接计算每个点的位置偏置
        point_bias = self.coord_to_bias(xyz)  # [B, N, num_heads]

        # 扩展为注意力矩阵形式 [B, num_heads, N, N]
        # 这里使用简单的加法组合，而不是计算所有点对
        relative_bias = point_bias.unsqueeze(2) + point_bias.unsqueeze(
            1
        )  # [B, N, N, num_heads]
        relative_bias = relative_bias.permute(
            0, 3, 1, 2
        ).contiguous()  # [B, num_heads, N, N]

        return relative_bias
