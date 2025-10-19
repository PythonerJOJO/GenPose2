import torch
import torch.nn as nn
import sys
import os
import torch.nn.functional as F

# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from networks.pts_encoder.attention import (
    EfficientRelativePositionalEncoding,
    GatedAttentionFusion,
    ImprovedPositionalEncoding,
    LightweightPositionalEncoding,
    LocalRelativePositionalEncoding,
    RelativePositionalEncoding,
    TransformerBlock,
    TransformerBlockWithRelativePE,
)
from networks.pts_encoder.pointnet2_utils.pointnet2.pointnet2_modules import (
    PointnetFPModule,
    PointnetSAModuleMSG,
)
import networks.pts_encoder.pointnet2_utils.pointnet2.pytorch_utils as pt_utils
from ipdb import set_trace
from configs.config import get_config


cfg = get_config()


def get_model(input_channels=0):
    return Pointnet2MSG(input_channels=input_channels)


MSG_CFG = {
    "NPOINTS": [512, 256, 128, 64],
    "RADIUS": [[0.01, 0.02], [0.02, 0.04], [0.04, 0.08], [0.08, 0.16]],
    "NSAMPLE": [[16, 32], [16, 32], [16, 32], [16, 32]],
    "MLPS": [
        [[16, 16, 32], [32, 32, 64]],
        [[64, 64, 128], [64, 96, 128]],
        [[128, 196, 256], [128, 196, 256]],
        [[256, 256, 512], [256, 384, 512]],
    ],
    "FP_MLPS": [[64, 64], [128, 128], [256, 256], [512, 512]],
    "CLS_FC": [128],
    "DP_RATIO": 0.5,
}

ClsMSG_CFG = {
    "NPOINTS": [512, 256, 128, 64, None],
    "RADIUS": [[0.01, 0.02], [0.02, 0.04], [0.04, 0.08], [0.08, 0.16], [None, None]],
    "NSAMPLE": [[16, 32], [16, 32], [16, 32], [16, 32], [None, None]],
    "MLPS": [
        [[16, 16, 32], [32, 32, 64]],
        [[64, 64, 128], [64, 96, 128]],
        [[128, 196, 256], [128, 196, 256]],
        [[256, 256, 512], [256, 384, 512]],
        [[512, 512], [512, 512]],
    ],
    "DP_RATIO": 0.5,
}

ClsMSG_CFG_Dense = {
    "NPOINTS": [512, 256, 128, None],
    "RADIUS": [[0.02, 0.04], [0.04, 0.08], [0.08, 0.16], [None, None]],
    "NSAMPLE": [[32, 64], [16, 32], [8, 16], [None, None]],
    "MLPS": [
        [[16, 16, 32], [32, 32, 64]],
        [[64, 64, 128], [64, 96, 128]],
        [[128, 196, 256], [128, 196, 256]],
        [[256, 256, 512], [256, 384, 512]],
    ],
    "DP_RATIO": 0.5,
}

ClsMSG_CFG_Light = {
    "NPOINTS": [512, 256, 128, 64, None],
    "RADIUS": [[0.01, 0.02], [0.02, 0.04], [0.04, 0.08], [0.08, 0.16], [None, None]],
    "NSAMPLE": [[16, 32], [16, 32], [16, 32], [16, 32], [None, None]],
    "MLPS": [
        [[16, 16, 32], [32, 32, 64]],
        [[64, 64, 128], [64, 96, 128]],
        [[128, 196, 256], [128, 196, 256]],
        [[256, 256, 512], [256, 384, 512]],
        [[512, 512], [512, 512]],
    ],
    "DP_RATIO": 0.5,
}
########## Best before 29th April ###########
ClsMSG_CFG_Light_raw = {
    "NPOINTS": [512, 256, 128, None],  # 每层采样点数
    # 不同尺度的采样半径
    "RADIUS": [[0.02, 0.04], [0.04, 0.08], [0.08, 0.16], [None, None]],
    # 不同半径内采样的点数
    "NSAMPLE": [[16, 32], [16, 32], [16, 32], [None, None]],
    # 不同尺度的 MLP 结构
    "MLPS": [
        [[16, 16, 32], [32, 32, 64]],
        [[64, 64, 128], [64, 96, 128]],
        [[128, 196, 256], [128, 196, 256]],
        [[256, 256, 512], [256, 384, 512]],
    ],
    "DP_RATIO": 0.5,  # 丢弃率
}


ClsMSG_CFG_Lighter = {
    "NPOINTS": [512, 256, 128, 64, None],
    "RADIUS": [[0.01], [0.02], [0.04], [0.08], [None]],
    "NSAMPLE": [[64], [32], [16], [8], [None]],
    "MLPS": [
        [[32, 32, 64]],
        [[64, 64, 128]],
        [[128, 196, 256]],
        [[256, 256, 512]],
        [[512, 512, 1024]],
    ],
    "DP_RATIO": 0.5,
}


if cfg.pointnet2_params == "light":
    SELECTED_PARAMS = ClsMSG_CFG_Light
elif cfg.pointnet2_params == "lighter":
    SELECTED_PARAMS = ClsMSG_CFG_Lighter
elif cfg.pointnet2_params == "dense":
    SELECTED_PARAMS = ClsMSG_CFG_Dense
else:
    raise NotImplementedError


class Pointnet2MSG(nn.Module):
    def __init__(self, input_channels=6):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

        skip_channel_list = [input_channels]
        for k in range(len(MSG_CFG["NPOINTS"])):
            mlps = MSG_CFG["MLPS"][k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                PointnetSAModuleMSG(
                    npoint=MSG_CFG["NPOINTS"][k],
                    radii=MSG_CFG["RADIUS"][k],
                    nsamples=MSG_CFG["NSAMPLE"][k],
                    mlps=mlps,
                    use_xyz=True,
                    bn=True,
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        self.FP_modules = nn.ModuleList()

        for k in range(MSG_CFG["FP_MLPS"].__len__()):
            pre_channel = (
                MSG_CFG["FP_MLPS"][k + 1][-1]
                if k + 1 < len(MSG_CFG["FP_MLPS"])
                else channel_out
            )
            self.FP_modules.append(
                PointnetFPModule(
                    mlp=[pre_channel + skip_channel_list[k]] + MSG_CFG["FP_MLPS"][k]
                )
            )

        cls_layers = []
        pre_channel = MSG_CFG["FP_MLPS"][0][-1]
        for k in range(0, MSG_CFG["CLS_FC"].__len__()):
            cls_layers.append(
                pt_utils.Conv1d(pre_channel, MSG_CFG["CLS_FC"][k], bn=True)
            )
            pre_channel = MSG_CFG["CLS_FC"][k]
        cls_layers.append(pt_utils.Conv1d(pre_channel, 1, activation=None))
        cls_layers.insert(1, nn.Dropout(0.5))
        self.cls_layer = nn.Sequential(*cls_layers)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud: torch.Tensor):
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])

            l_xyz.append(li_xyz)
            l_features.append(li_features)

        set_trace()
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return l_features[0]


class Pointnet2ClsMSG(nn.Module):
    def __init__(self, input_channels=6):
        super().__init__()

        self.SA_modules = nn.ModuleList()  # SA Modules 集合
        channel_in = input_channels

        # SELECTED_PARAMS 查看变量 ClsMSG_CFG_Light
        for k in range(SELECTED_PARAMS["NPOINTS"].__len__()):
            mlps = SELECTED_PARAMS["MLPS"][k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                PointnetSAModuleMSG(
                    npoint=SELECTED_PARAMS["NPOINTS"][k],
                    radii=SELECTED_PARAMS["RADIUS"][k],
                    nsamples=SELECTED_PARAMS["NSAMPLE"][k],
                    mlps=mlps,
                    use_xyz=True,
                    bn=True,
                )
            )
            channel_in = channel_out

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud: torch.Tensor):
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        return l_features[-1].squeeze(-1)


class Pointnet2ClsMSGFus(nn.Module):
    """
    从 Pointnet2ClsMSG 修改而来：
    将输入特征与每一层的特征进行拼接，以实现全特征融合。
    input_channels = self.dino = 384
    """

    def __init__(self, input_channels=6):
        super().__init__()
        channel_in = input_channels
        # channel_in = 387
        self.SA_modules = nn.ModuleList()
        # 新增：点级多头自注意力网络
        self.transformer_blocks = nn.ModuleList()
        # 门控注意力
        self.feature_fusions = nn.ModuleList()
        # 修改：将绝对位置编码器改为相对位置编码器
        self.relative_pos_encoders = nn.ModuleList()
        # 新增：Dropout层
        SELECTED_PARAMS["DP_RATIO"] = 0.1
        self.dropout = nn.Dropout(p=SELECTED_PARAMS["DP_RATIO"])

        # 存储所有层级的特征
        self.all_level_features = []
        # 存储每一层的输出通道数
        self.layer_output_channels = []
        # SELECTED_PARAMS 查看变量 ClsMSG_CFG_Light
        for k in range(len(SELECTED_PARAMS["NPOINTS"])):  # 4
            mlps = SELECTED_PARAMS["MLPS"][k].copy()
            channel_out = 0
            for idx in range(len(mlps)):  # 遍历每个 MLP 结构
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]
            # 记录当前层的输出通道数
            self.layer_output_channels.append(channel_out)
            self.SA_modules.append(
                PointnetSAModuleMSG(
                    npoint=SELECTED_PARAMS["NPOINTS"][k],  # 512,256,128,None
                    # [[0.02, 0.04], [0.04, 0.08], [0.08, 0.16], [None, None]]
                    radii=SELECTED_PARAMS["RADIUS"][k],
                    # [[16, 32], [16, 32], [16, 32], [None, None]]
                    nsamples=SELECTED_PARAMS["NSAMPLE"][k],
                    mlps=mlps,
                    use_xyz=True,
                    bn=True,
                )
            )
            # 修改：添加相对位置编码器
            self.relative_pos_encoders.append(
                # LocalRelativePositionalEncoding(
                #     channel_out, num_heads=8, k_neighbors=16
                # )
                # ImprovedPositionalEncoding(channel_out, True)
                # LightweightPositionalEncoding(channel_out, num_heads=8)
                EfficientRelativePositionalEncoding(channel_out, num_heads=8)
                # RelativePositionalEncoding(channel_out, num_heads=8)
            )
            # 修改：使用支持相对位置编码的Transformer块
            self.transformer_blocks.append(
                TransformerBlockWithRelativePE(channel_out, num_heads=8, dropout=0.1)
            )

            # 添加门控注意力融合模块（除了第一层）
            if k > 0:
                # print(
                #     f"Layer {k}: channel_in={channel_in}, input_channels={input_channels}"
                # )
                # 使用上一层的输出通道数作为当前层的输入通道数
                prev_output_channels = self.layer_output_channels[k - 1]
                self.feature_fusions.append(
                    GatedAttentionFusion(prev_output_channels, input_channels)
                )
            # channel_in = channel_out + input_channels
            channel_in = channel_out
        # print(f"Layer output channels: {self.layer_output_channels}")

    def forward(self, pointcloud: torch.Tensor):
        if self.training:
            pointcloud = pointcloud + torch.randn_like(pointcloud) * 1e-3
        xyz, features = self._break_up_pc(pointcloud)
        # xyz:      bs * npoints * 3
        # features: bs * F * npoints
        original_features = features.clone()
        l_xyz, l_features = [xyz], [features]  # 用于存储每一层的 xyz 坐标和特征
        all_level_features = [features]
        downsampled_original = original_features
        for i in range(len(self.SA_modules)):
            if i > 0:  # 从第二层开始，将当前层的特征与初始特征在维度 1 上进行拼接

                if downsampled_original.size(2) != l_features[i].size(2):
                    downsampled_original = F.interpolate(
                        downsampled_original,
                        size=l_features[i].size(2),
                        mode="linear",
                        align_corners=False,
                    )
                fused_features = self.feature_fusions[i - 1](
                    l_features[i], downsampled_original
                )
                l_features[i] = self.dropout(fused_features)

            # 调用 SA_modules 中的第 i 个模块进行特征提取和下采样
            li_xyz, li_features, idx = self.SA_modules[i](
                l_xyz[i], l_features[i], return_idx=True
            )

            # 修改：计算相对位置编码（偏置）
            relative_bias = self.relative_pos_encoders[i](li_xyz)

            # 修改：调用Transformer块，传入相对位置偏置
            li_features = self.transformer_blocks[i](
                li_features, relative_bias=relative_bias
            )

            l_xyz.append(li_xyz)
            l_features.append(li_features)
            all_level_features.append(li_features)  # 保存当前层特征
            if idx != None:  # 下采样初始特征
                features = torch.gather(
                    features,
                    2,
                    torch.unsqueeze(idx.type(torch.int64), 1).expand(
                        -1, features.shape[1], -1
                    ),
                )  # only keep features of remaining points
            else:
                assert i == len(self.SA_modules) - 1
        return l_features[-1].squeeze(-1)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features


class Pointnet2ClsMSGFus_position(nn.Module):
    """
    从 Pointnet2ClsMSG 修改而来：
    将输入特征与每一层的特征进行拼接，以实现全特征融合。
    input_channels = self.dino = 384
    """

    def __init__(self, input_channels=6):
        super().__init__()
        channel_in = input_channels
        self.SA_modules = nn.ModuleList()
        # 新增：点级多头自注意力网络
        self.transformer_blocks = nn.ModuleList()
        # 门控注意力
        self.feature_fusions = nn.ModuleList()
        self.pos_encoders = nn.ModuleList()
        # 新增：Dropout层
        self.dropout = nn.Dropout(p=SELECTED_PARAMS["DP_RATIO"])

        # 存储所有层级的特征
        self.all_level_features = []
        # 存储每一层的输出通道数
        self.layer_output_channels = []
        # SELECTED_PARAMS 查看变量 ClsMSG_CFG_Light
        for k in range(len(SELECTED_PARAMS["NPOINTS"])):  # 4
            mlps = SELECTED_PARAMS["MLPS"][k].copy()
            channel_out = 0
            for idx in range(len(mlps)):  # 遍历每个 MLP 结构
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]
            # 记录当前层的输出通道数
            self.layer_output_channels.append(channel_out)
            self.SA_modules.append(
                PointnetSAModuleMSG(
                    npoint=SELECTED_PARAMS["NPOINTS"][k],  # 512,256,128,None
                    # [[0.02, 0.04], [0.04, 0.08], [0.08, 0.16], [None, None]]
                    radii=SELECTED_PARAMS["RADIUS"][k],
                    # [[16, 32], [16, 32], [16, 32], [None, None]]
                    nsamples=SELECTED_PARAMS["NSAMPLE"][k],
                    mlps=mlps,
                    use_xyz=True,
                    bn=True,
                )
            )
            # 添加位置编码器
            self.pos_encoders.append(ImprovedPositionalEncoding(channel_out))
            # 注意力网络：输入为SA模块输出特征，输出每个点的注意力权重
            self.transformer_blocks.append(
                TransformerBlock(channel_out, num_heads=8, dropout=0.1)
            )

            # 添加门控注意力融合模块（除了第一层）
            if k > 0:
                # print(
                #     f"Layer {k}: channel_in={channel_in}, input_channels={input_channels}"
                # )
                # 使用上一层的输出通道数作为当前层的输入通道数
                prev_output_channels = self.layer_output_channels[k - 1]
                self.feature_fusions.append(
                    GatedAttentionFusion(prev_output_channels, input_channels)
                )
            # channel_in = channel_out + input_channels
            channel_in = channel_out
        # print(f"Layer output channels: {self.layer_output_channels}")

    def forward(self, pointcloud: torch.Tensor):
        if self.training:
            pointcloud = pointcloud + torch.randn_like(pointcloud) * 1e-3
        xyz, features = self._break_up_pc(pointcloud)
        # xyz:      bs * npoints * 3
        # features: bs * F * npoints
        original_features = features.clone()
        l_xyz, l_features = [xyz], [features]  # 用于存储每一层的 xyz 坐标和特征
        all_level_features = [features]
        downsampled_original = original_features
        for i in range(len(self.SA_modules)):
            if i > 0:  # 从第二层开始，将当前层的特征与初始特征在维度 1 上进行拼接

                if downsampled_original.size(2) != l_features[i].size(2):
                    downsampled_original = F.interpolate(
                        downsampled_original,
                        size=l_features[i].size(2),
                        mode="linear",
                        align_corners=False,
                    )
                fused_features = self.feature_fusions[i - 1](
                    l_features[i], downsampled_original
                )
                l_features[i] = self.dropout(fused_features)

            # 调用 SA_modules 中的第 i 个模块进行特征提取和下采样
            li_xyz, li_features, idx = self.SA_modules[i](
                l_xyz[i], l_features[i], return_idx=True
            )
            # 位置编码
            li_features_with_pos = self.pos_encoders[i](li_xyz, li_features)
            li_features = self.transformer_blocks[i](li_features_with_pos)

            l_xyz.append(li_xyz)
            l_features.append(li_features)
            all_level_features.append(li_features)  # 保存当前层特征
            if idx != None:  # 下采样初始特征
                features = torch.gather(
                    features,
                    2,
                    torch.unsqueeze(idx.type(torch.int64), 1).expand(
                        -1, features.shape[1], -1
                    ),
                )  # only keep features of remaining points
            else:
                assert i == len(self.SA_modules) - 1
        return l_features[-1].squeeze(-1)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features


class Pointnet2ClsMSGFus_raw2(nn.Module):
    """
    从 Pointnet2ClsMSG 修改而来：
    将输入特征与每一层的特征进行拼接，以实现全特征融合。
    input_channels = self.dino = 384
    """

    def __init__(self, input_channels=6):
        super().__init__()
        channel_in = input_channels
        self.SA_modules = nn.ModuleList()
        # 新增：点级注意力网络（每个SA模块后添加）
        self.point_attn = nn.ModuleList()
        # 新增：Dropout层（融合后添加）
        self.dropout = nn.Dropout(p=0.3)  # 可根据需求调整概率

        # SELECTED_PARAMS 查看变量 ClsMSG_CFG_Light
        for k in range(len(SELECTED_PARAMS["NPOINTS"])):  # 4
            mlps = SELECTED_PARAMS["MLPS"][k].copy()
            channel_out = 0
            for idx in range(len(mlps)):  # 遍历每个 MLP 结构
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                PointnetSAModuleMSG(
                    npoint=SELECTED_PARAMS["NPOINTS"][k],  # 512,256,128,None
                    # [[0.02, 0.04], [0.04, 0.08], [0.08, 0.16], [None, None]]
                    radii=SELECTED_PARAMS["RADIUS"][k],
                    # [[16, 32], [16, 32], [16, 32], [None, None]]
                    nsamples=SELECTED_PARAMS["NSAMPLE"][k],
                    mlps=mlps,
                    use_xyz=True,
                    bn=True,
                )
            )
            # 注意力网络：输入为SA模块输出特征，输出每个点的注意力权重
            self.point_attn.append(
                nn.Sequential(
                    pt_utils.Conv1d(channel_out, channel_out // 4, bn=True),
                    nn.ReLU(),
                    # pt_utils.Conv1d(channel_out // 2, channel_out // 4, bn=True),
                    # nn.ReLU(),
                    pt_utils.Conv1d(
                        channel_out // 4, 1, activation=nn.Sigmoid()
                    ),  # 输出0-1权重
                )
            )
            channel_in = channel_out + input_channels
        # print(self.SA_modules)

    def forward(self, pointcloud: torch.Tensor):
        if self.training:
            pointcloud = pointcloud + torch.randn_like(pointcloud) * 1e-3
        xyz, features = self._break_up_pc(pointcloud)
        # xyz:      bs * npoints * 3
        # features: bs * F * npoints

        l_xyz, l_features = [xyz], [features]  # 用于存储每一层的 xyz 坐标和特征
        original_features = features
        for i in range(len(self.SA_modules)):
            if i != 0:  # 从第二层开始，将当前层的特征与初始特征在维度 1 上进行拼接
                l_features[i] = torch.cat([l_features[i], features], dim=1)
                l_features[i] = self.dropout(l_features[i])
            # 调用 SA_modules 中的第 i 个模块进行特征提取和下采样
            li_xyz, li_features, idx = self.SA_modules[i](
                l_xyz[i], l_features[i], return_idx=True
            )
            # 新增：点级注意力加权
            attn_weight = self.point_attn[i](li_features)  # [bs, 1, npoints]
            li_features = li_features * attn_weight  # 加权增强关键特征

            l_xyz.append(li_xyz)
            l_features.append(li_features)
            if idx != None:  # 下采样初始特征
                features = torch.gather(
                    features,
                    2,
                    torch.unsqueeze(idx.type(torch.int64), 1).expand(
                        -1, features.shape[1], -1
                    ),
                )  # only keep features of remaining points
            else:
                assert i == len(self.SA_modules) - 1
        return l_features[-1].squeeze(-1)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features


def test01_data_forward():
    # net = Pointnet2ClsMSG(384).cuda()
    net = Pointnet2ClsMSGFus(384).cuda()
    pts = torch.randn(32, 1024, 3).cuda()
    rgb_feature = torch.randn(32, 1024, 384).cuda()  # RGB
    # print(torch.mean(pts, dim=1))
    pre = net(torch.cat([pts, rgb_feature], dim=-1))  # (32, 1024, 3+384)->32,1024
    print(pre.shape)


# ComputationGraph


def test02_model_param_view():
    # from torchsummary import summary
    from torchinfo import summary

    net = Pointnet2ClsMSGFus(384 + 512).cuda()
    # input_size = (64, 2048, 3 + 768)
    input_size = (64, 1024, 3 + 384 + 512)
    print("模型结构摘要:")
    summary(
        net,
        input_size=input_size,
        device="cuda",
        col_names=["input_size", "output_size", "num_params", "mult_adds"],  # 显示列
        col_width=20,  # 列宽
        depth=4,  # 显示嵌套模块的深度
    )


if __name__ == "__main__":
    seed = 100
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    test02_model_param_view()
