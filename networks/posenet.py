import sys
import os
import torch
import torch.nn as nn

from ipdb import set_trace

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from networks.pts_encoder.pointnets import PointNetfeat
from networks.pts_encoder.pointnet2 import Pointnet2ClsMSG
from networks.pts_encoder.pointnet2 import Pointnet2ClsMSGFus
from networks.gf_algorithms.samplers import (
    cond_ode_likelihood,
    cond_ode_sampler,
    cond_pc_sampler,
)
from networks.gf_algorithms.scorenet import PoseScoreNet, PoseDecoderNet
from networks.gf_algorithms.energynet import PoseEnergyNet
from networks.gf_algorithms.sde import init_sde
from networks.scalenet import ScaleNet

from networks.img_encoder.img_encoder import ImgEncoder
from configs.config import get_config
from utils.genpose_utils import encode_axes


class GFObjectPose(nn.Module):
    dino_name = "dinov2_vits14"
    dino_dim = 384
    embedding_dim = 60

    def __init__(self, cfg, prior_fn, marginal_prob_fn, sde_fn, sampling_eps, T):
        super().__init__()

        self.cfg = cfg
        self.device = cfg.device
        self.is_testing = False

        """ Load model, define SDE """
        # init SDE config
        self.prior_fn = prior_fn
        self.marginal_prob_fn = marginal_prob_fn
        self.sde_fn = sde_fn
        self.sampling_eps = sampling_eps
        self.T = T
        # self.prior_fn, self.marginal_prob_fn, self.sde_fn, self.sampling_eps = init_sde(cfg.sde_mode)

        """ dino v2 """
        if cfg.dino != "none":
            # self.dino: nn.Module = torch.hub.load(
            #     "facebookresearch/dinov2", GFObjectPose.dino_name
            # ).to(cfg.device)
            # self.dino.requires_grad_(False)
            # self.dino_dim = GFObjectPose.dino_dim
            # self.embedding_dim = GFObjectPose.embedding_dim
            self.dino = torch.hub.load(
                repo_or_dir="networks/dinov3",  # 本地克隆的DINOv3仓库路径
                model="dinov3_vits16plus",  # 模型名称（对应ViT-S+）
                source="local",  # 表明从本地仓库加载
                weights="networks/dinov3/weights/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",  # 本地权重文件路径
            )
            self.dino.requires_grad_(False)
            self.dino_dim = GFObjectPose.dino_dim
            self.embedding_dim = GFObjectPose.embedding_dim
            # self.img_encoder = ImgEncoder(self.dino_dim, 196, 16)
            self.img_encoder = ImgEncoder(self.dino_dim, 256, 16)
        """ encode pts """
        if self.cfg.pts_encoder == "pointnet":
            assert cfg.dino != "pointwise"  # not supported yet
            self.pts_encoder = PointNetfeat(
                num_points=self.cfg.num_points, out_dim=1024
            )
        elif self.cfg.pts_encoder == "pointnet2":  # True
            if cfg.dino == "pointwise":  # True
                self.pts_encoder = Pointnet2ClsMSGFus(self.dino_dim)
            else:
                self.pts_encoder = Pointnet2ClsMSG(0)
        elif self.cfg.pts_encoder == "pointnet_and_pointnet2":
            assert cfg.dino != "pointwise"  # not supported yet
            self.pts_pointnet_encoder = PointNetfeat(
                num_points=self.cfg.num_points, out_dim=1024
            )
            self.pts_pointnet2_encoder = Pointnet2ClsMSG(0)
            self.fusion_layer = nn.Linear(2048, 1024)
            self.act = nn.ReLU()
        else:
            raise NotImplementedError

        """ score network"""
        # if self.cfg.sde_mode == 'edm':
        #     self.pose_score_net = PoseDecoderNet(
        #         self.marginal_prob_fn,
        #         sigma_data=1.4148,
        #         pose_mode=self.cfg.pose_mode,
        #         regression_head=self.cfg.regression_head
        #     )
        # else:
        per_point_feat = False
        if self.cfg.agent_type == "score":
            self.pose_score_net = PoseScoreNet(
                self.marginal_prob_fn,
                (
                    0
                    if self.cfg.dino != "global"
                    else self.dino_dim + self.embedding_dim
                ),
                self.cfg.pose_mode,
                self.cfg.regression_head,
                per_point_feat,
            )
        elif self.cfg.agent_type == "energy":
            self.pose_score_net = PoseEnergyNet(
                marginal_prob_func=self.marginal_prob_fn,
                dino_dim=(
                    0
                    if self.cfg.dino != "global"
                    else self.dino_dim + self.embedding_dim
                ),
                pose_mode=self.cfg.pose_mode,
                regression_head=self.cfg.regression_head,
                energy_mode=self.cfg.energy_mode,
                s_theta_mode=self.cfg.s_theta_mode,
                norm_energy=self.cfg.norm_energy,
            )
        """ ToDo: ranking network """

    def extract_pts_feature(self, data):
        """extract the input pointcloud feature

        Args:
            data (dict): batch example without pointcloud feature. {'pts': [bs, num_pts, 3], 'sampled_pose': [bs, pose_dim], 't': [bs, 1]}
        Returns:
            data (dict): batch example with pointcloud feature. {'pts': [bs, num_pts, 3], 'pts_feat': [bs, c], 'sampled_pose': [bs, pose_dim], 't': [bs, 1]}
        """
        pts = data["pts"]
        if self.cfg.dino == "pointwise":  # True
            roi_rgb = data["roi_rgb"]
            feat_all = self.dino.get_intermediate_layers(
                roi_rgb,
                n=[2, 6, 11],
                reshape=False,
                norm=True,  # 归一化特征
                return_class_token=False,
            )
            # feats = feat_all[0]  # ([bs, 196, 384]),其中196=14*14，14=224/16
            feat = self.img_encoder(feat_all)
            # 256*256 时使用

            xs = data["roi_xs"] // 14  # ([bs, 1024]) 变成 patch 级对应索引
            ys = data["roi_ys"] // 14
            pos = xs * 16 + ys  # ([bs, 1024])
            # 224*224 时使用
            # xs = data["roi_xs"] // 16  # ([bs, 1024]) 变成 patch 级对应索引
            # ys = data["roi_ys"] // 16
            # pos = xs * 14 + ys  # ([bs, 1024])
            # ([bs, 1024, 384])
            pos = torch.unsqueeze(pos, -1).expand(-1, -1, self.dino_dim)
            # print(f"\n=== gather 操作检查 ===")
            # 1. 检查 feat 的形状和维度1大小（gather的维度是1）
            # print(f"feat 形状: {feat.shape}")  # 预期类似 (bs, N, 384)，N是feat的总点数
            feat_dim1 = feat.size(1)  # 获取feat在维度1上的大小
            # print(f"feat 维度1（总点数）: {feat_dim1}")

            # 2. 检查 pos 的核心信息（索引的关键属性）
            # print(f"pos 形状: {pos.shape}")  # 预期 (bs, 1024, 1)（因为要选1024个点，最后一维为1广播）
            # print(f"pos dtype: {pos.dtype}")  # 必须是整数型（torch.int32 / torch.int64）
            # print(f"pos 索引范围: min={pos.min().item()}, max={pos.max().item()}")  # 关键！看是否越界

            # 3. 检查 pos 形状是否与 feat 兼容
            # 要求：pos的前N-1个维度与feat匹配，最后一维为1（或与feat最后一维相同）
            if pos.shape[-1] != 1 and pos.shape[-1] != feat.shape[-1]:
                # print(f"警告：pos最后一维形状 {pos.shape[-1]} 不兼容，需为1或 {feat.shape[-1]}")
                # 若不兼容，强制调整最后一维为1（根据你的需求，这里假设是选点，最后一维应为1）
                pos = pos.unsqueeze(-1)
                # print(f"已将 pos 调整为 {pos.shape}")

            # 4. 检查索引是否越界（最核心问题）
            if pos.max().item() >= feat_dim1:
                # print(f"错误：pos的最大索引 {pos.max().item()} ≥ feat维度1大小 {feat_dim1}，会越界！")
                # 临时修正：截断越界索引（后续需根治pos的生成逻辑）
                pos = pos.clamp(0, feat_dim1 - 1)
                # print(f"已将 pos 截断到 [0, {feat_dim1-1}]")
            if pos.min().item() < 0:
                # print(f"错误：pos的最小索引 {pos.min().item()} < 0，会越界！")
                pos = pos.clamp(0, feat_dim1 - 1)

            # 5. 确保 pos 是整数型（CUDA gather 不支持浮点索引）
            if not pos.dtype in [torch.int32, torch.int64]:
                # print(f"警告：pos dtype {pos.dtype} 不是整数型，强制转为int64")
                pos = pos.type(torch.int64)
            try:
                rgb_feat = torch.gather(feat, 1, pos)
                # print(f"gather 成功！rgb_feat 最终形状: {rgb_feat.shape}")  # 预期 (bs,1024,384)
            except RuntimeError as e:
                print(f"gather 失败！详细错误: {str(e)}")
                raise
            # rgb_feat = torch.gather(feat, 1, pos)  # ([bs,1024,384])
            # rgb_feat.requires_grad_(False)
        if self.cfg.pts_encoder == "pointnet":
            assert 0
            pts_feat = self.pts_encoder(pts.permute(0, 2, 1))  # -> (bs, 3, 1024)
        elif self.cfg.pts_encoder in ["pointnet2"]:
            if self.cfg.dino == "pointwise":  # True
                # print(pts.device)
                # print(rgb_feat.device)
                # has_nan_pts = torch.isnan(pts).any().item()
                # has_inf_pts = torch.isinf(pts).any().item()
                # has_nan_rgb = torch.isnan(rgb_feat).any().item()
                # has_inf_rgb = torch.isinf(rgb_feat).any().item()

                # # 打印检查结果（关键！看是否有异常值）
                # print(f"pts - NaN: {has_nan_pts}, Inf: {has_inf_pts}")
                # print(f"rgb_feat - NaN: {has_nan_rgb}, Inf: {has_inf_rgb}")
                concate_data = torch.concatenate([pts, rgb_feat], dim=-1)
                pts_feat = self.pts_encoder(concate_data)
            else:
                pts_feat = self.pts_encoder(pts)
        elif self.cfg.pts_encoder == "pointnet_and_pointnet2":
            assert 0
            pts_pointnet_feat = self.pts_pointnet_encoder(pts.permute(0, 2, 1))
            pts_pointnet2_feat = self.pts_pointnet2_encoder(pts)
            pts_feat = self.fusion_layer(
                torch.cat((pts_pointnet_feat, pts_pointnet2_feat), dim=-1)
            )
            pts_feat = self.act(pts_feat)
        else:
            raise NotImplementedError
        return pts_feat

    def sample(
        self,
        data,
        sampler,
        atol=1e-5,
        rtol=1e-5,
        snr=0.16,
        denoise=True,
        init_x=None,
        T0=None,
    ):
        if sampler == "pc":
            in_process_sample, res = cond_pc_sampler(
                score_model=self,
                data=data,
                prior=self.prior_fn,
                sde_coeff=self.sde_fn,
                num_steps=self.cfg.sampling_steps,
                snr=snr,
                device=self.device,
                eps=self.sampling_eps,
                pose_mode=self.cfg.pose_mode,
                init_x=init_x,
            )

        elif sampler == "ode":
            T0 = self.T if T0 is None else T0
            in_process_sample, res = cond_ode_sampler(
                score_model=self,
                data=data,
                prior=self.prior_fn,
                sde_coeff=self.sde_fn,
                atol=atol,
                rtol=rtol,
                device=self.device,
                eps=self.sampling_eps,
                T=T0,
                num_steps=self.cfg.sampling_steps,
                pose_mode=self.cfg.pose_mode,
                denoise=denoise,
                init_x=init_x,
            )

        else:
            raise NotImplementedError

        return in_process_sample, res

    def calc_likelihood(self, data, atol=1e-5, rtol=1e-5):
        latent_code, log_likelihoods = cond_ode_likelihood(
            score_model=self,
            data=data,
            prior=self.prior_fn,
            sde_coeff=self.sde_fn,
            marginal_prob_fn=self.marginal_prob_fn,
            atol=atol,
            rtol=rtol,
            device=self.device,
            eps=self.sampling_eps,
            num_steps=self.cfg.sampling_steps,
            pose_mode=self.cfg.pose_mode,
        )
        return log_likelihoods

    def forward(self, data, mode="score", init_x=None, T0=None):
        """
        Args:
            data, dict {
                'pts': [bs, num_pts, 3]
                'pts_feat': [bs, c]
                'sampled_pose': [bs, pose_dim]
                't': [bs, 1]
            }
        """
        if mode == "score":
            out_score = self.pose_score_net(data)  # normalisation
            return out_score
        elif mode == "energy":
            out_energy = self.pose_score_net(data, return_item="energy")
            return out_energy
        elif mode == "likelihood":
            likelihoods = self.calc_likelihood(data)
            return likelihoods
        elif mode == "pts_feature":
            pts_feature = self.extract_pts_feature(data)
            return pts_feature
        elif mode == "rgb_feature":  # 不会用到全局 dinov2
            if self.cfg.dino != "global":
                return None
            rgb: torch.Tensor = data["roi_rgb"]
            assert rgb.shape[-1] % 14 == 0 and rgb.shape[-2] % 14 == 0
            assert self.embedding_dim % 6 == 0
            # lose backward compatibility
            return torch.concat(
                [
                    self.dino(rgb),
                    encode_axes(data["roi_center_dir"], dim=self.embedding_dim // 6),
                ],
                dim=-1,
            )

            # positional_embedding = []
            # exponent = (2 ** torch.arange(self.embedding_dim // 6, device=rgb.device, dtype=torch.float32)).reshape(1, -1)
            # for i in range(3):
            #     for fn in [torch.sin, torch.cos]:
            #         positional_embedding.append(fn(exponent * data['roi_center_dir'][:, i].reshape(-1, 1)))
            # positional_embedding = torch.concat(positional_embedding, dim=-1)
            # return torch.concat([self.dino(rgb), positional_embedding], dim=-1)
        elif mode == "pc_sample":
            in_process_sample, res = self.sample(data, "pc", init_x=init_x)
            return in_process_sample, res
        elif mode == "ode_sample":
            in_process_sample, res = self.sample(data, "ode", init_x=init_x, T0=T0)
            return in_process_sample, res
        else:
            raise NotImplementedError


def test():
    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {"Total": total_num, "Trainable": trainable_num}

    cfg = get_config()
    prior_fn, marginal_prob_fn, sde_fn, sampling_eps, T = init_sde("ve")
    net = GFObjectPose(cfg, prior_fn, marginal_prob_fn, sde_fn, sampling_eps, T)
    net_parameters_num = get_parameter_number(net)
    print(net_parameters_num["Total"], net_parameters_num["Trainable"])


if __name__ == "__main__":
    test()
