import torch


class DinoFeature:

    @staticmethod
    def get_dino(dino_name, dino_dim):
        model = torch.hub.load("facebookresearch/dinov2", dino_name)
        model.requires_grad_(False)
        model.dino_dim = dino_dim
        # self.embedding_dim = DinoFeature.embedding_dim
        return model

    @staticmethod
    def get_dinov3(
        dino_name="dinov3_vits16plus",
        weight_path="/home/dai/ai/reference/dinov3/weights/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
        dinolib_path=".",
    ):
        model = torch.hub.load(
            repo_or_dir=dinolib_path,  # 本地克隆的DINOv3仓库路径
            model=dino_name,  # 模型名称（对应ViT-S+）
            source="local",  # 表明从本地仓库加载
            weights=weight_path,  # 本地权重文件路径
        )
        return model


def demo1():
    device = torch.device("cuda")
    model = DinoFeature.get_dinov3(
        dinolib_path="/home/dai/ai/SpherNetPose2/lib/dinov3"
    ).to(device)
    import sys
    from pathlib import Path

    # 获取当前脚本所在目录的上级目录（即 lib 的父目录）
    # parent_dir = str(Path(__file__).resolve().parent.parent)
    # 将上级目录添加到 Python 路径
    # sys.path.append(parent_dir)
    # from lib.dinov3.dinov3.eval.utils import ModelWithMultiScale

    from dinov3.eval.utils import ModelWithMultiScale

    model.eval()
    # 封装模型为多尺度版本（mode="bilinear" 用于缩放特征）
    multi_scale_model = ModelWithMultiScale(model, mode="bilinear").to(device)

    random_imgs = torch.rand(4, 3, 256, 256).to(device)
    target_layers = [10]
    with torch.inference_mode():
        dino_feature_all = model.get_intermediate_layers(
            random_imgs,
            n=target_layers,
            return_class_token=False,
            norm=True,  # 归一化特征
        )
        multi_scale_feat = multi_scale_model(random_imgs)
        dino_feature = dino_feature_all[0]
    print(f"中间层特征形状: {dino_feature.shape}")  # 应为 (bs, 256, 384)
    print(f"中间层特征形状: {multi_scale_feat.shape}")  # 应为 (bs, 256, 384)


def demo2():
    """多尺度特征"""
    device = torch.device("cuda")
    model = (
        DinoFeature()
        .get_dinov3(dinolib_path="/home/dai/ai/mypose/lib/dinov3")
        .to(device)
    )
    # total_layers = model.backbone.depth  # 核心：获取Transformer的总层数
    print(f"模型总层数: {model}")
    inputs = torch.rand(4, 3, 256, 256).to(device=device)
    with torch.inference_mode():
        intermediate_outputs = model.get_intermediate_layers(
            inputs,
            # n=[3, 7, 11],  # 示例：提取第2、5、8层（多尺度）
            n=[2, 6, 11],  # 示例：提取第2、5、8层（多尺度）
            # n=3,  # 示例：提取第2、5、8层（多尺度）
            reshape=False,  # 不重塑为空间维度，保持序列格式
            norm=True,  # 归一化特征
            return_class_token=False,
        )
    for i, features in enumerate(intermediate_outputs):
        print(f"第{i+1}个中间层（索引{i}）特征形状: {features.shape}")


if __name__ == "__main__":
    # demo1()
    demo2()
