if __name__ == "__main__":
    from dino_feature import DinoFeature
    from attention import IndustrialAttentionFusion
else:
    from model.img_future.dino_feature import DinoFeature
    from model.img_future.attention import IndustrialAttentionFusion
from torch import device, nn
import torch

# from dinov3.models.vision_transformer import DinoVisionTransformer


class ImgEncoder(nn.Module):
    def __init__(self, cfg, lays=[2, 6, 11]) -> None:
        super().__init__()
        self.device = cfg.device
        self.future_lays = lays
        # self.dino: DinoVisionTransformer = DinoFeature.get_dinov3(
        self.dino = DinoFeature.get_dinov3(
            dino_name="dinov3_vits16plus",
            weight_path="/home/dai/ai/reference/dinov3/weights/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
            dinolib_path="/home/dai/ai/mypose/lib/dinov3",
        ).to(self.device)
        for param in self.dino.parameters():
            param.requires_grad = False
        self.dino.eval()
        self.dino_fusion = IndustrialAttentionFusion().to(self.device)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            intermediate_outputs: list[torch.Tensor] = (
                self.dino.get_intermediate_layers(
                    data,
                    # n=[3, 7, 11],  # 示例：提取第2、5、8层（多尺度）
                    n=self.future_lays,  # 示例：提取第2、5、8层（多尺度）
                    # n=3,  # 示例：提取第2、5、8层（多尺度）
                    reshape=False,  # 不重塑为空间维度，保持序列格式
                    norm=True,  # 归一化特征
                    return_class_token=False,
                )
            )
        img_future = self.dino_fusion(intermediate_outputs)
        return img_future

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.dino.eval()
        return self


if __name__ == "__main__":

    class Cfg:
        def __init__(self) -> None:
            self.device = torch.device("cuda")

    cfg = Cfg()
    imgEncoder = ImgEncoder(cfg)
    B, C, H, W = 128, 3, 224, 224
    norm_img = torch.rand(B, C, H, W).to(cfg.device)
    img_future: torch.Tensor = imgEncoder(norm_img)
    print(img_future.shape)  # (128, 196, 384)
    print(imgEncoder)
    # print("初始模式 - dino:", imgEncoder.dino.training)  # 应输出 False
    # print("初始模式 - fusion:", imgEncoder.dino_fusion.training)
    # imgEncoder.train()
    # print("训练模式 - dino:", imgEncoder.dino.training)  # 应输出 False（正确）
    # print("训练模式 - fusion:", imgEncoder.dino_fusion.training)
    # imgEncoder.eval()
    # print("评估模式 - dino:", imgEncoder.dino.training)  # 应输出 False
    # print("评估模式 - fusion:", imgEncoder.dino_fusion.training)
