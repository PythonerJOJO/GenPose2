import torch


def main():
    dino_feature = torch.rand(64, 384, 256)
    li_features = [
        torch.rand(64, 384, 512),
        torch.rand(64, 384, 256),
        torch.rand(64, 384, 128),
    ]
    l_features = [
        torch.rand(64, 384, 1024),  # 原始dino特征，通道优先
        torch.rand(64, 96, 512),
        torch.rand(64, 256, 256),
        torch.rand(64, 512, 128),
        # torch.rand(64, 1024, 1),
    ]
    for i in range(4):
        if i != 0:
            l_features[i] = torch.cat([l_features[i], li_features[i - 1]], dim=1)
    print(l_features[-1].shape)


if __name__ == "__main__":
    main()
