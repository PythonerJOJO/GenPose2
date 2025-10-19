import copy
import os
import pickle
import random
from sympy import im
import torch
from torch.utils.data import DataLoader, Subset

from datasets_my.xyzibd_dataset import XyzibdDataset

dataset_type_mapping = {"xyzibd": XyzibdDataset}


def fast_dataloader(cfg, dataset_type, object):
    if object < 0:
        object = "all"
        pkl_path = os.path.join("data", f"{dataset_type}_dataset_all.pkl")
    else:
        pkl_path = os.path.join("data", f"{dataset_type}_dataset_{object:02d}.pkl")
    with open(pkl_path, "rb") as f:
        datasets = pickle.load(f)
    train_ds = datasets["train_ds"]
    val_ds = datasets["val_ds"]
    test_ds = datasets["test_ds"]
    data_loaders = {}
    data_loaders["train_loader"] = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    data_loaders["val_loader"] = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    data_loaders["test_loader"] = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    if cfg.verbose:
        print(f"train_dataset: {len(train_ds)}")
        print(f"val_dataset: {len(val_ds)}")
        print(f"test_dataset: {len(test_ds)}")
    return data_loaders


from .xyzibd_dataset import compute_translation_stats

TRANS_STATS_SAVE_DIR = "/root/autodl-fs/GenPose2/configs"  # 你的config目录路径
TRANS_MEAN_FILENAME = "xyzibd_trans_mean.npy"
TRANS_STD_FILENAME = "xyzibd_trans_std.npy"


class DatasetPort:
    @staticmethod
    def save_trans_stats(trans_mean, trans_std, save_dir=TRANS_STATS_SAVE_DIR):
        import numpy as np

        """
        保存trans_mean和trans_std到config目录
        :param trans_mean: 计算出的平移均值 [3,]
        :param trans_std: 计算出的平移标准差 [3,]
        :param save_dir: 保存目录（默认是你的config目录）
        """
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)

        # 拼接保存路径
        mean_save_path = os.path.join(save_dir, TRANS_MEAN_FILENAME)
        std_save_path = os.path.join(save_dir, TRANS_STD_FILENAME)

        # 保存为npy文件
        np.save(mean_save_path, trans_mean)
        np.save(std_save_path, trans_std)
        print(f"[INFO] trans统计量已保存到：\n  {mean_save_path}\n  {std_save_path}")

    @staticmethod
    def get_data_loaders_from_cfg(
        cfg,
        dataset_type: str = "xyzibd",
        data_type: list = ["train", "val", "test"],
    ):
        assert dataset_type in {"xyzibd"}, f"不支持的数据集: {dataset_type}"
        assert (
            0 < cfg.sample_ratio <= 1
        ), f"抽样比例必须在(0, 1]之间，当前值: {cfg.sample_ratio}"
        sample_ratio = cfg.sample_ratio
        DATASET = dataset_type_mapping[dataset_type]
        # return fast_dataloader(cfg, dataset_type, cfg.target_obj_id)
        data_loaders = {}

        if "train" in data_type:
            train_ds = DATASET.get_train_dataset(cfg)
            # trans_mean, trans_std = compute_translation_stats(train_ds)
            # DatasetPort.save_trans_stats(trans_mean=trans_mean,trans_std=trans_std,save_dir=TRANS_STATS_SAVE_DIR)
            # 计算需要抽样的数量
            train_size = int(len(train_ds) * sample_ratio)
            # 随机选择索引
            train_indices = random.sample(range(len(train_ds)), train_size)
            # 创建子集
            train_ds_subset = Subset(train_ds, train_indices)
            data_loaders["train_loader"] = DataLoader(
                train_ds_subset,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
            )
        if "val" in data_type:
            val_ds = DATASET.get_val_dataset(cfg)
            # 计算需要抽样的数量
            val_size = int(len(val_ds) * sample_ratio)
            # 随机选择索引
            val_indices = random.sample(range(len(val_ds)), val_size)
            # 创建子集
            val_ds_subset = Subset(val_ds, val_indices)
            data_loaders["val_loader"] = DataLoader(
                val_ds_subset,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
            )
        if "test" in data_type:
            test_ds = DATASET.get_test_dataset(cfg)
            data_loaders["test_loader"] = DataLoader(
                test_ds,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
            )
        if cfg.verbose:
            if "train_ds" in locals() or "train_ds" in globals():
                print(f"train_dataset: {len(train_ds)}")
            else:
                print(f"不存在训练集")
            if "val_ds" in locals() or "val_ds" in globals():
                print(f"train_dataset: {len(val_ds)}")
            else:
                print(f"不存在训练集")
            # print(f"test_dataset: {len(test_ds)}")
        #
        return data_loaders

    @staticmethod
    def process_batch_xyzibd(
        batch_sample, device, pose_mode="se3", PTS_AUG_PARAMS=None
    ):
        PC_da = batch_sample["input"]["pcl_in"].to(device)  # 1024个采样的点云
        # gt_R_da = batch_sample["rotation"].to(device)
        # gt_t_da = batch_sample["translation"].to(device)

        processed_sample = {}

        processed_sample["pts"] = PC_da  # [bs, 1024, 3]
        processed_sample["pts_color"] = PC_da
        processed_sample["pts_idx"] = batch_sample["input"]["pcl_in_index"].to(device)
        processed_sample["roi_xs"] = batch_sample["input"]["roi_xs"].to(device)
        processed_sample["roi_ys"] = batch_sample["input"]["roi_ys"].to(device)

        # processed_sample["sym_info"] = batch_sample["sym_info"]  # [bs, 4]
        # rgb_view = batch_sample["roi_rgb"][0]  # 测试数据
        processed_sample["main_rgb"] = batch_sample["input"]["rgb_group"]["main"].to(
            device
        )  # [bs, 3, 224, 224]已归一化
        processed_sample["top_rgb"] = batch_sample["input"]["rgb_group"]["top"].to(
            device
        )
        processed_sample["left_rgb"] = batch_sample["input"]["rgb_group"]["left"].to(
            device
        )
        processed_sample["main_mask"] = batch_sample["input"]["map_group"][
            "main_mask"
        ].to(device)
        processed_sample["main_map"] = (
            batch_sample["input"]["map_group"]["main"].to(device).permute(0, 3, 1, 2)
        )
        processed_sample["top_map"] = (
            batch_sample["input"]["map_group"]["top"].to(device).permute(0, 3, 1, 2)
        )
        processed_sample["left_map"] = (
            batch_sample["input"]["map_group"]["left"].to(device).permute(0, 3, 1, 2)
        )
        assert (
            processed_sample["main_rgb"].shape[-1]
            == processed_sample["main_rgb"].shape[-2]
        )
        assert (
            processed_sample["top_rgb"].shape[-1]
            == processed_sample["top_rgb"].shape[-2]
        )
        assert (
            processed_sample["left_rgb"].shape[-1]
            == processed_sample["left_rgb"].shape[-2]
        )
        assert processed_sample["main_rgb"].shape[-1] % 14 == 0
        assert processed_sample["top_rgb"].shape[-1] % 14 == 0
        assert processed_sample["left_rgb"].shape[-1] % 14 == 0
        # processed_sample["roi_xs"] = batch_sample["input"]["roi_xs"].to(
        #     device
        # )  # [bs, 1024]
        # processed_sample["roi_ys"] = batch_sample["input"]["roi_ys"].to(
        #     device
        # )  # [bs, 1024]
        # processed_sample["roi_center_dir"] = batch_sample["input"]["roi_center_dir"].to(
        #     device
        # )  # [bs, 3]
        # [bs, 6] SE(3) se3
        # processed_sample["gt_quaternion"] = batch_sample["label"]["quaternion"].to(
        #     device
        # )
        processed_sample["gt_rotation"] = batch_sample["label"]["rotation"].to(device)
        # processed_sample["gt_translation"] = batch_sample["label"]["translation"].to(
        #     device
        # )
        # translation 为已归一化位移
        processed_sample["raw_translation"] = batch_sample["label"][
            "raw_translation"
        ].to(device)
        processed_sample["gt_pose"] = torch.cat(
            [batch_sample["label"]["quaternion"], batch_sample["label"]["translation"]],
            dim=1,
        ).to(device)

        """ zero center """
        # num_pts = processed_sample["pts"].shape[1]
        # zero_mean = torch.mean(processed_sample["pts"][:, :, :3], dim=1)
        # processed_sample["zero_mean_pts"] = copy.deepcopy(processed_sample["pts"])
        # processed_sample["zero_mean_pts"][:, :, :3] -= zero_mean.unsqueeze(1).repeat(
        #     1, num_pts, 1
        # )
        # processed_sample["zero_mean_gt_pose"] = copy.deepcopy(
        #     processed_sample["gt_pose_se3"]
        # )
        # processed_sample["zero_mean_gt_pose"][:, -3:] -= zero_mean
        # processed_sample["pts_center"] = zero_mean
        processed_sample["trans_mean"] = batch_sample["meta"]["trans_mean"]
        processed_sample["trans_std"] = batch_sample["meta"]["trans_std"]
        return processed_sample

    @staticmethod
    def save_dataset_2_pkl(cfg, save_name, dataset_type: str = "xyzibd"):
        save_path = os.path.join("data", f"{save_name}.pkl")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        assert dataset_type in {"xyzibd"}, f"不支持的数据集: {dataset_type}"
        DATASET = dataset_type_mapping[dataset_type]

        train_ds = DATASET.get_train_dataset(cfg)
        val_ds = DATASET.get_val_dataset(cfg)
        test_ds = DATASET.get_test_dataset(cfg)
        dataset = {"train_ds": train_ds, "val_ds": val_ds, "test_ds": test_ds}
        with open(save_path, "wb") as f:
            pickle.dump(dataset, f)
        print(f"保存 {dataset_type} 数据集，类别{cfg.target_obj_id} 完成")

    @staticmethod
    def get_dataset_from_pkl(pkl_path):
        with open(pkl_path, "rb") as f:
            datasets = pickle.load(f)
        return datasets
