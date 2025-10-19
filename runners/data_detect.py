from copy import deepcopy
import sys
import os
import torch
import numpy as np
import random
from tqdm import tqdm
import open3d as o3d


APPEND_PATH = os.path.dirname(os.path.dirname(__file__))
sys.path.append(APPEND_PATH)
from datasets_my.dataset_port import DatasetPort
from configs.config import get_config
from datasets_my.xyzibd_dataset import XyzibdDataset
from utils_my.data_utils import DataUtils
from utils_my.point_cloud_utils import PcdO3dUtils
from utils_my.point_cloud_utils import PcdO3dUtils, PointCloudUtils as PclUtils
from utils_my.pose_utils import PoseUtils


def cfg_add(cfg):
    cfg.data_root = "/mnt/e/ai/data/xyzibd"
    # cfg.target_obj_id = -1
    cfg.target_obj_id = 1
    cfg.verbose = True
    cfg.sample_ratio = 1
    cfg.load_per_object = True

    return cfg


def set_random(cfg):
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)


def get_dataloader(data_type):
    from datasets_my.dataset_port import DatasetPort

    data_loaders = DatasetPort.get_data_loaders_from_cfg(
        cfg, dataset_type="xyzibd", data_type=[data_type]
    )
    dataloader = data_loaders[f"{data_type}_loader"]
    print(f"{data_type}_set: ", len(dataloader))

    return dataloader


cfg = get_config()
cfg = cfg_add(cfg)
set_random(cfg)


def __getitem__(dataset: XyzibdDataset, idx):

    data = dataset.samples[idx]
    img_path = data["img_path"]
    depth_path = data["depth_path"]
    mask_visib_path = data["mask_visib_path"]

    rgb = DataUtils.read_rgb(img_path)
    depth = DataUtils.read_depth(depth_path, data["depth_scale"])

    mask_visib = DataUtils.read_mask_visib(mask_visib_path)
    assert mask_visib.shape[:2] == depth.shape[:2] == rgb.shape[:2], "invalid data"
    im_h, im_w, _ = rgb.shape
    mat_K = data["K"]

    roi_rgb, roi_depth, roi_mask, roi_K, roi_coord_2d, bbox_center = DataUtils.cut_roi(
        rgb, depth, mask_visib, mat_K, dataset.cfg
    )

    filter_rgb, filter_depth = DataUtils.roi_filter(roi_rgb, roi_depth, roi_mask)

    valid_depth_mask = filter_depth > 0
    roi_mask_def = DataUtils.defor_2D(
        roi_mask,
        rand_r=dataset.deform_2d_params["roi_mask_r"],
        rand_pro=dataset.deform_2d_params["roi_mask_pro"],
    )
    # roi_depth = np.expand_dims(roi_depth, axis=0)
    # valid = (np.squeeze(roi_depth, axis=0) > 0) * roi_mask_def > 0
    valid = (roi_depth > 0) * roi_mask_def > 0
    valid_coords = np.argwhere(valid_depth_mask)  # (N, 2)，格式：[ys, xs]（行, 列）
    xs, ys = np.argwhere(valid).transpose(1, 0)
    valid = valid.reshape(-1)
    pcl_in = PclUtils.depth_to_pcl(roi_depth, roi_K, roi_coord_2d, valid)
    if len(pcl_in) < 1024:
        print(f"点云点数小：{len(pcl_in)}")
    if len(pcl_in) == 0:
        return dataset.__getitem__((idx + 1) % dataset.__len__())
    pcd_main = PcdO3dUtils.o3d_create_pcd(filter_depth, roi_K)

    valid_colors = roi_rgb[valid_coords[:, 0], valid_coords[:, 1]]  # [ys, xs]索引
    pcd_main.colors = o3d.utility.Vector3dVector(valid_colors / 255.0)

    # print(f"主视图点云：{len(pcd_main.points)}个点")
    # -----------------生成另外的视图

    # main_map, main_mask = DataUtils.convert_valid_coords_to_dense_map(
    #     valid_coords, height=dataset.img_size, width=dataset.img_size, invalid_value=-1
    # )
    # top_rgb, top_map = DataUtils.transformer_image_view(
    #     roi_rgb, valid_coords, np.asarray(pcd_main.points), roi_K, -20, "x"
    # )
    # left_rgb, left_map = DataUtils.transformer_image_view(
    #     roi_rgb, valid_coords, np.asarray(pcd_main.points), roi_K, 20, "y"
    # )
    # -------------------点云可视化（验证旋转正确性）测试用
    # cam_coord, cam_intri = PcdO3dUtils.o3d_get_camera_coord(roi_K, (H, W))
    # frustum = PcdO3dUtils.create_camera_frustum(cam_intri, scale=1)
    # o3d.visualization.draw_geometries(
    #     [pcd_main, top_result["pcd"], left_result["pcd"], cam_coord, frustum],
    #     window_name="main_view + top_view 15° + left_view_15°",
    #     width=1200,
    #     height=800,
    # )
    # 该归一化的归一化
    main_rgb_norm = DataUtils.rgb_transform(roi_rgb)
    # top_rgb_norm = DataUtils.rgb_transform(top_rgb)
    # left_rgb_norm = DataUtils.rgb_transform(left_rgb)
    # -----------------------
    # point_sample = PointCloudSample(np.asarray(pcd_main.points), 1024, valid_coords)
    # pcd_in_idx, pcd_in = point_sample.random_sample()
    pcd_in_idx, pcd_in = PclUtils.random_sample_points(pcl_in, 1024)
    xs, ys = xs[pcd_in_idx], ys[pcd_in_idx]
    # 加载标签
    rotation = data["R"]
    translation = data["t"].flatten() / 1000
    # normalized_trans = (translation - dataset.trans_mean) / dataset.trans_std
    quaternion, _ = PoseUtils.convert_pose_2_quaternion(rotation, translation)
    quaternion = quaternion.numpy()[0]

    def set_sys_info(model_info: dict):
        sym_idx = {"none": 0, "any": 1, "half": 2, "quarter": 3}
        has_any_sym = 0  # 是否存在任何对称性（0：无，1：有）
        x_sym = "none"  # x轴对称性
        y_sym = "none"  # y轴对称性
        z_sym = "none"  # z轴对称性
        # 1. 先处理连续对称性（优先级不变）
        if "symmetries_continuous" in model_info:
            has_any_sym = 1
            for sym in model_info["symmetries_continuous"]:
                axis = sym.get("axis", [0, 0, 0])
                if abs(axis[0]) > 1e-6:
                    x_sym = "any"
                if abs(axis[1]) > 1e-6:
                    y_sym = "any"
                if abs(axis[2]) > 1e-6:
                    z_sym = "any"

        # 2. 处理离散对称性（升级为多矩阵分析）
        if "symmetries_discrete" in model_info and has_any_sym == 0:
            has_any_sym = 1
            sym_matrices = model_info["symmetries_discrete"]
            symmetry_axis = "none"  # 记录共性对称轴（x/y/z）

            # 步骤1：遍历所有矩阵，确定共性对称轴（如是否都绕z轴）
            for mat in sym_matrices:
                if len(mat) < 11:  # 确保矩阵长度足够（至少包含z轴旋转项）
                    continue
                # 提取3x3旋转子矩阵（聚焦x-y平面，因所有变换绕z轴）
                rot_x1, rot_x2 = mat[0], mat[1]  # x行的旋转项（对应y/z轴）
                rot_y1, rot_y2 = mat[4], mat[5]  # y行的旋转项（对应x/z轴）
                z_rot = mat[10]  # z轴的旋转/镜像项

                # 判断是否绕z轴旋转（x-y平面有旋转项，z轴无偏移）
                if (
                    abs(rot_x1) > 1e-6
                    or abs(rot_x2) > 1e-6
                    or abs(rot_y1) > 1e-6
                    or abs(rot_y2) > 1e-6
                ) and abs(z_rot) > 1e-6:
                    symmetry_axis = "z"  # 所有矩阵均绕z轴，确定共性轴为z

            # 步骤2：基于共性轴，判断对称类型（旋转/镜像）
            if symmetry_axis == "z":
                # 检测是否包含旋转对称（通过常见旋转角度的三角函数值判断）
                has_rotation = False
                for mat in sym_matrices:
                    rot_x1, rot_x2 = mat[0], mat[1]
                    rot_y1, rot_y2 = mat[4], mat[5]

                    # 识别常见旋转角度的矩阵特征（允许浮点误差）
                    # 1. 90度旋转：cos90=0, sin90=1 → 矩阵项为0/±1
                    is_90_rot = (
                        abs(rot_x1) < 1e-6
                        and abs(rot_x2) in [0.0, 1.0]
                        and abs(rot_y1) in [0.0, 1.0]
                        and abs(rot_y2) < 1e-6
                    )
                    # 2. 45度旋转：cos45=sin45≈0.7071
                    is_45_rot = (
                        abs(rot_x1 - 0.7071) < 1e-4
                        or abs(rot_x1 + 0.7071) < 1e-4
                        or abs(rot_x2 - 0.7071) < 1e-4
                        or abs(rot_x2 + 0.7071) < 1e-4
                    )
                    # 3. 22.5度旋转：cos22.5≈0.9238, sin22.5≈0.3826
                    is_225_rot = (
                        abs(rot_x1 - 0.9238) < 1e-4
                        or abs(rot_x1 + 0.9238) < 1e-4
                        or abs(rot_x2 - 0.3826) < 1e-4
                        or abs(rot_x2 + 0.3826) < 1e-4
                    )

                    if is_90_rot or is_45_rot or is_225_rot:
                        has_rotation = True
                        break

                # 若有旋转对称，映射到quarter；若无则判断是否为镜像（half）
                if has_rotation:
                    z_sym = "quarter"
                else:
                    # 检测z轴镜像（矩阵第10个元素为-1.0）
                    for mat in sym_matrices:
                        if abs(mat[10] + 1.0) < 1e-6:
                            z_sym = "half"
                            break
        sym_info = [has_any_sym, sym_idx[x_sym], sym_idx[y_sym], sym_idx[z_sym]]
        return sym_info

    sym_info = set_sys_info(data["model_info"])

    # new_rotation = PoseUtils.convert_quaternion_2_rotation(quaternion)
    # new_rotation = new_rotation.numpy()

    # 点云可视化（验证旋转正确性）测试用
    # cam_coord, cam_intri = PcdO3dUtils.o3d_get_camera_coord(roi_K, (H, W))
    # orgin_shape_h, orgin_shape_w = rgb.shape[0], rgb.shape[1]
    # cam_coord, cam_intri = PcdO3dUtils.o3d_get_camera_coord(
    #     mat_K, (orgin_shape_h, orgin_shape_w)
    # )
    # cam_coord: o3d.geometry.TriangleMesh = cam_coord
    # object_coord: o3d.geometry.TriangleMesh = (
    #     o3d.geometry.TriangleMesh.create_coordinate_frame(size=500)
    # )
    # object_coord.rotate(rotation)
    # object_coord.translate(translation)
    # raw_pointcloud = PcdO3dUtils.o3d_create_pcd(depth, mat_K)
    # from utils.point_cloud_utils import PointCloudUtils

    # raw_pointcloud_np = PointCloudUtils.transform_depth2pcd(depth, mat_K)
    # raw_pointcloud = o3d.geometry.PointCloud()
    # raw_pointcloud.points = o3d.utility.Vector3dVector(raw_pointcloud_np)
    # frustum = PcdO3dUtils.create_camera_frustum(cam_intri, scale=1)
    # o3d.visualization.draw_geometries(
    #     # [pcd_main, cam_coord, frustum, object_coord, raw_pointcloud],
    #     [pcd_main, cam_coord, object_coord, raw_pointcloud],
    #     window_name="main_view + top_view 15° + left_view_15°",
    #     width=1200,
    #     height=800,
    # )
    # ====== 组合数据 ======
    input_dict, label_dict, meta_dict = {}, {}, {}

    """---------- input_dict ----------"""
    input_dict["pcl_in"] = torch.as_tensor(pcd_in.astype(np.float32)).contiguous()
    input_dict["pcl_in_index"] = torch.as_tensor(
        pcd_in_idx.astype(np.int64)
    ).contiguous()
    input_dict["roi_xs"] = torch.as_tensor(
        np.ascontiguousarray(xs), dtype=torch.int64
    ).contiguous()
    input_dict["roi_ys"] = torch.as_tensor(
        np.ascontiguousarray(ys), dtype=torch.int64
    ).contiguous()
    input_dict["rgb_group"] = {}
    input_dict["rgb_group"]["main"] = torch.as_tensor(
        np.ascontiguousarray(main_rgb_norm), dtype=torch.float32
    ).contiguous()
    # input_dict["rgb_group"]["top"] = torch.as_tensor(
    #     np.ascontiguousarray(top_rgb_norm), dtype=torch.float32
    # ).contiguous()
    # input_dict["rgb_group"]["left"] = torch.as_tensor(
    #     np.ascontiguousarray(left_rgb_norm), dtype=torch.float32
    # ).contiguous()
    # input_dict["map_group"] = {}
    # input_dict["map_group"]["main_mask"] = torch.as_tensor(
    #     np.ascontiguousarray(main_mask), dtype=torch.float32
    # ).contiguous()
    # input_dict["map_group"]["main"] = torch.as_tensor(
    #     np.ascontiguousarray(main_map), dtype=torch.float32
    # ).contiguous()
    # input_dict["map_group"]["top"] = torch.as_tensor(
    #     np.ascontiguousarray(top_map), dtype=torch.float32
    # ).contiguous()
    # input_dict["map_group"]["left"] = torch.as_tensor(
    #     np.ascontiguousarray(left_map), dtype=torch.float32
    # ).contiguous()

    # input_dict["roi_xs"] = torch.as_tensor(
    #     np.ascontiguousarray(xs), dtype=torch.int64
    # ).contiguous()
    # input_dict["roi_ys"] = torch.as_tensor(
    #     np.ascontiguousarray(ys), dtype=torch.int64
    # ).contiguous()

    """---------- label ----------"""
    label_dict["rotation"] = torch.as_tensor(rotation, dtype=torch.float32).contiguous()
    label_dict["raw_translation"] = torch.as_tensor(
        translation, dtype=torch.float32
    ).contiguous()
    label_dict["translation"] = torch.as_tensor(
        translation, dtype=torch.float32
    ).contiguous()

    label_dict["quaternion"] = torch.as_tensor(
        quaternion, dtype=torch.float32
    ).contiguous()

    """---------- meta ----------"""
    # translation 归一化参数
    meta_dict["trans_mean"] = torch.as_tensor(
        dataset.trans_mean, dtype=torch.float32
    ).contiguous()
    meta_dict["trans_std"] = torch.as_tensor(
        dataset.trans_std, dtype=torch.float32
    ).contiguous()
    meta_dict["visib_fract"] = data["visib_fract"]
    meta_dict["mat_K"] = torch.as_tensor(
        mat_K, dtype=torch.float32
    ).contiguous()  # 相机内参矩阵
    # meta_dict["bbox_side_len"] = torch.as_tensor(
    #     data["bbox_side_len"], dtype=torch.float32
    # ).contiguous()
    meta_dict["scene_id"] = data["scene_id"]
    meta_dict["cam_id"] = data["cam_id"]
    meta_dict["im_id"] = data["im_id"]

    def pixel2xyz(h: int, w: int, pixel: np.ndarray, mat_K):
        """
        Transform `(pixel[0], pixel[1])` to normalized 3D vector under cv space, using camera intrinsics.

        :param h: height of the actual image
        :param w: width of the actual image
        """

        # scale camera parameters
        scale_x = w / im_w
        scale_y = h / im_h
        fx = mat_K[0, 0]
        fy = mat_K[1, 1]
        cx = mat_K[0, 2]
        cy = mat_K[1, 2]
        fx = fx * scale_x
        fy = fy * scale_y
        x_offset = cx * scale_x
        y_offset = cy * scale_y

        x = (pixel[1] - x_offset) / fx
        y = (pixel[0] - y_offset) / fy
        vec = np.array([x, y, 1])
        return vec / np.linalg.norm(vec)

    roi_center_dir = pixel2xyz(im_h, im_w, bbox_center, mat_K)

    affine = torch.eye(4)
    affine[:3, :3] = label_dict["rotation"]
    affine[:3, 3] = label_dict["translation"]
    intrinsics_list = [
        mat_K[0, 0],
        mat_K[1, 1],
        mat_K[0, 2],
        mat_K[1, 2],
        im_w,
        im_h,
    ]
    data_dict = {
        # "input": input_dict,
        # "label": label_dict,
        # "meta": meta_dict,
        "intrinsics": torch.as_tensor(
            intrinsics_list, dtype=torch.float32
        ).contiguous(),
        "path": img_path,
        "class_label": data["obj_id"],
        "pcl_in": input_dict["pcl_in"],
        "rotation": label_dict["rotation"],
        "translation": label_dict["translation"],
        "sym_info": torch.as_tensor(sym_info, dtype=torch.int8).contiguous(),
        "roi_rgb": input_dict["rgb_group"]["main"],
        "roi_xs": input_dict["roi_xs"],
        "roi_ys": input_dict["roi_ys"],
        "roi_center_dir": torch.as_tensor(
            roi_center_dir, dtype=torch.int8
        ).contiguous(),
        "affine": torch.as_tensor(affine, dtype=torch.float32).contiguous(),
        "bbox_side_len": torch.as_tensor(
            data["bbox_side_len"], dtype=torch.float32
        ).contiguous(),
    }

    return data_dict, len(pcl_in)


def detect_point_in(dataset: XyzibdDataset):
    min_pcls = []
    min_512 = []
    for i in tqdm(range(0, len(dataset.samples))):
        data, pcl_in_num = __getitem__(dataset, i)
        if pcl_in_num < 1024:
            min_pcls.append(pcl_in_num)
        if pcl_in_num < 512:
            min_512.append(pcl_in_num)
    print(f"小于1024的点数量: {len(min_pcls)}")
    print(f"最小的点，点个数: {min(min_pcls)}")
    print(f"小于500个点的数量: {len(min_512)}")


def main():
    train_loader = get_dataloader("train")
    train_dataset = deepcopy(train_loader.dataset.dataset)
    del train_loader
    # val_loader = get_dataloader("val")
    # val_dataset: XyzibdDataset = deepcopy(val_loader.dataset.dataset)
    # del val_loader
    detect_point_in(train_dataset)

    # for i, test_batch in enumerate(val_loader):
    #     batch_sample = DatasetPort.process_batch_xyzibd(
    #         test_batch,
    #         cfg.device,
    #         pose_mode=cfg.pose_mod,
    #     )
    #     print(f"batch_sample")


if __name__ == "__main__":
    main()

"""
train_dataset:
    样本数: 114800
    小于 1024 点云数: 698
    小于 512 点云数: 47
    最小点云数: 336
val_dataset:
    样本数: 8850
    小于 1024 点云数: 133
    小于 512 点云数: 4
    最小点云，点个数: 318

"""
