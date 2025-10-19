import json
import os
import random
import cv2
import numpy as np
import torch
from torch.utils import data
import open3d as o3d

from utils_my.data_utils import DataUtils, find_scenes
from utils_my.pose_utils import PoseUtils
from utils_my.point_cloud_utils import PcdO3dUtils, PointCloudUtils as PclUtils
from utils_my.point_cloud_utils import PointCloudSample


def compute_translation_stats(dataset):
    """计算整个训练集的平移量统计量"""
    all_trans = []
    for i in range(len(dataset)):
        data = dataset[i]
        # 获取原始平移量
        raw_trans = data["translation"].numpy()
        all_trans.append(raw_trans)

    all_trans = np.vstack(all_trans)  # [N, 3]
    trans_mean = np.mean(all_trans, axis=0)  # [3]
    trans_std = np.std(all_trans, axis=0)  # [3]

    # 防止除零错误
    trans_std[trans_std < 1e-8] = 1.0

    return trans_mean, trans_std


class XyzibdDataset(data.Dataset):
    def __init__(
        self,
        cfg,
        cam_ids=None,  # 相机 ID 列表，仅用于验证集
        # target_size=256,  # 目标图像大小
        img_size=256,  # 目标图像大小
        # img_size=224,  # 目标图像大小
        augment=False,  # 是否启用数据增强
        split="train",  # 数据集分割类型："train"、"val"、"test"
        trans_mean=None,
        trans_std=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.root_dir = cfg.data_root  # 数据集根目录
        self.obj_id: int = cfg.target_obj_id  # 目标物体 ID，0时为全部数据
        self.cam_ids = cam_ids
        self.img_size = img_size
        self.augment = augment
        self.dynamic_zoom_in_params = cfg.DYNAMIC_ZOOM_IN_PARAMS
        self.deform_2d_params = cfg.DEFORM_2D_PARAMS
        self.pts_sample_num = 1024  # 随机采样
        self.split = split
        self.dataset_path = self.get_dataset_path()
        self.scene_ids = find_scenes(self.dataset_path)
        # self.max_per_scene = cfg.max_per_scene

        print(
            f"[INFO] 找到 {len(self.scene_ids)} 个 {self.split} 场景,ID：\n {self.scene_ids}"
        )

        self.samples: list = self.get_sample()

        self.trans_mean = trans_mean
        self.trans_std = trans_std
        if self.trans_mean is None:
            self.trans_mean = np.zeros(3)
        if self.trans_std is None:
            self.trans_std = np.ones(3)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]
        img_path = data["img_path"]
        depth_path = data["depth_path"]
        mask_visib_path = data["mask_visib_path"]

        rgb = DataUtils.read_rgb(img_path)
        depth = DataUtils.read_depth(depth_path, data["depth_scale"])

        mask_visib = DataUtils.read_mask_visib(mask_visib_path)
        assert mask_visib.shape[:2] == depth.shape[:2] == rgb.shape[:2], "invalid data"
        im_h, im_w, _ = rgb.shape
        mat_K = data["K"]

        roi_rgb, roi_depth, roi_mask, roi_K, roi_coord_2d, bbox_center = (
            DataUtils.cut_roi(rgb, depth, mask_visib, mat_K, self.cfg)
        )

        filter_rgb, filter_depth = DataUtils.roi_filter(roi_rgb, roi_depth, roi_mask)

        valid_depth_mask = filter_depth > 0
        roi_mask_def = DataUtils.defor_2D(
            roi_mask,
            rand_r=self.deform_2d_params["roi_mask_r"],
            rand_pro=self.deform_2d_params["roi_mask_pro"],
        )
        # roi_depth = np.expand_dims(roi_depth, axis=0)
        # valid = (np.squeeze(roi_depth, axis=0) > 0) * roi_mask_def > 0
        valid = (roi_depth > 0) * roi_mask_def > 0
        valid_coords = np.argwhere(valid_depth_mask)  # (N, 2)，格式：[ys, xs]（行, 列）
        xs, ys = np.argwhere(valid).transpose(1, 0)
        valid = valid.reshape(-1)
        pcl_in = PclUtils.depth_to_pcl(roi_depth, roi_K, roi_coord_2d, valid)
        if len(pcl_in) == 0:
            return self.__getitem__((idx + 1) % self.__len__())
        pcd_main = PcdO3dUtils.o3d_create_pcd(filter_depth, roi_K)

        valid_colors = roi_rgb[valid_coords[:, 0], valid_coords[:, 1]]  # [ys, xs]索引
        pcd_main.colors = o3d.utility.Vector3dVector(valid_colors / 255.0)

        # print(f"主视图点云：{len(pcd_main.points)}个点")
        # -----------------生成另外的视图

        # main_map, main_mask = DataUtils.convert_valid_coords_to_dense_map(
        #     valid_coords, height=self.img_size, width=self.img_size, invalid_value=-1
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
        # normalized_trans = (translation - self.trans_mean) / self.trans_std
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
        label_dict["rotation"] = torch.as_tensor(
            rotation, dtype=torch.float32
        ).contiguous()
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
            self.trans_mean, dtype=torch.float32
        ).contiguous()
        meta_dict["trans_std"] = torch.as_tensor(
            self.trans_std, dtype=torch.float32
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
        return data_dict

    def get_dataset_path(self):
        dataset_dir_name = {"train": "train_pbr", "val": "val", "test": "test"}
        return os.path.join(self.root_dir, dataset_dir_name[self.split])

    def get_sample(self):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"目录 '{self.dataset_path}' 不存在于 {self.root_dir} 中！"
            )
        get_sample_mapping = {
            "train": self.get_train_samples,
            "val": self.get_val_samples,
            # "test": self.get_test_samples,
        }
        get_sample = get_sample_mapping[self.split]
        all_samples = get_sample()

        return all_samples

    def get_train_samples(self):
        # 收集所有 object 样本
        all_samples = []
        models_path = os.path.join(self.root_dir, "models")

        for sid in self.scene_ids:
            scene_path = os.path.join(self.dataset_path, sid)
            scene_count = 0
            info_file = os.path.join(scene_path, f"scene_gt_info.json")
            pose_file = os.path.join(scene_path, f"scene_gt.json")
            cam_file = os.path.join(scene_path, f"scene_camera.json")
            model_file = os.path.join(models_path, "models_info.json")

            gray_dir = os.path.join(scene_path, f"gray")
            depth_dir = os.path.join(scene_path, f"depth")
            mask_visib_dir = os.path.join(scene_path, f"mask_visib")
            if not all(
                os.path.exists(f)
                for f in [
                    info_file,
                    pose_file,
                    cam_file,
                    model_file,
                    gray_dir,
                    depth_dir,
                    mask_visib_dir,
                ]
                # os.path.exists(f)
                # for f in [info_file, pose_file, cam_file]
            ):
                continue
            with open(info_file, "r") as f1, open(pose_file, "r") as f2, open(
                cam_file, "r"
            ) as f3, open(model_file, "r") as f4:
                info_json = json.load(f1)
                pose_json = json.load(f2)
                cam_json = json.load(f3)
                model_json = json.load(f4)
            all_im_ids = sorted(info_json.keys(), key=lambda x: int(x))
            for im_id_s in all_im_ids:
                im_id = int(im_id_s)
                if im_id_s not in cam_json:
                    continue
                K = np.array(cam_json[im_id_s]["cam_K"], dtype=np.float32).reshape(3, 3)
                depth_scale = cam_json[im_id_s]["depth_scale"]
                img_name_jpg = os.path.join(gray_dir, f"{im_id:06d}.jpg")
                img_name_png = os.path.join(gray_dir, f"{im_id:06d}.png")
                depth_name_png = os.path.join(depth_dir, f"{im_id:06d}.png")
                mask_visib_png_prefix = os.path.join(mask_visib_dir, f"{im_id:06d}_")

                # Check if JPG exists, otherwise use PNG.
                if os.path.exists(img_name_jpg):
                    img_path = img_name_jpg
                elif os.path.exists(img_name_png):
                    img_path = img_name_png
                else:
                    continue  # Skip if neither format is found.
                if os.path.exists(depth_name_png):
                    depth_path = depth_name_png

                # 处理所有对象实例
                for idx, (inf, pos) in enumerate(
                    zip(info_json[im_id_s], pose_json[im_id_s])
                ):
                    if pos["obj_id"] != self.obj_id and self.obj_id > 0:
                        continue
                    model_info = model_json[str(pos["obj_id"])]
                    bbox_side_len = [
                        model_info["size_x"] / 1000,
                        model_info["size_y"] / 1000,
                        model_info["size_z"] / 1000,
                    ]

                    mask_visib_path = mask_visib_png_prefix + f"{idx:06d}.png"

                    # 可见边界框过滤
                    x, y, w_, h_ = inf["bbox_visib"]
                    if w_ <= 0 or h_ <= 0:
                        continue

                    # 获取BOP挑战指标
                    visib_fract = inf.get("visib_fract", 1.0)
                    px_count_all = inf.get("px_count_all", w_ * h_)
                    px_count_valid = inf.get("px_count_valid", px_count_all)

                    # Apply filtering thresholds.
                    if visib_fract < 0.1 or px_count_valid < 1000:
                        continue  # 可见性极低

                    R_mat = np.array(pos["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
                    t = np.array(pos["cam_t_m2c"], dtype=np.float32).reshape(3, 1)
                    all_samples.append(
                        {
                            "scene_id": sid,
                            # "cam_id": cam_id,
                            "obj_id": pos["obj_id"],
                            "im_id": im_id,  # 图片在场景scene_id中的序号
                            "img_path": img_path,
                            "depth_path": depth_path,
                            "mask_visib_path": mask_visib_path,
                            "K": K,
                            "depth_scale": depth_scale,
                            "R": R_mat,
                            "t": t,
                            "bbox_visib": [x, y, w_, h_],  # 可见物体的边界框
                            "bbox_obj": inf["bbox_visib"],  # 物体的整个边界框
                            "visib_fract": visib_fract,
                            "px_count_all": px_count_all,
                            "px_count_valid": px_count_valid,  # 物体有深度的像素数量
                            "model_info": model_info,
                            "bbox_side_len": bbox_side_len,
                            # "visib_fract": inf["visib_fract"],  # 可见像素/全部像素
                        }
                    )
            scene_count += 1
            # 可以设置少训练几个场景试试水
            # if self.max_per_scene > 0 and scene_count >= self.max_per_scene:
            #     break
        # del all_im_ids, cam_json, info_json, pose_json

        from collections import defaultdict

        groups = defaultdict(list)  # list[(场景id，相机id)]
        for s in all_samples:
            s["cam_id"] = "sim" if self.split == "train" else s["cam_id"]
            key = (s["scene_id"], s["cam_id"])
            groups[key].append(s)

        del all_samples

        final_samples = []

        for group_samples in groups.values():
            random.shuffle(group_samples)
            final_samples.extend(group_samples)
        return final_samples

    def get_val_samples(self):
        all_samples = []
        models_path = os.path.join(self.root_dir, "models")
        for sid in self.scene_ids:
            scene_path = os.path.join(self.dataset_path, sid)
            scene_count = 0
            for cam_id in self.cam_ids:
                info_file = os.path.join(scene_path, f"scene_gt_info_{cam_id}.json")
                pose_file = os.path.join(scene_path, f"scene_gt_{cam_id}.json")
                cam_file = os.path.join(scene_path, f"scene_camera_{cam_id}.json")
                model_file = os.path.join(models_path, "models_info.json")
                if cam_id == "realsense":
                    bgr_dir = os.path.join(scene_path, f"rgb_{cam_id}")
                else:
                    bgr_dir = os.path.join(scene_path, f"gray_{cam_id}")
                depth_dir = os.path.join(scene_path, f"depth_{cam_id}")
                mask_visib_dir = os.path.join(scene_path, f"mask_visib_{cam_id}")
                current_files = [
                    info_file,
                    pose_file,
                    cam_file,
                    model_file,
                    bgr_dir,
                    depth_dir,
                    mask_visib_dir,
                ]
                missing_files = [f for f in current_files if not os.path.exists(f)]
                if missing_files:
                    print(
                        f"[WARNING] 跳过场景 {sid} 相机 {cam_id}，缺少必要文件或目录：{missing_files}"
                    )
                    continue
                with open(info_file, "r") as f1, open(pose_file, "r") as f2, open(
                    cam_file, "r"
                ) as f3, open(model_file, "r") as f4:
                    info_json = json.load(f1)
                    pose_json = json.load(f2)
                    cam_json = json.load(f3)
                    model_json = json.load(f4)

                all_im_ids = sorted(info_json.keys(), key=lambda x: int(x))
                for im_id_s in all_im_ids:
                    im_id = int(im_id_s)
                    if im_id_s not in cam_json:
                        continue
                    K = np.array(cam_json[im_id_s]["cam_K"], dtype=np.float32).reshape(
                        3, 3
                    )
                    depth_scale = cam_json[im_id_s]["depth_scale"]
                    img_name_jpg = os.path.join(bgr_dir, f"{im_id:06d}.jpg")
                    img_name_png = os.path.join(bgr_dir, f"{im_id:06d}.png")
                    depth_name_png = os.path.join(depth_dir, f"{im_id:06d}.png")
                    mask_visib_png_prefix = os.path.join(
                        mask_visib_dir, f"{im_id:06d}_"
                    )

                    # Check if JPG exists, otherwise use PNG.
                    if os.path.exists(img_name_jpg):
                        img_path = img_name_jpg
                    elif os.path.exists(img_name_png):
                        img_path = img_name_png
                    else:
                        continue  # Skip if neither format is found.
                    if os.path.exists(depth_name_png):
                        depth_path = depth_name_png

                    # Loop through all object instances in the image.
                    for idx, (inf, pos) in enumerate(
                        zip(info_json[im_id_s], pose_json[im_id_s])
                    ):
                        if pos["obj_id"] != self.obj_id and self.obj_id >= 0:
                            continue
                        model_info = model_json[str(pos["obj_id"])]
                        bbox_side_len = [
                            model_info["size_x"] / 1000,
                            model_info["size_y"] / 1000,
                            model_info["size_z"] / 1000,
                        ]
                        mask_visib_path = mask_visib_png_prefix + f"{idx:06d}.png"

                        # Use bbox_visib for the visible part of the object.
                        x, y, w_, h_ = inf["bbox_visib"]
                        if w_ <= 0 or h_ <= 0:
                            continue

                        # Retrieve additional BOP challenge metrics if available.
                        visib_fract = inf.get("visib_fract", 1.0)
                        px_count_all = inf.get("px_count_all", w_ * h_)
                        px_count_valid = inf.get("px_count_valid", px_count_all)

                        # Apply filtering thresholds.
                        if visib_fract < 0.1 or px_count_valid < 1000:
                            continue

                        R_mat = np.array(pos["cam_R_m2c"], dtype=np.float32).reshape(
                            3, 3
                        )
                        t = np.array(pos["cam_t_m2c"], dtype=np.float32).reshape(3, 1)

                        all_samples.append(
                            {
                                "scene_id": sid,
                                "cam_id": cam_id,
                                "obj_id": pos["obj_id"],
                                "im_id": im_id,  # 图片在场景scene_id中的序号
                                "img_path": img_path,
                                "depth_path": depth_path,
                                "mask_visib_path": mask_visib_path,
                                "K": K,
                                "depth_scale": depth_scale,
                                "R": R_mat,
                                "t": t,
                                "bbox_visib": [x, y, w_, h_],  # 可见物体的边界框
                                "bbox_obj": inf["bbox_visib"],  # 物体的整个边界框
                                "visib_fract": visib_fract,
                                "px_count_all": px_count_all,
                                "px_count_valid": px_count_valid,  # 物体有深度的像素数量
                                "model_info": model_info,
                                "bbox_side_len": bbox_side_len,
                                # --------------------------------
                                # "scene_id": sid,
                                # "cam_id": cam_id,
                                # "im_id": im_id,
                                # "img_path": img_path,
                                # "depth_path": depth_path,
                                # "mask_visib_path": mask_visib_path,
                                # "K": K,
                                # "R": R_mat,
                                # "t": t,
                                # "bbox_visib": [x, y, w_, h_],
                                # # "visib_fract": visib_fract,
                                # # "px_count_all": px_count_all,
                                # "px_count_valid": px_count_valid,
                            }
                        )
                scene_count += 1
                # if self.max_per_scene is not None and scene_count >= self.max_per_scene:
                #     break
        return all_samples

    def get_test_samples(self):
        all_samples = []
        for sid in self.scene_ids:
            scene_path = os.path.join(self.dataset_path, sid)
            scene_count = 0
            for cam_id in self.cam_ids:
                cam_file = os.path.join(scene_path, f"scene_camera_{cam_id}.json")
                # test_targets_multiview_bop25_file = os.path.join(
                #     self.root_dir, "test_targets_multiview_bop25.json"
                # )
                test_targets_bop24 = os.path.join(
                    self.root_dir, "test_targets_bop24.json"
                )

                if cam_id == "realsense":
                    bgr_dir = os.path.join(scene_path, f"rgb_{cam_id}")
                else:
                    bgr_dir = os.path.join(scene_path, f"gray_{cam_id}")
                depth_dir = os.path.join(scene_path, f"depth_{cam_id}")
                current_files = [
                    cam_file,
                    bgr_dir,
                    depth_dir,
                ]
                missing_files = [f for f in current_files if not os.path.exists(f)]
                if missing_files:
                    print(
                        f"[WARNING] 跳过场景 {sid} 相机 {cam_id}，缺少必要文件或目录：{missing_files}"
                    )
                    continue
                with open(cam_file, "r") as f1, open(test_targets_bop24, "r") as f2:
                    scene_camera_json = json.load(f1)
                    test_targets_bop24_json = json.load(f2)

                cam_ids = sorted(scene_camera_json.keys(), key=lambda x: int(x))
                for im_id_s in cam_ids:
                    im_id = int(im_id_s)
                    meta = scene_camera_json[im_id_s]
                    if im_id_s not in scene_camera_json:
                        continue

                    img_name_jpg = os.path.join(bgr_dir, f"{im_id:06d}.jpg")
                    img_name_png = os.path.join(bgr_dir, f"{im_id:06d}.png")
                    depth_name_png = os.path.join(depth_dir, f"{im_id:06d}.png")

                    # Check if JPG exists, otherwise use PNG.
                    if os.path.exists(img_name_jpg):
                        img_path = img_name_jpg
                    elif os.path.exists(img_name_png):
                        img_path = img_name_png
                    else:
                        continue  # Skip if neither format is found.
                    if os.path.exists(depth_name_png):
                        depth_path = depth_name_png
                    # 相机参数
                    K = np.array(meta["cam_K"], dtype=np.float32).reshape(3, 3)
                    R_mat = np.array(meta["cam_R_w2c"], dtype=np.float32).reshape(3, 3)
                    t = np.array(meta["cam_t_w2c"], dtype=np.float32).reshape(
                        3,
                    )
                    # Loop through all object instances in the image.
                    all_samples.append(
                        {
                            "scene_id": sid,
                            "cam_id": cam_id,
                            "im_id": im_id,
                            "img_path": img_path,
                            "depth_path": depth_path,
                            "depth_scale": meta["depth_scale"],
                            "K": K,
                            "R": R_mat,
                            "t": t,
                        }
                    )
                scene_count += 1
                # if self.max_per_scene is not None and scene_count >= self.max_per_scene:
                #     break
        return all_samples

    @staticmethod
    def get_mean_std(path, dataset):
        mean_file = dataset + "_trans_mean.npy"
        std_file = dataset + "_trans_std.npy"
        mean_path = os.path.join(path, mean_file)
        std_path = os.path.join(path, std_file)
        mean = np.load(mean_path)
        std = np.load(std_path)
        return mean, std

    @staticmethod
    def get_train_dataset(cfg):
        mean_std_path = "configs"

        trans_mean, trans_std = XyzibdDataset.get_mean_std(mean_std_path, "xyzibd")
        trans_mean, trans_std = None, None
        return XyzibdDataset(
            cfg,
            # img_size=224,
            img_size=256,
            augment=True,
            split="train",
            trans_mean=trans_mean,
            trans_std=trans_std,
        )

    @staticmethod
    def get_val_dataset(cfg):
        mean_std_path = "configs"
        trans_mean, trans_std = XyzibdDataset.get_mean_std(mean_std_path, "xyzibd")
        return XyzibdDataset(
            cfg,
            cam_ids=["xyz", "realsense", "photoneo"],
            img_size=256,
            augment=False,
            split="val",
            trans_mean=trans_mean,
            trans_std=trans_std,
        )

    @staticmethod
    def get_test_dataset(cfg):
        mean_std_path = "/home/dai/ai/SpherNetPose/config"
        trans_mean, trans_std = XyzibdDataset.get_mean_std(mean_std_path, "xyzibd")
        return XyzibdDataset(
            cfg,
            cam_ids=["xyz", "realsense", "photoneo"],
            img_size=224,
            augment=False,
            split="test",
            trans_mean=trans_mean,
            trans_std=trans_std,
        )
