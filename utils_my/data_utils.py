import os

import cv2
import numpy as np


def _get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def _get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def _get_affine_transform(
    center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=False
):
    """
    adapted from CenterNet: https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py
    center: ndarray: (cx, cy)
    scale: (w, h)
    rot: angle in deg
    output_size: int or (w, h)
    """
    if isinstance(center, (tuple, list)):
        center = np.array(center, dtype=np.float32)

    if isinstance(scale, (int, float)):
        scale = np.array([scale, scale], dtype=np.float32)

    if isinstance(output_size, (int, float)):
        output_size = (output_size, output_size)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = _get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = _get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def find_scenes(directory):
    """
    在给定目录中查找场景ID（假定场景子目录以数字命名）。
    """
    all_items = os.listdir(directory)
    scene_ids = [item for item in all_items if item.isdigit()]
    scene_ids.sort()

    return scene_ids


class DataUtils:

    @staticmethod
    def read_gray(path, is_rgb: bool = False):
        if is_rgb:
            rgb = DataUtils.read_rgb(path)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise IOError(f"无法读取：{path}")
        return gray

    @staticmethod
    def read_rgb(path):
        bgr = cv2.imread(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb is None:
            raise IOError(f"无法读取：{path}")
        return rgb

    @staticmethod
    def read_depth(path, scale):
        """
        返回深度， float32
        """
        depth = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) * scale
        if len(depth.shape) == 3:
            depth = depth[:, :, 0]
        if depth is None:
            raise IOError(f"无法读取：{path}")
        return depth / 1000  # 单位 m

    @staticmethod
    def read_mask_visib(path):
        mask_visib = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if len(mask_visib.shape) == 3:
            mask_visib = mask_visib[:, :, 2]
        if mask_visib is None:
            raise IOError(f"无法读取：{mask_visib}")
        return mask_visib

    @staticmethod
    def get_2d_coord_np(width, height, low=0, high=1, fmt="CHW"):
        """
        Args:
            x 水平向右, y 垂直向下
            width:
            height:
        Returns:
            xy: (2, height, width)

        """
        # coords values are in [low, high]  [0,1] or [-1,1]
        x = np.linspace(0, width - 1, width, dtype=np.float32)
        y = np.linspace(0, height - 1, height, dtype=np.float32)
        xy = np.asarray(np.meshgrid(x, y)).astype(np.uint64)
        if fmt == "HWC":
            xy = xy.transpose(1, 2, 0)
        elif fmt == "CHW":
            pass
        else:
            raise ValueError(f"Unknown format: {fmt}")
        return xy

    @staticmethod
    def get_bbox(bbox, img_height=480, img_length=640):
        """Compute square image crop window."""
        y1, x1, y2, x2 = bbox
        window_size = (max(y2 - y1, x2 - x1) // 40 + 1) * 40
        window_size = min(window_size, img_height - 40, img_length - 40)
        center = [(y1 + y2) // 2, (x1 + x2) // 2]
        rmin = center[0] - int(window_size / 2)
        rmax = center[0] + int(window_size / 2)
        cmin = center[1] - int(window_size / 2)
        cmax = center[1] + int(window_size / 2)
        if rmin < 0:
            delt = -rmin
            rmin = 0
            rmax += delt
        if cmin < 0:
            delt = -cmin
            cmin = 0
            cmax += delt
        if rmax > img_height:
            delt = rmax - img_height
            rmax = img_height
            rmin -= delt
        if cmax > img_length:
            delt = cmax - img_length
            cmax = img_length
            cmin -= delt
        return rmin, rmax, cmin, cmax

    @staticmethod
    def aug_bbox_DZI(hyper_params, bbox_xyxy, im_H, im_W):
        """使用边界框数据增强(Data Zone Interpolation), 扩充后的方框是一个正方形（可能经过放大）
        DZI_TYPE:
            uniform: 均匀分布增强，默认
                缩放：边界框边长按比例随机缩放，
                    缩放因子 scale_ratio 服从 [1-ratio, 1+ratio] 的均匀分布
                    （例如 ratio=0.25 时，范围为 [0.75, 1.25]）
                平移：边界框中心按比例随机平移，
                    平移量 shift_ratio 服从 [-ratio, ratio] 的均匀分布
                    （例如 ratio=0.25 时，最大平移量为边界框宽 / 高的 25%）
                适用于需要随机改变目标大小和位置的场景，增加模型对不同尺度和位置的鲁棒性
            roi10d: 截断正态分布增强
                角点扰动：直接对边界框的四个角点 (x1, y1, x2, y2) 进行随机扰动，
                    每个角点的偏移量服从 [-0.15*宽/高, 0.15*宽/高] 的均匀分布
                边界约束：确保扰动后的边界框不超出原始图像范围
                常用于医学图像（如 ROI10D 数据集），通过微调边界框角点提高对小目标的定位精度

        Args:
            bbox_xyxy (np.ndarray):
        Returns:
            :center: 增强后边界框的中心点坐标 [cx, cy]
            :scale: 增强后边界框的边长（正方形），确保不超过原始图像的最大尺寸
        """
        x1, y1, x2, y2 = bbox_xyxy.copy()
        # 裁剪框中心点
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        # 裁剪框高、宽
        bh = y2 - y1
        bw = x2 - x1
        if hyper_params["DZI_TYPE"].lower() == "uniform":
            scale_ratio = 1 + hyper_params["DZI_SCALE_RATIO"] * (
                2 * np.random.random_sample() - 1
            )  # [1-0.25, 1+0.25]
            shift_ratio = hyper_params["DZI_SHIFT_RATIO"] * (
                2 * np.random.random_sample(2) - 1
            )  # [-0.25, 0.25]
            bbox_center = np.array(
                [cx + bw * shift_ratio[0], cy + bh * shift_ratio[1]]
            )  # (h/2, w/2)
            # scale_ratio * hyper_params["DZI_PAD_SCALE"], [1.125, 1.875)
            scale = max(y2 - y1, x2 - x1) * scale_ratio * hyper_params["DZI_PAD_SCALE"]
        elif hyper_params["DZI_TYPE"].lower() == "roi10d":  # ROI 扰动增强
            # shift (x1,y1), (x2,y2) by 15% in each direction
            _a = -0.15
            _b = 0.15
            x1 += bw * (np.random.rand() * (_b - _a) + _a)
            x2 += bw * (np.random.rand() * (_b - _a) + _a)
            y1 += bh * (np.random.rand() * (_b - _a) + _a)
            y2 += bh * (np.random.rand() * (_b - _a) + _a)
            x1 = min(max(x1, 0), im_W)
            x2 = min(max(x1, 0), im_W)
            y1 = min(max(y1, 0), im_H)
            y2 = min(max(y2, 0), im_H)
            bbox_center = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)])
            scale = max(y2 - y1, x2 - x1) * hyper_params["DZI_PAD_SCALE"]
        else:
            assert hyper_params["DZI_TYPE"].lower() == "none"
            bbox_center = np.array([cx, cy])  # (w/2, h/2)
            scale = max(y2 - y1, x2 - x1)
        scale = min(scale, max(im_H, im_W)) * 1.0
        return bbox_center, scale

    @staticmethod
    def crop_resize_by_warp_affine(
        img, center, scale, output_size, rot=0, interpolation=cv2.INTER_LINEAR
    ) -> np.ndarray:
        """
        output_size: int or (w, h)
        NOTE: if img is (h,w,1), the output will be (h,w)
        """
        if isinstance(scale, (int, float)):
            scale = (scale, scale)
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        trans = _get_affine_transform(center, scale, rot, output_size)

        dst_img = cv2.warpAffine(
            img, trans, (int(output_size[0]), int(output_size[1])), flags=interpolation
        )
        if len(dst_img.shape) == 2:
            return dst_img
        return dst_img.astype(np.uint8)

    @staticmethod
    def rgb_transform(rgb):
        """归一化图像"""
        rgb_ = np.transpose(rgb, (2, 0, 1)) / 255
        _mean = (0.485, 0.456, 0.406)
        _std = (0.229, 0.224, 0.225)
        for i in range(3):
            rgb_[i, :, :] = (rgb_[i, :, :] - _mean[i]) / _std[i]
        return rgb_

    @staticmethod
    def defor_2D(roi_mask, rand_r=2, rand_pro=0.3):
        """
        add noise to mask
        :param roi_mask: 256 x 256
        :param rand_r: randomly expand or shrink the mask iter rand_r
        :return:
        """
        roi_mask = roi_mask.copy().squeeze()
        if np.random.rand() > rand_pro:
            return roi_mask
        mask = roi_mask.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        mask_erode = cv2.erode(mask, kernel, rand_r)  # rand_r
        mask_dilate = cv2.dilate(mask, kernel, rand_r)
        change_list = roi_mask[mask_erode != mask_dilate]
        l_list = change_list.size
        if l_list < 1.0:
            return roi_mask
        choose = np.random.choice(l_list, l_list // 2, replace=False)
        change_list = np.ones_like(change_list)
        change_list[choose] = 0.0
        roi_mask[mask_erode != mask_dilate] = change_list
        roi_mask[roi_mask > 0.0] = 1.0
        return roi_mask

    @staticmethod
    def create_point_cloud_with_xy(rgb, depth, K):
        """从RGB和深度图重建点云，并保留主视图有效像素的(x, y)索引"""
        h, w = depth.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        Z = depth
        X = (x - cx) * Z / fx
        Y = (y - cy) * Z / fy
        mask = Z > 0  # 二维掩码，(h, w)，True表示有效像素
        points = np.stack([X[mask], Y[mask], Z[mask]], axis=1)  # 有效3D点
        colors = rgb[mask]  # 有效像素的颜色

        # 提取有效像素的原始(u, v)索引（展平为列表）
        valid_x = x[mask].flatten()  # 主视图有效u坐标
        valid_y = y[mask].flatten()  # 主视图有效v坐标

        return points, colors, mask, valid_x, valid_y

    @staticmethod
    def create_view(rgb, depth, K, xymap, rotation_angle=45, rotation_axis=0):
        """
        创建其他视图
        Args:
            rgb: 未归一化,(W,H,3)
            depth: 未归一化,(W,H)
            K: 相机内参
            rotation_angle: 旋转角度
            rotation_axis: 旋转轴 0,1,2 X,Y,Z
        Returns:
            :img: 新视角图像
            :xymap : np.ndarray: 索引
        """
        # 1. 重建点云并获取主视图有效(u, v)
        points, colors, valid_mask, valid_x, valid_y = (
            DataUtils.create_point_cloud_with_xy(rgb, depth, K)
        )

    @staticmethod
    def get_oriented_bbox(mask):
        """计算旋转物体的最小外接矩形"""
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        # 找到最大轮廓
        max_contour = max(contours, key=cv2.contourArea)

        # 计算最小外接旋转矩形
        rect = cv2.minAreaRect(max_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # 计算旋转矩形的AABB
        x_min = min(box[:, 0])
        x_max = max(box[:, 0])
        y_min = min(box[:, 1])
        y_max = max(box[:, 1])

        return (y_min, y_max, x_min, x_max)

    @staticmethod
    def cut_roi(
        rgb: np.ndarray, depth: np.ndarray, mask: np.ndarray, K: np.ndarray, cfg
    ):
        im_H, im_W = rgb.shape[0], rgb.shape[1]
        # output_size = 224
        output_size = 256
        coord_2d = DataUtils.get_2d_coord_np(im_W, im_H, fmt="HWC")
        target_mask_id = np.unique(mask)[1]
        object_bool_mask = np.equal(mask, target_mask_id)
        ys, xs = np.argwhere(object_bool_mask).transpose(1, 0)
        rmin, rmax, cmin, cmax = (
            np.min(ys),
            np.max(ys),
            np.min(xs),
            np.max(xs),
        )
        rmin, rmax, cmin, cmax = DataUtils.get_bbox(
            [rmin, cmin, rmax, cmax], im_H, im_W
        )
        bbox_xyxy = np.array([cmin, rmin, cmax, rmax])  # 方框的左上和右下两个角
        bbox_center, scale = DataUtils.aug_bbox_DZI(
            cfg.DYNAMIC_ZOOM_IN_PARAMS, bbox_xyxy, im_H, im_W
        )

        roi_coord_2d = DataUtils.crop_resize_by_warp_affine(
            coord_2d, bbox_center, scale, output_size, interpolation=cv2.INTER_NEAREST
        ).transpose(2, 0, 1)
        roi_rgb = DataUtils.crop_resize_by_warp_affine(
            rgb, bbox_center, scale, output_size, interpolation=cv2.INTER_LINEAR
        )
        mask_target = mask.copy().astype(np.float32)
        mask_target[mask == target_mask_id] = 1.0
        roi_mask = DataUtils.crop_resize_by_warp_affine(
            mask_target,
            bbox_center,
            scale,
            output_size,
            interpolation=cv2.INTER_NEAREST,  # 掩码建议用最近邻插值
        )
        roi_depth = DataUtils.crop_resize_by_warp_affine(
            depth, bbox_center, scale, output_size, interpolation=cv2.INTER_LINEAR
        )
        # # 4. 调整相机内参 K
        roi_K = K.copy()

        s = output_size / scale
        # # 4.1 调整焦距（缩放影响）
        roi_K[0, 0] = K[0, 0] * s  # fx' = fx * 缩放因子
        roi_K[1, 1] = K[1, 1] * s  # fy' = fy * 缩放因子

        # 4.2 调整主点（cx, cy）：通过仿射变换映射原始主点到新图像
        original_cx, original_cy = K[0, 2], K[1, 2]  # 原始主点坐标
        # 获取仿射变换矩阵（已包含裁剪和缩放）
        trans = _get_affine_transform(
            center=bbox_center,
            scale=(scale, scale),
            rot=0,
            output_size=(output_size, output_size),
        )
        # # 仿射变换公式：u' = M[0,0]*u + M[0,1]*v + M[0,2]
        new_cx = trans[0, 0] * original_cx + trans[0, 1] * original_cy + trans[0, 2]
        new_cy = trans[1, 0] * original_cx + trans[1, 1] * original_cy + trans[1, 2]
        # # 4. 边界保护：确保主点在合理范围内
        roi_K[0, 2] = new_cx
        roi_K[1, 2] = new_cy

        return roi_rgb, roi_depth, roi_mask, roi_K, roi_coord_2d, bbox_center

    @staticmethod
    def roi_filter(rgb, depth, mask):
        target_label = np.unique(mask)[1]
        bool_mask = mask == target_label
        filtered_rgb = rgb.copy()
        filtered_depth = depth.copy()
        filtered_rgb[~bool_mask] = 0
        filtered_depth[~bool_mask] = 0
        return filtered_rgb, filtered_depth.astype(np.float32)

    @staticmethod
    def _backward_to_forward_map(backward_map, h, w):
        """将反向映射转为正向映射，并处理亚像素位置"""
        forward_map = np.full((h, w, 2), -1, dtype=int)  # 整数坐标

        # 四舍五入获取最近的整数坐标
        int_coords = np.round(backward_map).astype(int)

        # 检查有效性并填充正向映射
        valid_mask = (backward_map[..., 0] >= 0) & (backward_map[..., 1] >= 0)
        for i in range(h):
            for j in range(w):
                if valid_mask[i, j]:
                    x, y = int_coords[i, j]
                    if 0 <= x < w and 0 <= y < h:
                        forward_map[y, x] = [j, i]  # 注意坐标顺序

        return forward_map

    @staticmethod
    def transformer_image_view(
        filter_rgb: np.ndarray,
        valid_coords: np.ndarray,  # (N,3)[y,x]
        pcd: np.ndarray,
        roi_K: np.ndarray,
        rotate_angle: float,  # 角度制
        axis: str,  # 旋转轴 ('x'或'y')
    ):
        from scipy.spatial.transform import Rotation

        """
        生成旋转视图（双线性插值 + 深度测试）

        参数:
            filter_rgb: 物体RGB图像 (H, W, 3) uint8
            pcd: 物体点云 (H, W, 3) 无效点为0
            roi_K: 相机内参 (3, 3)
            rotate_angle: 旋转角度（度）
            axis: 旋转轴 ('x'或'y')

        返回:
            rotated_view: 旋转视图 (H, W, 3) uint8
            backward_map: 反向映射 (H, W, 2) 存储旋转视图坐标
        """
        # 角度转弧度
        angle_rad = np.radians(rotate_angle)
        h, w = filter_rgb.shape[:2]
        fx, fy = roi_K[0, 0], roi_K[1, 1]
        cx, cy = roi_K[0, 2], roi_K[1, 2]

        # 提取有效点
        points = pcd
        # colors = filter_rgb[valid_coords[:, 0], valid_coords[:, 1]].astype(np.float32)
        colors = filter_rgb[valid_coords[:, 0], valid_coords[:, 1]]

        # 创建旋转矩阵
        if axis.upper() == "X":
            R = Rotation.from_euler("x", angle_rad).as_matrix()
        else:  # 默认Y轴
            R = Rotation.from_euler("y", angle_rad).as_matrix()

        # 旋转点云
        centroid = np.mean(points, axis=0)
        rotated_points = (points - centroid) @ R.T + centroid

        # 初始化缓冲区
        rotated_view = np.zeros((h, w, 3), dtype=np.uint8)
        backward_map = np.full((h, w, 2), -1.0)  # 存储旋转视图坐标

        # 投影计算
        x_proj = fx * rotated_points[:, 0] / rotated_points[:, 2] + cx
        y_proj = fy * rotated_points[:, 1] / rotated_points[:, 2] + cy
        depths = rotated_points[:, 2]

        # 双线性插值
        x0 = np.floor(x_proj).astype(int)
        y0 = np.floor(y_proj).astype(int)
        x1 = x0 + 1
        y1 = y0 + 1

        dx = x_proj - x0
        dy = y_proj - y0

        w00 = (1 - dx) * (1 - dy)
        w01 = dx * (1 - dy)
        w10 = (1 - dx) * dy
        w11 = dx * dy

        # 创建颜色和权重缓冲区
        color_buffer = np.zeros((h, w, 3), dtype=np.float32)
        weight_buffer = np.zeros((h, w), dtype=np.float32)

        # 更新四个角点的颜色和权重
        for (dx_offset, dy_offset), weight in [
            ((0, 0), w00),
            ((1, 0), w01),
            ((0, 1), w10),
            ((1, 1), w11),
        ]:
            px = np.clip(x0 + dx_offset, 0, w - 1).astype(int)
            py = np.clip(y0 + dy_offset, 0, h - 1).astype(int)

            # 更新颜色缓冲区
            for c in range(3):
                np.add.at(color_buffer[:, :, c], (py, px), colors[:, c] * weight)

            # 更新权重缓冲区
            np.add.at(weight_buffer, (py, px), weight)

            # 更新反向映射（深度测试）
            for i in range(len(px)):
                x, y = px[i], py[i]
                current_depth = depths[i]
                prev_depth = (
                    backward_map[valid_coords[i, 0], valid_coords[i, 1], 0]
                    if backward_map[valid_coords[i, 0], valid_coords[i, 1], 0] > 0
                    else np.inf
                )

                # 只保留最近的映射点
                if current_depth < prev_depth:
                    backward_map[valid_coords[i, 0], valid_coords[i, 1]] = [
                        x_proj[i],
                        y_proj[i],
                    ]

        # 合成图像
        valid_weights = weight_buffer > 0
        rotated_view[valid_weights] = (
            color_buffer[valid_weights] / weight_buffer[valid_weights, None]
        ).astype(np.uint8)

        # 空洞填充 - 使用快速向量化方法
        mask = ~valid_weights
        if np.any(mask):
            # 获取最近的有效点（使用距离变换）
            dist = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)

            # 获取最近邻索引
            max_val = np.max(dist)
            dist_normalized = (dist / max_val * 255).astype(np.uint8)
            _, labels = cv2.connectedComponents(dist_normalized)

            # 填充空洞
            for label in range(1, np.max(labels) + 1):
                region_mask = labels == label
                nearest_point = np.argwhere(region_mask & valid_weights)
                if len(nearest_point) > 0:
                    nearest_color = rotated_view[
                        nearest_point[0, 0], nearest_point[0, 1]
                    ]
                    rotated_view[region_mask] = nearest_color
        forward_map = DataUtils._backward_to_forward_map(backward_map, h, w)
        return rotated_view, forward_map

    @staticmethod
    def convert_valid_coords_to_dense_map(
        valid_coords, height=224, width=224, invalid_value=-1
    ):
        """
        将有效坐标列表转换为稠密坐标映射

        参数:
            valid_coords: 有效坐标数组，形状为 [N, 2]，每个元素为 [y, x]
            height, width: 输出映射的高度和宽度
            invalid_value: 无效位置填充的值

        返回:
            coords_map: 稠密坐标映射，形状为 [height, width, 2]
            mask: 有效位置掩码，形状为 [height, width]
        """
        # 创建全量坐标映射，初始值为无效值
        coords_map = np.full((height, width, 2), invalid_value, dtype=np.float32)

        # 创建掩码，初始值为False（无效）
        mask = np.zeros((height, width), dtype=bool)

        # 过滤超出边界的坐标
        valid_coords = valid_coords[
            (valid_coords[:, 0] >= 0)
            & (valid_coords[:, 0] < height)
            & (valid_coords[:, 1] >= 0)
            & (valid_coords[:, 1] < width)
        ]

        # 将有效坐标分散到映射中
        y_indices = valid_coords[:, 0].astype(int)
        x_indices = valid_coords[:, 1].astype(int)

        # 设置坐标值（注意这里存储的是[x,y]，与输入的[y,x]顺序不同，根据需要调整）
        coords_map[y_indices, x_indices, 0] = valid_coords[:, 1]  # x坐标
        coords_map[y_indices, x_indices, 1] = valid_coords[:, 0]  # y坐标

        # 设置有效掩码
        mask[y_indices, x_indices] = True

        return coords_map.astype(np.int64), mask
