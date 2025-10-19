import open3d as o3d
import numpy as np
import torch
from typing import Union, Type
from scipy.spatial import cKDTree  # 用于快速查找邻域点
import cv2 as cv
from copy import deepcopy


class MultiViewTable:
    __slots__ = ("rgb", "forward_map", "reverse_map", "valid_mask")

    def __init__(self, output_size, orig_size):
        h, w = output_size
        orig_h, orig_w = orig_size
        self.rgb = np.zeros((h, w, 3), dtype=np.uint8)
        self.forward_map = np.full(
            (h, w, 4), -1, dtype=np.float32
        )  # (x_rot, y_rot, x_orig, y_orig)
        self.reverse_map = np.full(
            (orig_h, orig_w, 2), -1, dtype=np.float32
        )  # (x_rot, y_rot)
        self.valid_mask = np.zeros((h, w), dtype=bool)


class PcdO3dUtils:

    @staticmethod
    def o3d_create_pcd(depth, K):
        """open32 类型点云"""
        H, W = depth.shape
        # 提取内参参数
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # 生成像素坐标网格 (u, v)
        u = np.arange(W)  # 列坐标（宽度方向）
        v = np.arange(H)  # 行坐标（高度方向）
        u, v = np.meshgrid(u, v)  # 生成 (H, W) 的网格坐标

        # 过滤无效深度值（深度为0或负数的点）
        valid_mask = depth > 0
        u = u[valid_mask]
        v = v[valid_mask]
        z = depth[valid_mask]  # 有效深度值（z坐标）

        # 计算三维坐标 (x, y, z)
        # 公式：x = (u - cx) * z / fx；y = (v - cy) * z / fy；z = z
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        # 构造点云数据（形状为 (N, 3)，N为有效点数量）
        points = np.column_stack([x, y, z])
        pcd = o3d.geometry.PointCloud()

        pcd.points = o3d.utility.Vector3dVector(points)
        R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        pcd.rotate(R)
        return pcd

    @staticmethod
    def rotate_with_axes(pcd, roi_K, angle_rad, axis="X"):
        """
        绕指定轴旋转点云（X轴=俯视/仰视，Y轴=左视/右视）
        返回：旋转后点云，主视图点索引（与旋转点一一对应）
        """
        cx, cy = roi_K[0, 2], roi_K[1, 2]
        points = np.asarray(pcd.points)
        if len(points) == 0:
            return pcd, np.array([])

        # 图像中心3D坐标（用点云平均深度）
        mean_z = np.mean(points[:, 2])
        fx, fy = roi_K[0, 0], roi_K[1, 1]
        center_3d = np.array([0, 0, mean_z])  # 主点在光轴上，x=0,y=0

        # 平移到原点旋转
        pcd_trans = deepcopy(pcd)
        pcd_trans.translate(-center_3d)

        # 旋转矩阵（X轴=俯仰，Y轴=偏航）
        if axis == "X":
            R = pcd.get_rotation_matrix_from_xyz((angle_rad, 0, 0))
        elif axis == "Y":
            R = pcd.get_rotation_matrix_from_xyz((0, angle_rad, 0))
        else:
            raise ValueError("axis must be 'X' (俯视) or 'Y' (左视)")

        pcd_rot = deepcopy(pcd_trans)
        pcd_rot.rotate(R, center=(0, 0, 0))
        pcd_rot.translate(center_3d)  # 平移回原位置

        # 旋转点与主视图点索引一一对应（无增减）
        main_indices = np.arange(len(pcd.points))  # 主视图点的原始索引
        return pcd_rot, main_indices

    @staticmethod
    def project_with_xs_ys(rotated_pcd, roi_K, main_xs_ys, output_size):
        """投影函数"""
        h, w = output_size
        fx, fy = roi_K[0, 0], roi_K[1, 1]
        cx, cy = roi_K[0, 2], roi_K[1, 2]

        # 确定原始图像尺寸
        orig_h, orig_w = (
            int(main_xs_ys[:, 1].max()) + 1,
            int(main_xs_ys[:, 0].max()) + 1,
        )
        result = MultiViewTable(output_size, (orig_h, orig_w))

        points = np.asarray(rotated_pcd.points)
        colors = np.asarray(rotated_pcd.colors) * 255
        valid_mask = points[:, 2] > 0
        points = points[valid_mask]
        colors = colors[valid_mask]
        orig_indices = np.where(valid_mask)[0]

        if len(points) == 0:
            return result

        # 计算投影坐标 (向量化)
        x_proj = fx * points[:, 0] / points[:, 2] + cx
        y_proj = fy * points[:, 1] / points[:, 2] + cy

        # 有效区域掩码
        valid_x = (x_proj >= 0) & (x_proj < w - 1)
        valid_y = (y_proj >= 0) & (y_proj < h - 1)
        valid_pts = valid_x & valid_y

        # 提取有效点
        valid_x_proj = x_proj[valid_pts]
        valid_y_proj = y_proj[valid_pts]
        valid_colors = colors[valid_pts]
        valid_orig_indices = orig_indices[valid_pts]

        # 双线性插值坐标
        x0 = np.floor(valid_x_proj).astype(int)
        y0 = np.floor(valid_y_proj).astype(int)
        x1 = x0 + 1
        y1 = y0 + 1

        dx = valid_x_proj - x0
        dy = valid_y_proj - y0

        w00 = (1 - dx) * (1 - dy)
        w01 = dx * (1 - dy)
        w10 = (1 - dx) * dy
        w11 = dx * dy

        # 获取原始坐标
        orig_coords = main_xs_ys[valid_orig_indices]

        # 创建颜色和权重缓冲区
        color_buffer = np.zeros((h, w, 3), dtype=np.float32)
        weight_buffer = np.zeros((h, w), dtype=np.float32)

        # 更新四个角点的颜色和权重
        for (dx, dy), weight in [
            ((0, 0), w00),
            ((1, 0), w01),
            ((0, 1), w10),
            ((1, 1), w11),
        ]:
            px = np.clip(x0 + dx, 0, w - 1).astype(int)
            py = np.clip(y0 + dy, 0, h - 1).astype(int)

            # 更新颜色缓冲区
            for c in range(3):
                np.add.at(color_buffer[:, :, c], (py, px), valid_colors[:, c] * weight)

            # 更新权重缓冲区
            np.add.at(weight_buffer, (py, px), weight)

            # 更新前向映射（取最近点）
            for i in range(len(px)):
                x, y = px[i], py[i]
                if weight[i] > weight_buffer[y, x] * 0.9:  # 避免频繁更新
                    result.forward_map[y, x] = [
                        x,
                        y,
                        orig_coords[i, 0],
                        orig_coords[i, 1],
                    ]

        # 更新反向映射（原始视图到旋转视图）
        for i in range(len(valid_orig_indices)):
            orig_x, orig_y = int(orig_coords[i, 0]), int(orig_coords[i, 1])
            # 只保留深度最小的映射
            current_depth = points[valid_pts[i], 2]
            prev_depth = result.reverse_map[orig_y, orig_x, 0]  # 暂时借用存储深度
            if prev_depth < 0 or current_depth < prev_depth:
                result.reverse_map[orig_y, orig_x] = [valid_x_proj[i], valid_y_proj[i]]

        # 归一化颜色
        valid_weights = weight_buffer > 0
        result.rgb[valid_weights] = (
            color_buffer[valid_weights] / weight_buffer[valid_weights, None]
        ).astype(np.uint8)
        result.valid_mask = valid_weights

        return result

    @staticmethod
    def o3d_get_camera_coord(camera_K, img_size):
        fx, fy = camera_K[0, 0], camera_K[1, 1]
        cx, cy = camera_K[0, 2], camera_K[1, 2]
        roi_h, roi_w = img_size

        # 2. 创建相机内参对象（基于 roi_K）
        camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=roi_w, height=roi_h, fx=fx, fy=fy, cx=cx, cy=cy
        )
        camera_coord: o3d.geometry.TriangleMesh = (
            o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.1,  # 坐标轴长度，根据点云尺度调整
                origin=[0, 0, 0],  # 相机光心在世界坐标系中的位置（默认原点）
            )
        )
        return camera_coord, camera_intrinsic

    @staticmethod
    def create_camera_frustum(intrinsic, scale=0.5):
        """基于内参创建相机视锥体
        可视化相机视锥体（显示相机的视野范围，增强关联性）
        视锥体顶点：从相机光心出发，指向 ROI 图像的四个角点（在Z=0.5米处的投影）
        """
        w, h = intrinsic.width, intrinsic.height
        fx, fy = intrinsic.get_focal_length()
        cx, cy = intrinsic.get_principal_point()

        # 计算 ROI 图像四个角点在相机坐标系中的 3D 坐标（Z=scale 平面）
        frustum_points = [
            [0, 0, 0],  # 相机光心
            [(0 - cx) * scale / fx, (0 - cy) * scale / fy, scale],  # 左上角
            [(w - cx) * scale / fx, (0 - cy) * scale / fy, scale],  # 右上角
            [(w - cx) * scale / fx, (h - cy) * scale / fy, scale],  # 右下角
            [(0 - cx) * scale / fx, (h - cy) * scale / fy, scale],  # 左下角
        ]

        # 创建视锥体的线框
        lines = [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],  # 光心到四个角点
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1],  # 四个角点连接成面
        ]
        colors = [[1, 0, 0] for _ in range(len(lines))]  # 红色线框

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(frustum_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set


class PointCloudSample:
    """
    点云采样类 </br>
    注：需要上采样时，统一返回的是 point_cloud_num_check() 方法的值

    Args:
        point_cloud (np.ndarray): 输入点云数据，形状为(N, 3)
        sample_num (int): 采样点的数量
    Returns:
        PointCloudSample
    """

    def __init__(
        self, point_cloud: np.ndarray, sample_num: int, image_coords: np.ndarray
    ):
        self.point_cloud = point_cloud
        self.total_point_cloud_num = point_cloud.shape[0]
        self.sample_num = sample_num
        self.is_up_sample: bool = (
            True if self.total_point_cloud_num < self.sample_num else False
        )

        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(point_cloud)
        if image_coords.shape[0] != self.total_point_cloud_num:
            raise ValueError(
                f"点云数量（{self.total_point_cloud_num}）与图像坐标数量（{image_coords.shape[0]}）不匹配"
            )
        self.image_coords = image_coords
        self.pcd_idx = self.image_coords[:, 0] * 224 + self.image_coords[:, 1]

    def point_cloud_up_sample(self) -> tuple[np.ndarray, np.ndarray]:
        """
        确保点云数量满足采样需求，若原始点云数量不足则通过重复填充补充
        当需要的采样点数（sample_num）大于原始点云总点数（total_point_cloud_num）时，
        会将原始点云重复若干次并截取部分点，以满足采样数量要求，同时返回对应的点索引；
        若原始点云数量足够（大于等于采样数），则返回None，表示无需填充。
        Returns:
            tuple[np.ndarray, np.ndarray] | None: 返回点云索引和采样点云数据
            - ids (np.ndarray): 点云索引
            - point_cloud (np.ndarray): 采样后的点云数据
            如果原始点云数量足够，则直接返回None
        """
        total_pts_num = self.total_point_cloud_num
        n_pts = self.sample_num
        pcl = np.concatenate(
            [
                # 重复复制 n 个完整原始点云
                # 将原始点云pcl沿行方向复制(n_pts // total_pts_num)次
                np.tile(pcl, (n_pts // total_pts_num, 1)),
                pcl[: n_pts % total_pts_num],  # 不足一个完整的位置补零
            ],
            axis=0,
        )
        ids = np.concatenate(
            [
                np.tile(np.arange(total_pts_num), n_pts // total_pts_num),
                np.arange(n_pts % total_pts_num),
            ],
            axis=0,
        )

        return ids, pcl

    def random_sample(self) -> tuple[np.ndarray, np.ndarray]:
        if self.is_up_sample:
            return self.point_cloud_up_sample()
        sample_ids = np.random.permutation(self.total_point_cloud_num)[
            : self.sample_num
        ]
        sampled_point_cloud = self.point_cloud[sample_ids]
        # sampled_pcd_idx = self.pcd_idx[sample_ids]  # 采样对应的图像索引
        return sample_ids, sampled_point_cloud

    def farthest_point_sample(self) -> tuple[np.ndarray, np.ndarray]:
        """
        最远点采样
        Returns:
            point_cloud_sampled (np.ndarray): 采样后的点云数据，形状为(M, 3)
        """
        if self.is_up_sample:
            return self.point_cloud_up_sample()
        ids = np.zeros(self.sample_num, dtype=int)
        # 记录每个点到已选点集的最小距离
        distances = np.ones(self.total_point_cloud_num, dtype=np.float32) * np.inf
        # 随机的起始点
        start_idx = np.random.randint(self.total_point_cloud_num)
        ids[0] = start_idx  # 记录起始点至索引
        farthest_point = points[start_idx]
        # 迭代选择最远点,从第二个点开始计算
        for i in range(1, self.sample_num):
            # 计算当前所有点到最新选中点的距离（平方，避免开方加速计算）
            dist = np.sum((points - farthest_point) ** 2, axis=1)
            # 更新每个点到已选点集的最小距离
            distances = np.minimum(distances, dist)
            # 选择距离最大的点作为下一个采样点
            farthest_idx = np.argmax(distances)
            ids[i] = farthest_idx
            farthest_point = points[farthest_idx]

        point_cloud = self.point_cloud[ids]
        return ids, point_cloud

    def geometric_sampling(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        几何采样
        Args:
            k: 邻域点数量,根据点云密度调整,建议5-32
        """
        if self.is_up_sample:
            return self.point_cloud_up_sample()
        # 1. 构建KDTree快速查找邻域
        tree = cKDTree(self.point_cloud)
        # 存储每个点的曲率
        curvatures = np.zeros(self.total_point_cloud_num, dtype=np.float32)
        for i in range(self.total_point_cloud_num):
            # 查找当前点的k个最近邻（包含自身，后续排除）
            # 多查1个排除自身
            distances, indices = tree.query(self.point_cloud[i], k=k + 1)
            neighbor_indices = indices[1:]  # 排除自身
            neighbors = self.point_cloud[neighbor_indices]  # 邻域点，形状 (k, 3)

            # 2. 计算邻域点的协方差矩阵（反映几何分布）
            centroid = np.mean(neighbors, axis=0)  # 邻域中心点
            # 协方差矩阵 (3,3)
            covariance = np.cov(neighbors - centroid, rowvar=False)

            # 3. 从协方差矩阵的特征值计算曲率
            # 特征值越小，说明该方向的几何变化越平缓；特征值差异越大，曲率越大
            eigenvalues = np.linalg.eigvalsh(covariance)  # 计算特征值（从小到大排序）
            eigenvalues = np.clip(eigenvalues, 1e-10, None)  # 避免除以0

            # 曲率定义：最小特征值 / 特征值之和（值越大，曲率越大）
            curvature = eigenvalues[0] / (np.sum(eigenvalues) + 1e-10)
            curvatures[i] = curvature

        # 第三步：基于曲率采样（优先选择曲率大的点）
        # 按曲率降序排序，取前sample_num个点的索引
        sorted_indices = np.argsort(-curvatures)  # 降序排列的索引
        ids = sorted_indices[: self.sample_num]  # 选择前sample_num个点
        # 采样后的点云
        points = self.point_cloud[ids]
        return ids, points

    def adaptiveVoxelBlockHybridSampling(self):
        """自适应体素块混合采样"""
        # TODO: 未完成
        pass


"""-------------------- Interface --------------------"""


class PointCloudUtils:
    @staticmethod
    def depth_to_pcl(depth, K, xymap, valid):
        K = K.reshape(-1)
        cx, cy, fx, fy = K[2], K[5], K[0], K[4]
        depth = depth.reshape(-1).astype(np.float32)  # 展平为一维数组
        depth = depth[valid]  # 把 true 的点保留
        # depth = depth.reshape(-1).astype(np.float32)[valid]
        x_map = xymap[0].reshape(-1)[valid]  # 索引也仅保留有效点
        y_map = xymap[1].reshape(-1)[valid]
        real_x = (x_map - cx) * depth / fx
        real_y = (y_map - cy) * depth / fy
        pcl = np.stack((real_x, real_y, depth), axis=-1)  # (有效点云数,3)
        return pcl.astype(np.float32)

    @staticmethod
    def load_point_cloud(
        file_path: str,
        return_type: Type[
            Union[o3d.geometry.PointCloud, np.ndarray, torch.Tensor]
        ] = o3d.geometry.PointCloud,
    ) -> Union[o3d.geometry.PointCloud, np.ndarray, torch.Tensor]:
        """
        从文件加载点云，并转换为指定类型

        Args:
            file_path (str): Path to the point cloud file.

        Returns:
            o3d.geometry.PointCloud: 加载的点云
        """
        pcd_o3d: o3d.geometry.PointCloud = o3d.io.read_point_cloud(file_path)
        if return_type is o3d.geometry.PointCloud:
            return pcd_o3d
        elif return_type is np.ndarray:
            # 转换为numpy数组 (N, 3)
            return np.asarray(pcd_o3d.points)
        elif return_type is torch.Tensor:
            # 转换为torch张量 (N, 3)，默认在CPU上
            return torch.from_numpy(np.asarray(pcd_o3d.points))
        else:
            # 不支持的类型，抛出异常
            raise ValueError(
                f"不支持的返回类型: {return_type}。"
                f"支持的类型: o3d.geometry.PointCloud, np.ndarray, torch.Tensor"
            )

    @staticmethod
    def visualize_point_cloud(
        pcd_input: Union[torch.Tensor, np.ndarray, o3d.geometry.PointCloud],
    ) -> None:
        """
        可视化点云

        Args:
            pcd_input (torch.Tensor | numpy.ndarray | o3d.geometry.PointCloud):
                输入的点云数据。支持的格式包括：
                - torch.Tensor: 形状为(N, 3)或(3, N)，数据类型为float32或float64
                - numpy.ndarray: 形状为(N, 3)或(3, N)，数据类型为float32或float64
                - o3d.geometry.PointCloud: Open3D原生点云类型，可直接可视化
        """
        # 将输入转换为Open3D的PointCloud类型
        if isinstance(pcd_input, o3d.geometry.PointCloud):
            # 若已是PointCloud类型，直接使用
            pcd = pcd_input
        elif isinstance(pcd_input, torch.Tensor):
            # 处理PyTorch张量类型
            # 若张量在GPU上，先移至CPU并转为numpy数组；若已在CPU，直接转换
            if pcd_input.device.type != "cpu":
                pcd_np = pcd_input.cpu().detach().numpy()
            else:
                pcd_np = pcd_input.numpy()

            # 处理点云形状：支持(3, N)格式，转置为(N, 3)（Open3D要求的格式）
            # 判断条件：第一维为3且第二维长度大于3（避免误判3x3等特殊形状）
            if pcd_np.shape[0] == 3 and pcd_np.shape[1] > 3:
                pcd_np = pcd_np.T

            # 创建PointCloud对象并设置点数据
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd_np)
        elif isinstance(pcd_input, np.ndarray):
            # 处理numpy数组类型，形状转换逻辑与张量一致
            if pcd_input.shape[0] == 3 and pcd_input.shape[1] > 3:
                pcd_input = pcd_input.T

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd_input)
        else:
            raise TypeError(
                f"不支持的输入类型: {type(pcd_input)}。"
                f"支持的类型: torch.Tensor, numpy.ndarray, o3d.geometry.PointCloud"
            )

        o3d.visualization.draw_geometries([pcd])  # Open3D 可视化接口

    @staticmethod
    def random_sample_points(pcl, n_pts):
        """Down sample the point cloud.
        TODO: use farthest point sampling

        Args:
            pcl (torch tensor or numpy array):  NumPoints x 3
            num (int): target point number
        """
        total_pts_num = pcl.shape[0]
        if total_pts_num < n_pts:
            pcl = np.concatenate(
                [
                    # 重复复制 n 个完整原始点云
                    # 将原始点云pcl沿行方向复制(n_pts // total_pts_num)次
                    np.tile(pcl, (n_pts // total_pts_num, 1)),
                    pcl[: n_pts % total_pts_num],  # 不足一个完整的位置补零
                ],
                axis=0,
            )
            ids = np.concatenate(
                [
                    np.tile(np.arange(total_pts_num), n_pts // total_pts_num),
                    np.arange(n_pts % total_pts_num),
                ],
                axis=0,
            )
        else:
            # 从[0,total_pts_num)中随机无重复地抽取 total_pts_num 个索引
            ids = np.random.permutation(total_pts_num)
            ids = ids[:n_pts]  # 保留前面抽取的 n_pts个作为采样得到的点索引
            pcl = pcl[ids]
        return ids, pcl

    @staticmethod
    def sample_point_cloud(
        point_cloud: np.ndarray,
        sample_num: int = 1000,
        sample_method: str = "random",
    ) -> np.ndarray:
        """
        点云采样
        Args:
            point_cloud (np.ndarray): 输入点云数据，形状为(N, 3)。
            sample_num (int): 采样点的数量。
            sample_method (str): 采样方法，支持
                ["random" , "farthest","geometric","adaptive"]
                即[随机、最远点、几何(基于曲率)、自适应体素块混合]
        Returns:
            point_cloud_sampled (np.ndarray): 采样后的点云数据，形状为(M, 3)
        """
        points_sample = PointCloudSample(point_cloud, sample_num)
        if sample_method == "random":
            return points_sample.random_sample()

    @staticmethod
    def transform_depth2pcl(
        depth: np.ndarray,
        camera_K: np.ndarray,
        xymap: np.ndarray,
        valid: np.ndarray,
    ):
        """深度图与相机内参转点云"""
        fx, fy = camera_K[0, 0], camera_K[1, 1]
        cx, cy = camera_K[0, 2], camera_K[1, 2]
        depth = depth.astype(np.float32)[valid]
        x_map = xymap[0][valid]
        y_map = xymap[1][valid]
        real_x = (x_map - cx) * depth / fx
        real_y = (y_map - cy) * depth / fy
        pcl = np.stack((real_x, real_y, depth), axis=-1)
        return pcl.astype(np.float32)

    @staticmethod
    def transform_depth2pcd(
        depth: np.ndarray,
        camera_K: np.ndarray,
    ):
        """仅使用深度图和相机内参转换点云（自动过滤无效点）

        Args:
            depth: 深度图，形状为(H, W)，无效点值为0
            camera_K: 相机内参矩阵，形状为(3, 3)
            depth_unit: 深度图单位，'m'或'mm'

        Returns:
            pcl: 点云，形状为(N, 3)，其中N为有效点数量
        """
        # 提取相机内参
        fx, fy = camera_K[0, 0], camera_K[1, 1]
        cx, cy = camera_K[0, 2], camera_K[1, 2]

        # 创建像素坐标网格
        height, width = depth.shape
        u, v = np.meshgrid(np.arange(width), np.arange(height))

        # 过滤无效点（深度为0的点）
        valid_mask = depth > 0
        depth_valid = depth[valid_mask]
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]

        # 计算3D坐标
        x = (u_valid - cx) * depth_valid / fx
        y = (v_valid - cy) * depth_valid / fy

        # 构建点云
        pcl = np.stack((x, y, depth_valid), axis=-1)
        return pcl.astype(np.float32)

    @staticmethod
    def pcl_with_rgb(
        pcl_in: np.ndarray,  # 由transform_depth2pcl生成的点云，形状(N, 3)
        roi_rgb_: np.ndarray,  # ROI区域的原始RGB图像，形状(H, W, 3)
        xymap: list[
            np.ndarray
        ],  # 即roi_coord_2d，形状(2, H, W)，存储每个像素的(x,y)坐标
        valid: np.ndarray,  # 有效点的掩码，形状(H*W,)，与transform_depth2pcl中一致
    ):
        """
        给点云添加RGB信息，返回带RGB的点云
        输出：形状(N, 6)，每个点格式为(x, y, z, r, g, b)
        """
        # 1. 获取点云每个点在ROI图像中的像素坐标（u, v）
        # xymap[0]是所有像素的x坐标（u），xymap[1]是y坐标（v），通过valid筛选有效点
        x_map = xymap[0][valid]  # 有效点的u坐标（x方向），形状(N,)
        y_map = xymap[1][valid]  # 有效点的v坐标（y方向），形状(N,)

        # 2. 确保坐标在ROI图像范围内（避免越界）
        H, W = roi_rgb_.shape[:2]  # ROI图像的高和宽
        # 强制坐标在[0, H-1]和[0, W-1]范围内（防止索引越界）
        u = np.clip(x_map.astype(int), 0, W - 1)
        v = np.clip(y_map.astype(int), 0, H - 1)

        # 3. 从ROI图像中提取对应像素的RGB值
        # roi_rgb_形状是(H, W, 3)，v是行索引，u是列索引
        rgb_values = roi_rgb_[v, u]  # 形状(N, 3)，每个点的(r, g, b)

        # 4. 将点云与RGB值拼接
        pcl_with_rgb = np.concatenate([pcl_in, rgb_values], axis=1)  # 形状(N, 6)

        return pcl_with_rgb.astype(np.float32)

    @staticmethod
    def generate_virtual_views_by_pcl(
        pcl_with_rgb: np.ndarray,  # 带RGB的点云，形状(N, 6)
        camera_K: np.ndarray,  # 相机内参矩阵
        view_angles: list = [
            (0, 0),
            (30, 0),
            (-30, 0),
            (0, 30),
            (0, -30),
        ],  # 视角角度列表 [(rx1, ry1), (rx2, ry2), ...]
        img_size=(224, 224),  # 输出图像尺寸
        background_color: tuple = (255, 255, 255),  # 背景颜色
    ):
        """
        基于带 RGB 信息的点云，生成不同视角的 RGB 图像
        模拟相机绕点云中心旋转生成虚拟视角：
        1. 点云固定在世界坐标系（中心为原点）
        2. 相机绕点云中心旋转（改变相机外参）
        3. 将点云从世界坐标系转换到相机坐标系
        """
        virtual_images = []
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        H, W = img_size
        K = camera_K.reshape(3, 3)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # 点云在世界坐标系中（已中心化，中心为原点）
        pcl_xyz_world = pcl_with_rgb[:, :3]  # (N, 3)，世界坐标系：点云中心为原点
        rgb_values = pcl_with_rgb[:, 3:6]  # (N, 3)

        for rx, ry in view_angles:
            # 1. 计算相机绕点云中心旋转的外参（相机姿态）
            # 相机旋转顺序：先绕世界坐标系y轴（偏航），再绕相机x轴（俯仰）
            # 旋转矩阵表示相机从初始位置（正视点云）旋转到新视角
            R_yaw = np.array(
                [  # 偏航角ry（绕世界y轴）
                    [np.cos(np.radians(ry)), 0, np.sin(np.radians(ry))],
                    [0, 1, 0],
                    [-np.sin(np.radians(ry)), 0, np.cos(np.radians(ry))],
                ]
            )
            R_pitch = np.array(
                [  # 俯仰角rx（绕相机x轴）
                    [1, 0, 0],
                    [0, np.cos(np.radians(rx)), -np.sin(np.radians(rx))],
                    [0, np.sin(np.radians(rx)), np.cos(np.radians(rx))],
                ]
            )
            R_cam = R_pitch @ R_yaw  # 相机旋转矩阵（世界→相机）

            # 2. 将点云从世界坐标系转换到相机坐标系
            # 相机坐标系定义：z轴向前（指向点云），x轴向右，y轴向下
            pcl_xyz_cam = (R_cam @ pcl_xyz_world.T).T  # (N, 3)，相机坐标系下的点云

            # 3. 过滤相机前方的点（z>0）
            valid_z = pcl_xyz_cam[:, 2] > 1e-3  # 只保留相机前方的点
            if not np.any(valid_z):
                # 无有效点，生成空白图像
                virtual_img = np.full((H, W, 3), background_color, dtype=np.uint8)
                virtual_images.append(virtual_img)
                continue
            x_cam = pcl_xyz_cam[valid_z, 0]
            y_cam = pcl_xyz_cam[valid_z, 1]
            z_cam = pcl_xyz_cam[valid_z, 2]
            rgb_valid = rgb_values[valid_z]

            # 4. 投影到图像平面
            u = (x_cam * fx / z_cam + cx).astype(int)
            v = (y_cam * fy / z_cam + cy).astype(int)
            # 裁剪到图像范围内
            u = np.clip(u, 0, W - 1)
            v = np.clip(v, 0, H - 1)

            # 5. 处理遮挡（深度缓冲区）
            virtual_img = np.full((H, W, 3), background_color, dtype=np.uint8)
            depth_buffer = np.ones((H, W)) * np.inf  # 记录每个像素的最小z值（最近点）
            # 按深度排序（近的点后绘制，覆盖远的点）
            sorted_indices = np.argsort(z_cam)[::-1]  # 从近到远
            for idx in sorted_indices:
                ui, vi = u[idx], v[idx]
                zi = z_cam[idx]
                if zi < depth_buffer[vi, ui]:
                    depth_buffer[vi, ui] = zi
                    virtual_img[vi, ui] = rgb_valid[idx]

            virtual_images.append(virtual_img)
        return virtual_images

    # @staticmethod
    # def transform_point_cloud_2_spherical_coords(point_cloud):
    #     """点云转球坐标"""
    #     pass


"""-------------------- Tests --------------------"""


def demo01(points):
    """最远点采样"""
    pointSample = PointCloudSample(points, 1024)
    ids, points = pointSample.farthest_point_sample()
    return ids, points


def demo02(points):
    """随机采样"""
    pointSample = PointCloudSample(points, 1024)
    ids, points = pointSample.random_sample()
    return ids, points


def demo03(points):
    """几何采样"""
    pointSample = PointCloudSample(points, 1024)
    ids, points = pointSample.geometric_sampling(16)
    return ids, points


def demo04(points):
    """"""
    pointSample = PointCloudSample(points, 1024)


if __name__ == "__main__":
    # 示例用法
    # 生成随机点云数据进行测试
    np.random.seed(42)
    points = np.random.rand(7000, 3) * 100  # 随机分布点
    points[:3000, 2] = 0  # XY平面
    points[3000:5000] = points[3000:5000] * 0.2 + np.array([50, 50, 30])
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points)
    # PointCloudUtils.visualize_point_cloud(pcd)

    # ids, points = demo01(points)
    # ids, points = demo02(points)
    ids, points = demo03(points)

    pcd.points = o3d.utility.Vector3dVector(points)

    PointCloudUtils.visualize_point_cloud(pcd)
