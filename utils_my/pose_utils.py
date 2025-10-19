import torch
import numpy as np

from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
from pytorch3d.transforms.se3 import se3_exp_map  # , Transform3d
from pytorch3d.transforms.transform3d import (
    Transform3d,
    _check_valid_rotation_matrix,
)
from typing import Union


def _check_R_t(
    rotation: Union[torch.Tensor, np.ndarray],
    translation: Union[torch.Tensor, np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(rotation, np.ndarray):
        rotation = torch.tensor(rotation)
    if isinstance(translation, np.ndarray):
        translation = torch.tensor(translation)
    if rotation.shape == (1, 3, 3) and translation.shape == (
        1,
        3,
    ):
        return rotation, translation
    if rotation.shape == (3, 3):
        rotation = rotation.unsqueeze(0)
    if translation.shape == (3, 1):
        translation = translation.flatten()
        translation = translation.unsqueeze(0)
    return rotation, translation


"""-------------------- Interface --------------------"""


class PoseUtils:
    @staticmethod
    def convert_pose_2_se3(
        rotation: torch.Tensor, translation: torch.Tensor
    ) -> torch.Tensor:
        """
        将旋转矩阵R和平移向量t转换为SE(3)李代数表示
        Args:
            R: 3x3旋转矩阵
            t: 长度为3的平移向量
        Returns:
            se3: 长度为6的李代数向量，格式为[平移部分, 旋转部分]
        """
        if not isinstance(rotation, torch.Tensor):
            rotation = torch.tensor(rotation)
        if not isinstance(translation, torch.Tensor):
            translation = torch.tensor(translation)

        rotation, translation = _check_R_t(rotation, translation)
        device = rotation.device
        _check_valid_rotation_matrix(rotation, tol=1e-6)
        se3_matrix = torch.zeros((1, 4, 4), dtype=rotation.dtype, device=device)
        se3_matrix[:, :3, :3] = rotation
        # se3_matrix[:, 3, :3] = translation  # 将平移向量放到R下面 [R,0],[t,1]
        se3_matrix[:, :3, 3] = translation
        se3_matrix[:, 3, 3] = 1.0
        se3_transform = Transform3d(matrix=se3_matrix)
        se3_lie = se3_transform.get_se3_log(eps=1e-6, cos_bound=1e-6)
        return se3_lie.squeeze(0)  # 返回长度为6的向量

    @staticmethod
    def convert_pose_2_SE3():

        pass

    @staticmethod
    def convert_pose_2_6d():
        pass

    @staticmethod
    def convert_pose_2_quaternion(
        rotation: torch.Tensor, translation: torch.Tensor
    ) -> torch.Tensor:
        """
        w, x, y, z

        """
        rotation, translation = _check_R_t(rotation, translation)
        quaternion = matrix_to_quaternion(rotation)
        return quaternion, translation

    @staticmethod
    def convert_quaternion_2_rotation(quaternion):
        """
        (w, x, y, z)
        """
        if isinstance(quaternion, np.ndarray):
            quaternion = torch.tensor(quaternion)
        if quaternion.shape == 4:
            quaternion.unsqueeze(0)
        rotation_matrix = quaternion_to_matrix(quaternion)
        return rotation_matrix

    @staticmethod
    def convert_se3_2_SE3(se3: torch.Tensor, esp=1e-8) -> torch.Tensor:
        """
        将SE(3)李代数表示转换为SE(3)矩阵
        Args:
            se3: 长度为6的李代数向量，格式为[平移部分, 旋转部分]
        Returns:
            SE3: 4x4的SE(3)矩阵
        """
        SE3 = se3_exp_map(se3, eps=esp)
        return SE3

    @staticmethod
    def convert_se3_2_pose(se3) -> tuple[np.ndarray, np.ndarray]:
        """
        将SE(3)李代数表示转换为旋转矩阵R和平移向量t

        Args:
            se3: 长度为6的李代数向量，格式为[平移部分, 旋转部分]

        Returns:
            R: 3x3旋转矩阵
            t: 长度为3的平移向量
        """
        if isinstance(se3, np.ndarray):
            se3 = torch.tensor(se3).unsqueeze(0)
        SE3 = se3_exp_map(se3, eps=1e-6)
        return SE3[0]

    @staticmethod
    def convert_SE3_2_quaternion():
        pass
