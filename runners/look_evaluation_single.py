import pickle

# 替换为实际的all_dm.pkl路径（通常在results/evaluation_results/single/下）
result_path = "/home/dai/ai/GenPose2/results/evaluation_results/single/aggregated.pkl"

# 加载数据
with open(result_path, "rb") as f:
    detect_match = pickle.load(f)

# 查看基本信息（示例）
print("预测姿态数量：", len(detect_match.pred_affine))  # 预测的姿态矩阵
print("真实姿态数量：", len(detect_match.gt_affine))  # 真实的姿态矩阵
print("旋转误差（示例）：", detect_match.rot_errors[:5])  # 前5个样本的旋转误差
print("平移误差（示例）：", detect_match.trans_errors[:5])  # 前5个样本的平移误差
