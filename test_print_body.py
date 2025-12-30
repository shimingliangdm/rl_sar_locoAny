import numpy as np

# 替换你的路径
file_path = "/home/dm/rl_sar_locoAny/policy/g1/whole_body_tracking/lafan_motion/retarget_poses_g1_interact_29dof.npz" 
data = np.load(file_path, allow_pickle=True)
body_pos = data['body_pos_w'] # (Frames, 30, 3)

print("=== Body Height Scan ===")
print(f"{'Index':<6} | {'Avg Height (m)':<15} | {'Guess'}")
print("-" * 40)

# 存储 (index, height) 以便排序分析（可选）
stats = []

for i in range(body_pos.shape[1]):
    avg_z = np.mean(body_pos[:, i, 2])
    stats.append((i, avg_z))
    
    # 简单的推测逻辑
    guess = ""
    if avg_z < 0.25: guess = "Foot/Ankle ??"
    elif 0.8 < avg_z < 1.2: guess = "ROOT / Pelvis / Torso ??"
    elif avg_z > 1.5: guess = "Head ??"
    
    print(f"{i:<6} | {avg_z:<15.4f} | {guess}")

print("-" * 40)
