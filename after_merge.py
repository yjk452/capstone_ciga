import torch

# 1. 파일 로드
ckpt_path = "/home/jinholee/Ciga/git/tmp/capstone_ciga/outputs/v2block_final/checkpoints/epoch=49-step=30000.ckpt" 
ckpt = torch.load(ckpt_path, map_location="cpu")

# 2. 기준이 되는 가우시안 개수 확인 (1083619)
target_num = ckpt['state_dict']['gaussian_model.gaussians.means'].shape[0]
print(f"Target Gaussian Count: {target_num}")

# 3. Density Controller 관련 항목들 초기화 (크기 맞추기)
# 이 항목들은 학습 시 다시 쌓여야 하는 값들이므로 0으로 초기화하는 것이 정석입니다.
density_keys = [
    "density_controller.max_radii2D",
    "density_controller.xyz_gradient_accum",
    "density_controller.denom"
]

for key in density_keys:
    if key in ckpt['state_dict']:
        old_shape = ckpt['state_dict'][key].shape
        # 기존 4.2M 데이터를 버리고 1.08M 크기의 0 텐서로 교체
        new_tensor = torch.zeros((target_num, *old_shape[1:]), dtype=ckpt['state_dict'][key].dtype)
        ckpt['state_dict'][key] = new_tensor
        print(f"Fixed {key}: {old_shape} -> {new_tensor.shape}")

# 4. 수정된 체크포인트 저장
torch.save(ckpt, "outputs/v2block_final/checkpoints/merge_fixed.ckpt")
print("Successfully saved fixed checkpoint to 'merged_fixed.ckpt'")

# 5. 옵티마이저 상태 초기화
if "optimizer_states" in ckpt:
    print(f"[DEBUG] Removing old optimizer states (size mismatch prevention)")
    ckpt["optimizer_states"] = [] # 빈 리스트로 초기화

# 6. 루프/에폭 정보 초기화 
# Resume 시 발생할 수 있는 경고를 방지
if "loops" in ckpt:
    del ckpt["loops"]

# 7. 수정된 체크포인트 저장
import os
save_path = "outputs/v2block_final/checkpoints/merge_fixed_v2.ckpt"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(ckpt, save_path)
print(f"Successfully saved to: {save_path}")