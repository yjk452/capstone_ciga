import torch
import sys
import os

# 경로 추가
sys.path.append(os.getcwd())

# Ours_1
#ckpt_path = "outputs/mlp_v1_coarse/checkpoints/epoch=43-step=30000.ckpt"
#ckpt_path = "outputs/mlp_v1_org/mlp_v1_train_partition/checkpoints/epoch=43-step=30000.ckpt"
#ckpt_path = "outputs/mlp_v1_train_partition/blocks/block_0/checkpoints/epoch=62-step=30000.ckpt"
#ckpt_path = "outputs/mlp_v1_train_partition/blocks/block_1/checkpoints/epoch=358-step=30000.ckpt"
#ckpt_path = "outputs/mlp_v1_train_partition/blocks/block_2/checkpoints/epoch=400-step=30000.ckpt"
# CityGS V2
#ckpt_path = "/home/yeonchanheum/CityGaussianORG_V2/outputs/V2_coarse/checkpoints/epoch=43-step=30000.ckpt"
ckpt_path = "/home/yeonchanheum/CityGaussianORG_V2/outputs/V2/checkpoints/epoch=43-step=30000.ckpt"
#ckpt_path = "/home/yeonchanheum/CityGaussianORG_V2/outputs/V2/blocks/block_0/checkpoints/epoch=59-step=30000.ckpt"
#ckpt_path = "/home/yeonchanheum/CityGaussianORG_V2/outputs/V2/blocks/block_1/checkpoints/epoch=477-step=30000.ckpt"
#ckpt_path = "/home/yeonchanheum/CityGaussianORG_V2/outputs/V2/blocks/block_2/checkpoints/epoch=455-step=30000.ckpt"
try:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    print("성공적으로 로드되었습니다!")

    hparams = checkpoint.get("hyper_parameters", {})
    
    if "density" in hparams:
        print("\n=== [Densification 관련 설정] ===")
        density_obj = hparams["density"]
        
        # 1. 객체인 경우와 딕셔너리인 경우 모두 대응
        if hasattr(density_obj, "__dict__"):
            # 객체라면 내부 변수들을 딕셔너리로 변환
            density_args = vars(density_obj)
        elif isinstance(density_obj, dict):
            # 딕셔너리라면 그대로 사용 (또는 init_args 추출)
            density_args = density_obj.get("init_args", density_obj)
        else:
            density_args = {}

        # 2. 주요 파라미터 출력
        relevant_keys = [
            'densify_from_iter', 'densify_until_iter', 'densification_interval',
            'densify_grad_threshold', 'prune_interval', 'opacity_reset_interval'
        ]
        
        found_any = False
        for key in relevant_keys:
            if key in density_args:
                print(f"{key}: {density_args[key]}")
                found_any = True
        
        # 3. 만약 위 키들이 안 보인다면 객체의 모든 속성 출력
        if not found_any:
            print("\n[상세 속성 목록]")
            for k, v in density_args.items():
                if not k.startswith('_'): # 내부 변수 제외
                    print(f"{k}: {v}")
    else:
        print("\n'density' 설정을 찾을 수 없습니다.")

except Exception as e:
    import traceback
    print(f"에러 상세 발생: {traceback.format_exc()}")