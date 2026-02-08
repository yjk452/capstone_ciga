import torch
import sys
import os

# internal 모듈 로드를 위해 경로 추가
sys.path.append(os.getcwd())


# Ours_1
ckpt_path = "outputs/mlp_v1_coarse/checkpoints/epoch=43-step=30000.ckpt"
#ckpt_path = "outputs/mlp_v1_org/mlp_v1_train_partition/checkpoints/epoch=43-step=30000.ckpt"
#ckpt_path = "outputs/mlp_v1_train_partition/blocks/block_0/checkpoints/epoch=62-step=30000.ckpt"
#ckpt_path = "outputs/mlp_v1_train_partition/blocks/block_1/checkpoints/epoch=358-step=30000.ckpt"
#ckpt_path = "outputs/mlp_v1_train_partition/blocks/block_2/checkpoints/epoch=400-step=30000.ckpt"
# CityGS V2
#ckpt_path = "/home/yeonchanheum/CityGaussianORG_V2/outputs/V2_coarse/checkpoints/epoch=43-step=30000.ckpt"
#ckpt_path = "/home/yeonchanheum/CityGaussianORG_V2/outputs/V2/checkpoints/epoch=43-step=30000.ckpt"
#ckpt_path = "/home/yeonchanheum/CityGaussianORG_V2/outputs/V2/blocks/block_0/checkpoints/epoch=59-step=30000.ckpt"
#ckpt_path = "/home/yeonchanheum/CityGaussianORG_V2/outputs/V2/blocks/block_1/checkpoints/epoch=477-step=30000.ckpt"
#ckpt_path = "/home/yeonchanheum/CityGaussianORG_V2/outputs/V2/blocks/block_2/checkpoints/epoch=455-step=30000.ckpt"import torch
try:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    hparams = checkpoint.get("hyper_parameters", {})

    if "renderer" in hparams:
        print(f"=== [Trimming 관련 설정 확인] ===")
        renderer_obj = hparams["renderer"]
        
        # 객체인 경우 vars()로, 딕셔너리인 경우 .get()으로 접근
        if hasattr(renderer_obj, "__dict__"):
            args = vars(renderer_obj)
        else:
            args = renderer_obj.get("init_args", renderer_obj)

        # Trimming에 영향을 주는 핵심 파라미터 리스트
        trim_keys = [
            'depth_ratio',                  # 깊이 손실 반영 비율
            'prune_ratio',                  # 제거할 가우시안 비율 (기본 0.1)
            'contribution_prune_from_iter', # Trimming 시작 스텝 (중요!)
            'contribution_prune_interval',  # Trimming 수행 주기
            'diable_trimming',              # Trimming 전체 비활성화 여부 (오타 주의)
            'diable_start_trimming'         # 시작 단계 Trimming 비활성화 여부
        ]

        for key in trim_keys:
            if key in args:
                print(f" - {key}: {args[key]}")
            else:
                print(f" - {key}: 설정값을 찾을 수 없음 (기본값 사용 중일 가능성)")
    else:
        print("체크포인트 내에 'renderer' 설정이 없습니다.")

except Exception as e:
    print(f"에러 발생: {e}")