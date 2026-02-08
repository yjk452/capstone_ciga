import sys
import os
import torch

# 현재 실행 위치(프로젝트 루트)를 파이썬 경로에 추가
sys.path.append(os.getcwd())

# Ours_3
#ckpt_path = "outputs/mlp_v2_train_partition_3/blocks/block_6/checkpoints/epoch=64-step=30000.ckpt" # 실제 경로로 수정
#ckpt_path = "outputs/mlp_v2_train_partition_3/checkpoints/epoch=43-step=30000.ckpt"
#ckpt_path = "outputs/mlp_v2_coarse/checkpoints/epoch=43-step=30000.ckpt"
#ckpt_path = "outputs/mlp_v2_train_partition_3/blocks/block_0/checkpoints/epoch=65-step=30000.ckpt"
#ckpt_path = "outputs/mlp_v2_train_partition_3/blocks/block_1/checkpoints/epoch=477-step=30000.ckpt"
#ckpt_path = "outputs/mlp_v2_train_partition_3/blocks/block_2/checkpoints/epoch=300-step=30000.ckpt" 
# Ours_1
#ckpt_path = "outputs/mlp_v1_coarse/checkpoints/epoch=43-step=30000.ckpt"
#ckpt_path = "outputs/mlp_v1_train_partition/checkpoints/epoch=43-step=30000.ckpt"
#ckpt_path = "outputs/mlp_v1_train_partition/blocks/block_0/checkpoints/epoch=62-step=30000.ckpt"
#ckpt_path = "outputs/mlp_v1_train_partition/blocks/block_1/checkpoints/epoch=358-step=30000.ckpt"
#ckpt_path = "outputs/mlp_v1_train_partition/blocks/block_2/checkpoints/epoch=400-step=30000.ckpt"
#ckpt_path = "outputs/mlp_v1_train_partition/blocks/block_3/checkpoints/epoch=111-step=30000.ckpt"

#ckpt_path = "outputs/mlp_v1_final/checkpoints/epoch=15-step=10000.ckpt"

# CityGS V2
#ckpt_path = "/home/yeonchanheum/CityGaussianORG_V2/outputs/V2_coarse/checkpoints/epoch=43-step=30000.ckpt"
#ckpt_path = "/home/yeonchanheum/CityGaussianORG_V2/outputs/V2/checkpoints/epoch=43-step=30000.ckpt"
#ckpt_path = "/home/yeonchanheum/CityGaussianORG_V2/outputs/V2/blocks/block_0/checkpoints/epoch=59-step=30000.ckpt"
#ckpt_path = "/home/yeonchanheum/CityGaussianORG_V2/outputs/V2/blocks/block_1/checkpoints/epoch=477-step=30000.ckpt"
#ckpt_path = "/home/yeonchanheum/CityGaussianORG_V2/outputs/V2/blocks/block_2/checkpoints/epoch=455-step=30000.ckpt"
# test
#ckpt_path = "outputs/test/checkpoints/epoch=2-step=1000.ckpt" 
#ckpt_path = "outputs/test2/blocks/block_0/checkpoints/epoch=3-step=1000.ckpt" 

# V3
ckpt_path="outputs/mlp_v3_coarse/checkpoints/epoch=43-step=30000.ckpt"
try:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    print("성공적으로 로드되었습니다!")
    #print("Keys:", checkpoint.keys())
    

    # 가우시안 개수 확인 예시
    if "state_dict" in checkpoint:
        sd = checkpoint["state_dict"]
        print(f"가우시안 개수: {sd['gaussian_model.gaussians.means'].shape[0]}")
        #gau_pos = sd['gaussian_model.gaussians.means']
        #print(gau_pos.max(0), gau_pos.min(0))
            
    #MLP 관련 키가 있는지 확인
    # mlp_keys = [k for k in checkpoint.get("state_dict", {}).keys() if "mlp" in k]
    # print(f"MLP 관련 파라미터 수: {len(mlp_keys)}")
    # for idx, i in enumerate(mlp_keys):
    #     print(f"{idx}번째 MLP 파라미터 :{i}, ")
    #     print(sd[i])
    #print(sd["mlp_model.mlp.0.weight"].min(0))
    


    keys = [k for k in checkpoint.get("state_dict", {}).keys()]
    for i in keys:
        print(i)
        print(sd[i].shape)

    # hparams = [k for k in checkpoint.get("hyper_parameters", {}).keys()]
    
    # if "density" in hparams:
    #     density_cfg = hparams["density"]
    #     print("--- Densification 관련 설정 ---")
        
    #     # jsonargparse 구조인 경우 init_args 내부에 데이터가 있습니다.
    #     if "init_args" in density_cfg:
    #         args = density_cfg["init_args"]
    #         for key, value in args.items():
    #             print(f"{key}: {value}")
    #     else:
    #         print(density_cfg)
    # else:
    #     print("해당 체크포인트에서 density 설정을 찾을 수 없습니다.")
except Exception as e:
    print(f"에러 발생: {e}")