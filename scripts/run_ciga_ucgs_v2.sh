BASE=configs/ciga_yaml/v2
COARSE_NAME=mlp_v2_coarse
NAME=mlp_v2_train_partition
TEST_PATH=/home/yeonchanheum/data/dataset/SF/train1.5
PROJECT=ucdata_v2


#gpu_id=$(get_available_gpu)
gpu_id=0

# 전역 학습
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python main.py fit \
                                    --config $BASE/$COARSE_NAME.yaml \
                                    -n $COARSE_NAME 

# 테스트
echo "GPU $gpu_id is available."
CFG=$(ls -d outputs/$COARSE_NAME/lightning_logs/version_* | sort -V | tail -n 1)/config.yaml
CUDA_VISIBLE_DEVICES=$gpu_id python main.py test \
    --config "$CFG" \
    --save_val \
    --data.path $TEST_PATH 


# 전역 장면 분할
echo "GPU $gpu_id is available."
# CUDA_VISIBLE_DEVICES=$gpu_id python utils/partition_citygs.py --config_path configs/$NAME.yaml --force  # --reorient
CUDA_VISIBLE_DEVICES=$gpu_id python utils/partition_ciga.py --config_path $BASE/$NAME.yaml --force

# # 분할 학습
# python utils/train_citygs_partitions_ciga.py -n $NAME -p $PROJECT -c $BASE

# # 병합
# echo "GPU $gpu_id is available."
# CUDA_VISIBLE_DEVICES=$gpu_id python utils/merge_ciga.py outputs/$NAME \

# # mlp 병합
# echo "GPU $gpu_id is available."
# CUDA_VISIBLE_DEVICES=$gpu_id python internal/models/ensamble_mlp.py outputs/$NAME 


# # ensamble mlp로 테스트
# CFG=$(ls -d outputs/$COARSE_NAME/lightning_logs/version_* | sort -V | tail -n 1)/config.yaml
# ENSAMBLE_MLP_PATH=(outputs/$NAME/checkpoints_mlp/*.pt)
# COARSE_MLP_PATH=(outputs/$COARSE_NAME/checkpoints_mlp/*.pt)
# echo "GPU $gpu_id is available."
# CUDA_VISIBLE_DEVICES=$gpu_id python main.py test \
#     --config "$CFG" \
#     -n $NAME \
#     --data.path $TEST_PATH \
#     --model.mlp_cfg.path "$ENSAMBLE_MLP_PATH" \
#     --save_val \
#     --test_speed 

# # Coarse mlp로 테스트
# echo "GPU $gpu_id is available."
# CUDA_VISIBLE_DEVICES=$gpu_id python main.py test \
#     --config "$CFG" \
#     -n $NAME \
#     --data.path $TEST_PATH \
#     --model.mlp_cfg.path "$COARSE_MLP_PATH" \
#     --save_val \
#     --test_speed 

# echo "GPU $gpu_id is available."
# CUDA_VISIBLE_DEVICES=$gpu_id python utils/gs2d_mesh_extraction.py \
#                                       outputs/$NAME \
#                                     --voxel_size 0.01 \
#                                     --sdf_trunc 0.04 \
#                                     --depth_trunc 5.0 \

# echo "GPU $gpu_id is available."
# CUDA_VISIBLE_DEVICES=$gpu_id python tools/eval_tnt/run.py \
#                                     --scene Block_all_ds \
#                                     --dataset-dir data/geometry_gt/MC_Aerial \
#                                     --ply-path "outputs/$NAME/fuse_post.ply"
