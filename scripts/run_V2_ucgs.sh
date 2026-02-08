BASE=configs/ciga_yaml/v2
COARSE_NAME=mlp_v2_coarse
NAME=mlp_v2_train_partition
TEST_PATH=/home/yeonchanheum/data/dataset/SF/train1.5
PROJECT=V2_ucdata


#gpu_id=$(get_available_gpu)
gpu_id=0

# # 전역 학습
# echo "GPU $gpu_id is available."
# CUDA_VISIBLE_DEVICES=$gpu_id python main.py fit \
#                                     --config $BASE/$COARSE_NAME.yaml \
#                                     -n $COARSE_NAME 

# echo "qlalfqjsgh" | sudo -S shutdown -h now

python main.py fit --config configs/ciga_yaml/v2/mlp_v2_train_partition.yaml --data.parser.block_id 6

python main.py fit --ckpt_path /home/yeonchanheum/capstone_ciga/outputs/mlp_v2_train_partition/blocks/block_7/checkpoints/epoch=67-step=9999.ckpt --config configs/ciga_yaml/v2/mlp_v2_train_partition.yaml --data.parser.block_id 7

python main.py fit --config configs/ciga_yaml/v2/mlp_v2_train_partition.yaml --data.parser.block_id 8


# 전역 장면 분할
# echo "GPU $gpu_id is available."
# CUDA_VISIBLE_DEVICES=$gpu_id python utils/partition_citygs.py --config_path configs/$NAME.yaml --force  # --reorient


# 분할 학습
#python utils/train_citygs_partitions_ciga.py -n $NAME -p $PROJECT -c $BASE

# 병합
# echo "GPU $gpu_id is available."
# CUDA_VISIBLE_DEVICES=$gpu_id python utils/merge_ciga.py outputs/$NAME \
