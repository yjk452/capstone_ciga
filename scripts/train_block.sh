BASE=configs/ciga_yaml/v1
COARSE_NAME=mlp_v1_coarse
NAME=mlp_v1_train_partition
TEST_PATH=/home/yeonchanheum/data/dataset/SF/train1.5
PROJECT=ucdata_v1

#gpu_id=$(get_available_gpu)
gpu_id=0

python main.py fit --config $BASE/$NAME.yaml -n $NAME --data.parser.block_id 0
python main.py fit --config $BASE/$NAME.yaml -n $NAME --data.parser.block_id 1
python main.py fit --config $BASE/$NAME.yaml -n $NAME --data.parser.block_id 4
python main.py fit --config $BASE/$NAME.yaml -n $NAME --data.parser.block_id 5
python main.py fit --config $BASE/$NAME.yaml -n $NAME --data.parser.block_id 6
python main.py fit --config $BASE/$NAME.yaml -n $NAME --data.parser.block_id 7
python main.py fit --config $BASE/$NAME.yaml -n $NAME --data.parser.block_id 8

echo "qlalfqjsgh" | sudo -S shutdown -h now