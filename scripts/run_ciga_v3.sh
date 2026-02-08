BASE=configs/ciga_yaml/v3
COARSE_NAME=mlp_v3_coarse
NAME=mlp_v3_train_partition
FINAL=mlp_v3_merge
TEST_PATH=/home/yeonchanheum/data/dataset/SF/train1.5
PROJECT=ucdata_V3




#python main.py fit --config $BASE/$COARSE_NAME.yaml -n $COARSE_NAME

# python main.py fit --config $BASE/$NAME.yaml -n $NAME --data.parser.block_id 0
# python main.py fit --config $BASE/$NAME.yaml -n $NAME --data.parser.block_id 1
# python main.py fit --config $BASE/$NAME.yaml -n $NAME --data.parser.block_id 2
# python main.py fit --config $BASE/$NAME.yaml -n $NAME --data.parser.block_id 3
# python main.py fit --config $BASE/$NAME.yaml -n $NAME --data.parser.block_id 4
#python main.py fit --config $BASE/$NAME.yaml -n $NAME --data.parser.block_id 5
#python main.py fit --config $BASE/$NAME.yaml -n $NAME --data.parser.block_id 6
#python main.py fit --config $BASE/$NAME.yaml -n $NAME --data.parser.block_id 7
python main.py fit --config $BASE/$NAME.yaml -n $NAME --data.parser.block_id 8

#echo "qlalfqjsgh" | sudo -S shutdown -h now