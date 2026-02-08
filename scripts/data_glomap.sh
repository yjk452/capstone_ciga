
# base/input 경로에 이미지들이 존재하고, base 디렉토리는 input 폴더 제외 아무것도 존재하지 않는 상태
# 이미지셋으로부터 SfM 과정 + estimated_depth 생성 과정을 수행
BASE=/home/yeonchanheum/data/dataset/SF/test1.6d5


colmap feature_extractor \
    --database_path $BASE/database.db \
    --image_path $BASE/input/ \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model OPENCV \

# colmap sequential_matcher \
#     --database_path $BASE/database.db \
#     --SequentialMatching.overlap 20 \

colmap exhaustive_matcher \
  --database_path $BASE/database.db \

glomap mapper \
    --database_path $BASE/database.db \
    --image_path $BASE/input \
    --output_path $BASE/sparse \

colmap image_undistorter \
    --image_path $BASE/images \
    --input_path $BASE/distorted/sparse/0 \
    --output_path $BASE \
    --output_type COLMAP

gpu_id=0 \
echo "GPU $gpu_id is available." \
CUDA_VISIBLE_DEVICES=$gpu_id python ~/CityGaussianORG_V2/utils/estimate_dataset_depths.py $BASE \
