import struct
import argparse
import os

# COLMAP에서 제공하는 바이너리 리더 함수
def read_images_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = struct.unpack("Q", fid.read(8))[0]
        for _ in range(num_reg_images):
            binary_image_properties = struct.unpack("IQdddddddu", fid.read(64))
            image_id = binary_image_properties[0]
            qvec = float(binary_image_properties[1]), float(binary_image_properties[2]), float(binary_image_properties[3]), float(binary_image_properties[4])
            tvec = float(binary_image_properties[5]), float(binary_image_properties[6]), float(binary_image_properties[7])
            camera_id = binary_image_properties[8]
            
            image_name = ""
            current_char = struct.unpack("c", fid.read(1))[0]
            while current_char != b"\x00":   # 널 문자(\x00)가 나올 때까지 읽음
                image_name += current_char.decode("utf-8")
                current_char = struct.unpack("c", fid.read(1))[0]
            
            num_points2D = struct.unpack("Q", fid.read(8))[0]
            # 포인트 데이터는 건너뜀 (이미지 이름만 확인하면 되므로)
            fid.read(num_points2D * 24) # x,y(16) + id(8) = 24 bytes per point
            
            images[image_id] = (image_name, camera_id)
    return images

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # images.bin 파일이 있는 폴더 경로 (예: sparse/0/)
    parser.add_argument("--model_path", type=str, required=True) 
    args = parser.parse_args()

    bin_file = os.path.join(args.model_path, "images.bin")
    
    if not os.path.exists(bin_file):
        print(f"Error: {bin_file} 파일을 찾을 수 없습니다.")
        exit(1)

    print(f"Reading {bin_file}...")
    images = read_images_binary(bin_file)
    
    print(f"총 {len(images)}개의 이미지가 등록되어 있습니다.\n")
    print("--- 이미지 이름 예시 (상위 10개) ---")
    
    # 딕셔너리 정렬 후 출력
    for idx, (img_id, (name, cam_id)) in enumerate(sorted(images.items())):
        if idx >= 10: