import os
import glob

def renumber_images_start_from_1(folder_path, extension="jpg"):
    """
    폴더 내의 이미지를 읽어서 0001.ext 부터 순차적으로 재정렬합니다.
    안전한 변경을 위해 임시 이름 변경 -> 최종 이름 변경 2단계를 거칩니다.
    """
    
    # 1. 파일 찾기
    search_pattern = os.path.join(folder_path, f"*.{extension}")
    files = sorted(glob.glob(search_pattern))

    if not files:
        print("해당 폴더에서 이미지를 찾을 수 없습니다.")
        return

    print(f"총 {len(files)}개의 파일을 발견했습니다. 작업을 시작합니다...")

    # 2. [안전 장치] 임시 이름으로 먼저 변경 (temp_000000.jpg ...)
    temp_files = []
    for idx, file_path in enumerate(files):
        directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        file_ext = filename.split('.')[-1]
        
        temp_name = f"temp_{idx:06d}.{file_ext}"
        temp_path = os.path.join(directory, temp_name)
        
        os.rename(file_path, temp_path)
        temp_files.append(temp_path)

    print("임시 이름 변경 완료. 0001번부터 최종 넘버링을 진행합니다.")

    # 3. 최종 이름(0001.jpg ~)으로 변경
    for idx, temp_path in enumerate(temp_files):
        directory = os.path.dirname(temp_path)
        file_ext = temp_path.split('.')[-1]
        
        # [수정된 부분] idx는 0부터 시작하므로 1을 더해줍니다.
        # 결과: 0001.jpg, 0002.jpg ...
        new_name = f"{idx + 1:04d}.{file_ext}"
        new_path = os.path.join(directory, new_name)
        
        os.rename(temp_path, new_path)

    print(f"작업 완료! {os.path.basename(temp_files[0])} -> {os.path.basename(new_path)} (총 {len(files)}장)")

# ==========================================
# 사용 설정
# ==========================================
if __name__ == "__main__":
    # 이미지가 있는 폴더 경로를 여기에 입력하세요
    TARGET_FOLDER = "/home/yeonchanheum/data/TUK/input" 
    
    # 확장자 설정 (jpg, png 등)
    TARGET_EXTENSION = "jpg"

    renumber_images_start_from_1(TARGET_FOLDER, TARGET_EXTENSION)