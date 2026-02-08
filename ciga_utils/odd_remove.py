import os
import glob

def remove_even_numbered_images(folder_path, extension="jpg"):
    """
    파일명이 숫자로 된 이미지들 중 짝수 번호(예: 0002.jpg, 0004.jpg)를 삭제합니다.
    """
    
    # 1. 파일 찾기
    search_pattern = os.path.join(folder_path, f"*.{extension}")
    files = sorted(glob.glob(search_pattern))

    if not files:
        print("해당 폴더에서 이미지를 찾을 수 없습니다.")
        return

    print(f"총 {len(files)}개의 파일을 검사합니다...")

    deleted_count = 0

    # 2. 파일명 분석 및 삭제
    for file_path in files:
        filename = os.path.basename(file_path)
        name_without_ext = os.path.splitext(filename)[0] # "0002.jpg" -> "0002"

        try:
            # 파일명을 숫자로 변환
            number = int(name_without_ext)
            
            # 짝수인지 확인 (2로 나누어 떨어지면 짝수)
            if number % 2 == 0:
                os.remove(file_path)
                # print(f"[삭제됨] {filename}") # 삭제 내역을 다 보고 싶으면 주석 해제
                deleted_count += 1
                
        except ValueError:
            # 파일명이 숫자가 아닌 경우(예: hidden files) 건너뜀
            continue

    print("-" * 30)
    print(f"작업 완료! 총 {deleted_count}장의 짝수 이미지를 삭제했습니다.")
    print(f"남은 이미지: {len(files) - deleted_count}장")

# ==========================================
# 사용 설정
# ==========================================
if __name__ == "__main__":
    # 이미지가 있는 폴더 경로
    TARGET_FOLDER = "/home/yeonchanheum/data/TUK/input" 
    
    # 확장자 설정
    TARGET_EXTENSION = "jpg"

    remove_even_numbered_images(TARGET_FOLDER, TARGET_EXTENSION)