import os
import shutil

# Đường dẫn tới thư mục Dataset
base_path = "Dataset/Altered"

# Duyệt qua tất cả các thư mục con và file
for root, dirs, files in os.walk(base_path):
    for file in files:
        # Kiểm tra định dạng file (ở đây là BMP)
        if file.endswith(".BMP"):
            current_file_path = os.path.join(root, file)

            # Lấy thư mục cha của file (Person_1, Person_2, ...)
            relative_path = os.path.relpath(root, base_path)
            person_folder = relative_path.split(os.sep)[0]  # Lấy Person_1

            # Tạo đường dẫn đích mới
            new_dir = os.path.join(base_path, person_folder)
            new_file_path = os.path.join(new_dir, file)

            # Đảm bảo thư mục đích tồn tại
            os.makedirs(new_dir, exist_ok=True)

            # Di chuyển file tới vị trí mới
            shutil.move(current_file_path, new_file_path)

# Sau khi di chuyển file, xóa các thư mục rỗng còn lại
for root, dirs, files in os.walk(base_path, topdown=False):
    for dir in dirs:
        dir_path = os.path.join(root, dir)
        if not os.listdir(dir_path):  # Nếu thư mục trống
            os.rmdir(dir_path)

print("Refactor hoàn tất!")
