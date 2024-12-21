import cv2
import os

# Đường dẫn tới thư mục gốc chứa các thư mục con
base_input_dir = "Dataset/Real"  # Thay bằng đường dẫn của bạn
base_output_dir = "output"  # Thư mục lưu ảnh kết quả

# Tạo thư mục đầu ra nếu chưa tồn tại
if not os.path.exists(base_output_dir):
    os.makedirs(base_output_dir)

# Duyệt qua từng thư mục con trong thư mục gốc
for subdir, _, files in os.walk(base_input_dir):
    # Lấy danh sách tất cả các file ảnh trong thư mục con
    image_files = [f for f in files if f.lower().endswith(('.jpg', '.png', '.bmp'))]

    # Kiểm tra số lượng ảnh
    print(f"Processing directory: {subdir}")
    print(f"Number of images found: {len(image_files)}")

    # Duyệt qua từng ảnh trong dataset
    for idx, image_file in enumerate(image_files, start=1):
        # Đọc ảnh
        image_path = os.path.join(subdir, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Kiểm tra ảnh đã đọc thành công chưa
        if image is None:
            print(f"[{idx}/{len(image_files)}] Error reading image {image_file}. Skipping...")
            continue

        # Áp dụng Otsu's Thresholding
        _, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Tạo đường dẫn lưu ảnh kết quả tương ứng với cấu trúc thư mục gốc
        relative_path = os.path.relpath(subdir, base_input_dir)
        output_subdir = os.path.join(base_output_dir, relative_path)
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        # Lưu ảnh sau xử lý
        output_path = os.path.join(output_subdir, image_file)
        cv2.imwrite(output_path, otsu_thresh)