import cv2
import os

# Đường dẫn tới thư mục ảnh gốc
input_dir = "dataset"  # Thay bằng đường dẫn của bạn
output_dir = "output"  # Thư mục lưu ảnh kết quả

# Tạo thư mục đầu ra nếu chưa tồn tại
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Lấy danh sách tất cả các file ảnh trong thư mục
image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.bmp'))]

# Kiểm tra số lượng ảnh
print(f"Number of images found: {len(image_files)}")

# Duyệt qua từng ảnh trong dataset
for idx, image_file in enumerate(image_files, start=1):
    # Đọc ảnh
    image_path = os.path.join(input_dir, image_file)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Kiểm tra ảnh đã đọc thành công chưa
    if image is None:
        print(f"[{idx}/{len(image_files)}] Error reading image {image_file}. Skipping...")
        continue

    # Áp dụng Otsu's Thresholding
    _, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Lưu ảnh sau xử lý
    output_path = os.path.join(output_dir, image_file)
    cv2.imwrite(output_path, otsu_thresh)

