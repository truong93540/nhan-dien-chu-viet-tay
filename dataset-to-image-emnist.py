import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

# Đường dẫn file CSV và file mapping
csv_file = "emnist/emnist-letters-train.csv"  # Thay bằng đường dẫn file CSV của bạn
mapping_file = "emnist/emnist-letters-mapping.txt"  # Thay bằng đường dẫn file mapping
output_folder = "emnist_letters_images"

# Đọc file mapping
print("Đang đọc file mapping...")
mapping = {}
with open(mapping_file, "r") as f:
    for line in f:
        cols = line.strip().split()  # Tách cột trong file mapping
        label = int(cols[0])         # Nhãn (class_label)
        char_ascii = int(cols[1])    # Giá trị ASCII của chữ cái viết hoa
        mapping[label] = chr(char_ascii)  # Tạo ánh xạ: nhãn -> chữ cái

# Đọc file CSV
print("Đang đọc file CSV...")
data = pd.read_csv(csv_file, header=None)

# Nhãn (cột đầu tiên)
labels = data.iloc[:, 0].values  # Lấy nhãn (cột đầu tiên)
images = data.iloc[:, 1:].values  # Lấy dữ liệu pixel (784 cột sau)

# Tạo thư mục đầu ra
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Tạo thư mục con cho từng chữ cái
for letter in set(mapping.values()):
    letter_folder = os.path.join(output_folder, letter)
    if not os.path.exists(letter_folder):
        os.makedirs(letter_folder)

# Xuất ảnh
print("Bắt đầu lưu ảnh...")
for idx, (label, pixel_values) in tqdm(enumerate(zip(labels, images)), total=len(labels)):
    # Chuyển đổi pixel thành ảnh (28x28)
    image_array = np.array(pixel_values, dtype=np.uint8).reshape(28, 28)
    
    # Lấy ký tự từ nhãn
    letter = mapping.get(label, "unknown")
    letter_folder = os.path.join(output_folder, letter)
    
    # Đường dẫn ảnh
    image_path = os.path.join(letter_folder, f"{letter}_{idx}.png")
    image = Image.fromarray(image_array, mode="L")  # Grayscale image
    image = image.transpose(Image.Transpose.ROTATE_270)  # Xoay ảnh
    image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)  # Lật ngang
    image.save(image_path)

print(f"Lưu ảnh hoàn tất! Ảnh được lưu tại thư mục: {output_folder}")
