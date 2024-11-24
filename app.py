import tkinter as tk
from tkinter import messagebox
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from tkinter import filedialog
# Tải mô hình đã huấn luyện
model = load_model('modelHandWritten.h5')

# Dictionary chuyển từ chỉ số sang chữ cái
word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}

def predict_image(img):
    img = cv2.flip(img, 1)    # Lật ảnh theo trục dọc (chiều gương)
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) # Xoay ảnh 90 độ theo chiều kim đồng hồ và lật gương

    # Chuyển đổi ảnh và dự đoán chữ
    img_copy = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (400, 400))

    img_copy = cv2.GaussianBlur(img_copy, (7,7), 0)
    img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)

    img_final = cv2.resize(img_thresh, (28, 28))
    img_final = np.reshape(img_final, (1, 28, 28, 1))

    # Dự đoán với mô hình
    prediction = model.predict(img_final).flatten()
    
    # Chuyển các kết quả thành dictionary với phần trăm
    prediction_dict = {word_dict[i]: float(prediction[i]) * 100 for i in range(26)}

    # Sắp xếp và lấy 3 kết quả lớn nhất
    sorted_prediction = sorted(prediction_dict.items(), key=lambda x: x[1], reverse=True)[:3]
    
    return sorted_prediction


def open_file():
    # Mở cửa sổ để chọn hình ảnh
    file_path = filedialog.askopenfilename(
        title="Open Image", 
        initialdir="imgTest",  # Chỉ định thư mục imgTest là thư mục mặc định
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]  # Chỉ cho phép chọn các tệp ảnh
    )
    
    if file_path:
        # Đọc và xử lý hình ảnh
        img = cv2.imread(file_path)

        # Dự đoán chữ từ ảnh
        results = predict_image(img)

        # Hiển thị kết quả trong cửa sổ Tkinter
        result_str = "\n".join([f"{label}: {prob:.2f}%" for label, prob in results])
        
        # Cập nhật Label với kết quả
        result_label.config(text=result_str)

        # Hiển thị ảnh lên cửa sổ Tkinter
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển đổi sang RGB
        img_resized = cv2.resize(img_rgb, (300, 300))
        img_pil = Image.fromarray(img_resized)  # Chuyển từ NumPy array sang Image
        img_tk = ImageTk.PhotoImage(img_pil)  # Chuyển sang kiểu Tkinter

        # Cập nhật ảnh hiển thị
        image_label.config(image=img_tk)
        image_label.image = img_tk  # Lưu tham chiếu để ảnh không bị xóa

# Tạo cửa sổ Tkinter
root = tk.Tk()
root.title("Handwritten Character Prediction")
root.geometry("500x600")

# Tạo nút để mở file hình ảnh
open_button = tk.Button(root, text="Open Image", command=open_file, font=("Arial", 14))
open_button.pack(pady=20)

# Tạo Label để hiển thị ảnh
image_label = tk.Label(root)  # Đảm bảo khai báo trước khi sử dụng
image_label.pack(pady=10)

# Tạo Label để hiển thị kết quả dự đoán
result_label = tk.Label(root, text="Prediction Result: ", font=("Arial", 14), width=50, height=4, relief="solid")
result_label.pack(pady=20)


# Chạy ứng dụng Tkinter
root.mainloop()
