import cv2
import os
import json
from PIL import Image
import numpy as np

DATASET_DIR = "faces_db"
METADATA_FILE = os.path.join(DATASET_DIR, "metadata.json")

def create_dataset():
    """Thu thập ảnh khuôn mặt và thông tin sinh viên"""
    
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
    
    # Load metadata cũ (nếu có)
    metadata = {"uniforms": {}}
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    print("\n=== TRAIN KHUÔN MẶT VÀ THÔNG TIN SINH VIÊN ===\n")
    
    while True:
        name = input("Nhập tên sinh viên (hoặc 'q' để thoát): ").strip()
        if name.lower() == 'q':
            break
        
        if not name:
            print("⚠ Tên không được để trống!")
            continue
        
        # Nhập thông tin đồng phục
        print("\nMàu đồng phục:")
        print("1. Trắng (white)")
        print("2. Xanh navy (blue)")
        uniform_choice = input("Chọn (1/2): ").strip()
        uniform_color = "white" if uniform_choice == "1" else "blue"
        
        metadata["uniforms"][name] = uniform_color
        
        # Tạo thư mục cho sinh viên
        person_dir = os.path.join(DATASET_DIR, name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
        
        print(f"\n📸 Thu thập ảnh cho {name}...")
        print("Hướng dẫn: Nhìn thẳng vào camera, thay đổi góc độ nhẹ")
        print("Nhấn SPACE để chụp (30 ảnh) | ESC để bỏ qua\n")
        
        count = 0
        target = 30
        
        while count < target:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            progress = f"{count}/{target}"
            cv2.putText(frame, progress, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, name, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.imshow("Train Faces", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32:  # SPACE
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    face_img = gray[y:y+h, x:x+w]
                    img_path = os.path.join(person_dir, f"{name}_{count}.jpg")
                    cv2.imwrite(img_path, face_img)
                    count += 1
                    print(f"✓ Đã lưu {count}/{target}")
                else:
                    print("⚠ Không phát hiện khuôn mặt!")
            
            elif key == 27:  # ESC
                print("⚠ Đã bỏ qua")
                break
        
        print(f"\n✓ Hoàn tất thu thập cho {name}: {count} ảnh\n")
    
    # Lưu metadata
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n✓ Đã lưu toàn bộ dữ liệu!")
    print(f"✓ Metadata: {METADATA_FILE}")
    print(f"✓ Tổng số sinh viên: {len(metadata['uniforms'])}\n")

def view_dataset():
    """Xem danh sách sinh viên đã train"""
    if not os.path.exists(METADATA_FILE):
        print("⚠ Chưa có dữ liệu!")
        return
    
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print("\n=== DANH SÁCH SINH VIÊN ĐÃ TRAIN ===\n")
    for name, uniform in metadata['uniforms'].items():
        person_dir = os.path.join(DATASET_DIR, name)
        img_count = len(os.listdir(person_dir)) if os.path.exists(person_dir) else 0
        print(f"• {name} - Đồng phục: {uniform} - Số ảnh: {img_count}")
    print()

def delete_person():
    """Xóa một sinh viên khỏi dataset"""
    view_dataset()
    name = input("\nNhập tên sinh viên cần xóa: ").strip()
    
    if not name:
        return
    
    person_dir = os.path.join(DATASET_DIR, name)
    if os.path.exists(person_dir):
        import shutil
        shutil.rmtree(person_dir)
        print(f"✓ Đã xóa thư mục {name}")
    
    # Xóa khỏi metadata
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        if name in metadata['uniforms']:
            del metadata['uniforms'][name]
            with open(METADATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            print(f"✓ Đã xóa {name} khỏi metadata")

if __name__ == "__main__":
    while True:
        print("\n=== QUẢN LÝ DATASET SINH VIÊN ===")
        print("1. Train khuôn mặt mới")
        print("2. Xem danh sách")
        print("3. Xóa sinh viên")
        print("4. Thoát")
        
        choice = input("\nChọn: ").strip()
        
        if choice == "1":
            create_dataset()
        elif choice == "2":
            view_dataset()
        elif choice == "3":
            delete_person()
        elif choice == "4":
            break
        else:
            print("⚠ Lựa chọn không hợp lệ!")
