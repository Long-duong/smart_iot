import cv2
import os

# ================= CONFIG =================
DATASET_DIR = "faces_db"
IMG_SIZE = 200
# =========================================


def main():
    print("▶ TRAIN KHUÔN MẶT SINH VIÊN")

    name = input("Nhập tên sinh viên (vd: sv01, NguyenVanA): ").strip()
    if not name:
        print("❌ Tên không hợp lệ")
        return

    save_path = os.path.join(DATASET_DIR, name)
    os.makedirs(save_path, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)
    count = 0

    print("📸 Nhấn SPACE để chụp | Q để thoát")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(80, 80)
        )

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(frame, f"Captured: {count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("TRAIN FACE", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if len(faces) == 1:
                cv2.imwrite(f"{save_path}/{count}.jpg", face_img)
                count += 1
                print(f"✓ Da luu anh {count}")
            else:
                print("⚠️ Chỉ đứng 1 người trước camera")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✓ Hoàn thành train cho {name} ({count} ảnh)")


if __name__ == "__main__":
    main()
