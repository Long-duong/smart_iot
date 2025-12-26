import cv2
import pickle
import os
import numpy as np

FACE_DATA_FILE = "face_data.pkl"

class FaceTrainer:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_names = {}
        self.face_encodings = []

        self.load_data()

    def load_data(self):
        if os.path.exists(FACE_DATA_FILE):
            with open(FACE_DATA_FILE, "rb") as f:
                data = pickle.load(f)
                self.face_names = data["names"]
                self.face_encodings = data["encodings"]
                if self.face_encodings:
                    labels = list(range(len(self.face_encodings)))
                    self.face_recognizer.train(self.face_encodings, np.array(labels))
            print(f"✓ Đã load {len(self.face_names)} khuôn mặt")

    def save_data(self):
        with open(FACE_DATA_FILE, "wb") as f:
            pickle.dump({
                "names": self.face_names,
                "encodings": self.face_encodings
            }, f)
        print("✓ Đã lưu dữ liệu khuôn mặt")

    def train(self, name):
        print(f"📸 TRAIN: {name}")
        print("SPACE: chụp ảnh | Q: thoát")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

            cv2.imshow("Training", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    face = gray[y:y+h, x:x+w]
                    face = cv2.resize(face, (200, 200))

                    label = len(self.face_names)
                    self.face_names[label] = name
                    self.face_encodings.append(face)

                    labels = list(range(len(self.face_encodings)))
                    self.face_recognizer.train(self.face_encodings, np.array(labels))

                    self.save_data()
                    print("✓ Đã chụp & train")

            elif key == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    name = input("Nhập tên sinh viên: ").strip()
    if name:
        FaceTrainer().train(name)
