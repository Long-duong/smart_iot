import cv2
import numpy as np
import time
import os
import json
import threading
import requests
from datetime import datetime
from flask import Flask, jsonify
from flask_socketio import SocketIO

# ================= CONFIG =================
DATASET_DIR = "faces_db"
YUNET_MODEL = "face_detection_yunet_2023mar.onnx"

ABSENT_THRESHOLD = 1
TEMP_THRESHOLD = 30

ESP_IP = "192.168.1.100"
ESP_USER = "admin"
ESP_PASS = "1234"
# =========================================

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
monitor = None

# ============ ESP =========================
class ESP8266Controller:
    def __init__(self):
        self.auth = (ESP_USER, ESP_PASS)

    def led(self, red=False, yellow=False):
        try:
            requests.post(
                f"http://{ESP_IP}/led",
                json={"red": red, "yellow": yellow},
                auth=self.auth,
                timeout=1
            )
        except:
            pass

    def temp_humidity(self):
        try:
            r = requests.get(f"http://{ESP_IP}/dht11", auth=self.auth, timeout=2)
            j = r.json()
            return j.get("temp"), j.get("humidity")
        except:
            return None, None

# ============ SMART CLASS =================
class SmartMonitor:
    def __init__(self):
        print("▶ SMART CLASSROOM – FULL FEATURE (NO MediaPipe)")

        # CAMERA FIX
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # WINDOW FIX
        cv2.namedWindow("Smart Classroom – Full", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Smart Classroom – Full", 1280, 720)

        # YuNet
        self.detector = cv2.FaceDetectorYN.create(
            YUNET_MODEL, "", (320, 320),
            score_threshold=0.7,
            nms_threshold=0.3
        )

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.labels = {}
        self.uniforms = {}

        self.load_faces()

        self.esp = ESP8266Controller()
        self.violations = {}
        self.absent_warned = False

        self.stats = {
            "present": [],
            "absent": [],
            "violations": {},
            "temp": None,
            "humidity": None,
            "time": ""
        }

    # ============ LOAD DATA =================
    def load_faces(self):
        faces, ids = [], []
        idx = 0

        meta = os.path.join(DATASET_DIR, "metadata.json")
        if os.path.exists(meta):
            with open(meta, "r", encoding="utf-8") as f:
                self.uniforms = json.load(f).get("uniforms", {})

        for name in os.listdir(DATASET_DIR):
            p = os.path.join(DATASET_DIR, name)
            if not os.path.isdir(p):
                continue

            self.labels[idx] = name
            for img in os.listdir(p):
                g = cv2.imread(os.path.join(p, img), 0)
                if g is not None:
                    faces.append(g)
                    ids.append(idx)
            idx += 1

        if faces:
            self.recognizer.train(faces, np.array(ids))
            print(f"✓ Load {len(self.labels)} sinh viên")

    def recognize(self, gray, box):
        x, y, w, h = box
        roi = gray[y:y+h, x:x+w]
        try:
            label, conf = self.recognizer.predict(roi)
            if conf < 85:
                return self.labels[label]
        except:
            pass
        return "Unknown"

    # ============ UNIFORM ===================
    def check_uniform(self, frame, box):
        x, y, w, h = box
        roi = frame[y+h:y+h+60, x:x+w]
        if roi.size == 0:
            return "unknown"

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, (0, 0, 200), (180, 40, 255))

        return "white" if cv2.countNonZero(white) / roi.size > 0.3 else "other"

    # ============ HEURISTIC =================
    def turning_head(self, w, h):
        r = w / h
        return r < 0.75 or r > 1.3

    def sleeping(self, y, frame_h):
        return y > frame_h * 0.6

    # ============ REPORT ====================
    def report(self, name, msg):
        if name == "Unknown":
            return

        if name not in self.violations:
            self.violations[name] = []

        if msg not in self.violations[name]:
            t = datetime.now().strftime("%H:%M:%S")
            self.violations[name].append(msg)
            print(f"⚠ [{t}] {name} - {msg}")

            socketio.emit("violation", {
                "name": name,
                "type": msg,
                "time": t
            })

            if "GIAN LẬN" in msg:
                self.esp.led(red=True)
                threading.Timer(3, lambda: self.esp.led()).start()

    # ============ MAIN LOOP =================
    def run(self):
        last_temp = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = frame.shape[:2]

            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(frame)

            present = []

            if faces is not None:
                for f in faces:
                    x, y, bw, bh = map(int, f[:4])
                    name = self.recognize(gray, (x, y, bw, bh))

                    if name != "Unknown":
                        present.append(name)

                        if self.turning_head(bw, bh):
                            self.report(name, "GIAN LẬN (Quay đầu)")

                        if self.sleeping(y, h):
                            self.report(name, "NGỦ GẬT")

                        if self.check_uniform(frame, (x, y, bw, bh)) != self.uniforms.get(name, "white"):
                            self.report(name, "SAI ĐỒNG PHỤC")

                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x+bw, y+bh), color, 2)
                    cv2.putText(frame, name, (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            absent = list(set(self.labels.values()) - set(present))
            if len(absent) >= ABSENT_THRESHOLD and not self.absent_warned:
                print("⚠ VẮNG MẶT:", ", ".join(absent))
                self.absent_warned = True

            if time.time() - last_temp > 5:
                t, hmd = self.esp.temp_humidity()
                self.stats["temp"] = t
                self.stats["humidity"] = hmd
                if t and t > TEMP_THRESHOLD:
                    self.esp.led(yellow=True)
                last_temp = time.time()

            self.stats.update({
                "present": present,
                "absent": absent,
                "violations": self.violations,
                "time": datetime.now().isoformat()
            })

            cv2.imshow("Smart Classroom – Full", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()

# ============ WEB =========================
@app.route("/api/stats")
def api_stats():
    return jsonify(monitor.stats if monitor else {})

@app.route("/api/violations")
def api_violations():
    return jsonify(monitor.violations if monitor else {})

def start():
    global monitor
    monitor = SmartMonitor()
    monitor.run()

if __name__ == "__main__":
    threading.Thread(target=start, daemon=True).start()
    socketio.run(app, host="0.0.0.0", port=5000)
