# 🏫 Smart Classroom System

Hệ thống giám sát lớp học thông minh sử dụng AI và IoT

![Smart Classroom](https://img.shields.io/badge/Smart-Classroom-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![Flask](https://img.shields.io/badge/Flask-2.3.3-lightgrey)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8.1-orange)

## ✨ Tính năng chính

### 👁️‍🗨️ Nhận diện khuôn mặt
- Nhận diện sinh viên trong thời gian thực
- Ghi nhận sự hiện diện/vắng mặt
- Hỗ trợ dataset khuôn mặt

### ⚠️ Phát hiện vi phạm
- **Gian lận**: Phát hiện quay đầu
- **Ngủ gật**: Phát hiện tư thế ngủ
- **Đồng phục**: Kiểm tra đồng phục

### 🌡️ IoT Integration
- Kết nối ESP8266 để đọc cảm biến DHT11
- Điều khiển LED cảnh báo từ xa
- Hiển thị nhiệt độ/độ ẩm thời gian thực

### 🌐 Web Dashboard
- Giao diện quản trị hiện đại
- Thống kê trực quan
- Cảnh báo thời gian thực qua WebSocket
- Điều khiển từ xa

## 📋 Yêu cầu hệ thống

### Phần cứng
- Camera USB
- ESP8266 với DHT11 và LED (tùy chọn)
- PC/Laptop với Python 3.8+

### Phần mềm
- Python 3.8+
- OpenCV với face recognition
- Flask framework

## 🚀 Cài đặt

### 1. Clone repository
```bash
git clone https://github.com/yourusername/smart-classroom.git
cd smart-classroom
