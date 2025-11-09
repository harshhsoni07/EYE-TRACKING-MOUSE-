# 👁️ Eye Tracking Mouse

Control your mouse cursor and clicks using just your eyes — powered by **OpenCV**, **MediaPipe**, and **PyAutoGUI**.

---

## 🚀 Features

- **Real-time Eye Tracking** using your webcam  
- **Cursor Movement** controlled by iris position  
- **Left Click** by blinking left eye  
- **Right Click** by blinking right eye  
- **Double Click** by blinking both eyes  
- **Smooth Cursor Movement** with adjustable sensitivity and smoothing  

---

## 🧠 How It Works

- Uses **MediaPipe Face Mesh** to detect 468 facial landmarks.
- Calculates the **Eye Aspect Ratio (EAR)** to detect blinks.
- Maps iris movement to **screen coordinates** using **PyAutoGUI**.
- Blinks are used to trigger mouse clicks:
  - 👁️ Left eye blink → Left Click  
  - 👁️ Right eye blink → Right Click  
  - 👁️👁️ Both eyes blink → Double Click

---

## 🧩 Requirements

- Python 3.7 or above  
- Webcam  

Install dependencies:
```bash
pip install -r requirements.txt
