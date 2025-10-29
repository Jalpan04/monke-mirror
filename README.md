# monk-mirror

This is a Python application that uses your webcam to detect facial expressions and hand gestures in real-time, displaying a corresponding image "mirror" of your pose.

It's built to be a fun, interactive demo of computer vision and real-time landmark detection.

## ✨ Features

The application tracks your face and hands to recognize several distinct poses:

* **Wink:** Detects a single eye closing.
* **Shocked:** Detects wide-open eyes and an open mouth.
* **Smile / Squint:** Detects when both eyes are squeezed shut.
* **"Idea":** Recognizes the left hand with the index finger pointed up.
* **"Thumbs Up":** Recognizes the right hand in a thumbs-up gesture.
* **Neutral:** Displays a default image when no other specific gesture is held.

## 🛠️ Technology Used

* **Python**
* **OpenCV:** For capturing the webcam feed and handling image display/processing.
* **MediaPipe:** For high-fidelity, real-time face mesh and hand landmark detection.
* **NumPy:** For numerical operations and combining the video/image frames.
