# BLIND-ASSISTANT
Vision Assist for the Blind
Overview
Vision Assist for the Blind is a web-based application designed to assist visually impaired users in navigating their environment. It uses real-time object detection and voice feedback to identify objects in the user’s surroundings, such as books, cups, or chairs, and announces them via text-to-speech. The app leverages Flask for video streaming, YOLOv8 (yolov8n.pt) for object detection, and the Web Speech API for voice output.
Features

Live Video Streaming: Streams webcam feed to the browser.
Object Detection: Uses YOLOv8 (yolov8n.pt) to detect 80 COCO classes (e.g., person, book, cup, chair).
Voice Feedback: Announces detected objects using Web Speech API.
Network Accessibility: Hosted on 0.0.0.0:5000 for access from other devices on the network.
Custom Object Support: Can be fine-tuned to detect custom objects like pen, pencil, speaker, and door.

Prerequisites

Operating System: Windows (due to DirectShow backend in OpenCV).
Python: Version 3.7 or higher.
Webcam: A functioning webcam for video input.
Browser: Chrome, Edge, or Firefox (for Web Speech API support).
Git: To clone the repository (optional).



Set Up a Virtual Environment:
python -m venv venv
.\venv\Scripts\activate  # On Windows


Install Dependencies:Ensure you have a requirements.txt file with the following:
Flask==2.0.1
opencv-python==4.6.0.66
numpy==1.23.5
Werkzeug==2.0.3
ultralytics==8.0.196
torch==2.0.1
torchvision==0.15.2

Then install:
pip install -r requirements.txt


Place the YOLOv8 Model:

Ensure yolov8n.pt is in the project root directory (vision-assist/yolov8n.pt).
If you don’t have yolov8n.pt, download it using:python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"




Run the Application:
python app.py


The app will start on http://127.0.0.1:5000.
To access from another device on the same network, use your machine’s local IP (e.g., http://192.168.1.100:5000).



File Structure
vision-assist/
├── app.py                  # Flask app with video streaming and YOLOv8 object detection
├── yolov8n.pt             # Pretrained YOLOv8 model (place it here)
├── templates/
│   └── index.html         # Frontend UI, fetches detection results, and handles voice output
├── static/
│   └── css/
│       └── styles.css     # Optional CSS for styling (if used)
├── logs/
│   └── app.log            # Log file for debugging
├── requirements.txt       # Python dependencies
├── data.yaml              # YOLOv8 dataset configuration for fine-tuning
├── capture_images.py      # Script for capturing training images (if used)
├── dataset/
│   ├── train/
│   │   ├── images/        # Training images for custom objects
│   │   └── labels/        # Training labels for custom objects
│   ├── val/
│   │   ├── images/        # Validation images for custom objects
│   │   └── labels/        # Validation labels for custom objects
├── runs/
│   └── detect/
│       └── train/
│           └── weights/
│               └── best.pt # Fine-tuned YOLOv8 model (after training)
└── venv/                  # Virtual environment

Usage

Access the App:

Open http://127.0.0.1:5000 in your browser.
Or use your local IP (e.g., http://192.168.1.100:5000) from another device.


Interact with the App:

On page load, the app attempts to play a welcome message (“Hi, welcome to Visual Assistant, let’s go”). If it doesn’t play due to browser autoplay restrictions, click anywhere on the screen.
Click again to start object detection.
The app will detect objects in the webcam feed and announce them (e.g., “Detected: book, chair”).


Test Object Detection:

Hold objects like a book, cup, or chair in front of the webcam (these are part of the 80 COCO classes yolov8n.pt recognizes).
Check the browser console (F12 > Console) for logs or errors.



Fine-Tuning for Custom Objects
The pretrained yolov8n.pt model recognizes 80 COCO classes but not custom objects like pen, pencil, speaker, or door. To detect these:

Prepare Your Dataset:

Add images to dataset/train/images and dataset/val/images.
Label them using a tool like labelimg:pip install labelimg
labelimg


Save labels in dataset/train/labels and dataset/val/labels.


Update data.yaml:

Edit data.yaml to include your classes:train: ./dataset/train/images
val: ./dataset/val/images
nc: 4
names: ['pen', 'pencil', 'speaker', 'door']




Fine-Tune the Model:
yolo train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640


This will create a fine-tuned model in runs/detect/train/weights/best.pt.


Update app.py:

Change the model path in app.py to use the fine-tuned model:model = YOLO('runs/detect/train/weights/best.pt')




Test Custom Objects:

Restart the app and test with your custom objects.



Troubleshooting
1. Error: YOLOv8 Model Not Loaded

Cause: The app couldn’t load yolov8n.pt.
Solution:
Check logs/app.log for the error (e.g., “No such file or directory”).
Ensure yolov8n.pt is in the project root (vision-assist/yolov8n.pt).
Verify ultralytics is installed:pip show ultralytics
pip install ultralytics==8.0.196


Test model loading:python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"





2. Welcome Note Not Playing

Cause: Browser autoplay restrictions block the welcome message.
Solution:
Click the screen to play the message.
Test speech in the browser console:testSpeech();


Ensure browser sound is enabled and not muted.



3. Webcam Not Working

Cause: Webcam might be in use by another app or not accessible.
Solution:
Close other apps using the webcam.
Check logs/app.log for errors like “Error: Could not open any webcam”.
Test webcam access:python -c "import cv2; cap = cv2.VideoCapture(0, cv2.CAP_DSHOW); print(cap.isOpened()); cap.release()"





Future Improvements

Custom Object Detection: Fine-tune YOLOv8 to detect more objects specific to the user’s needs.
Depth Estimation: Implement MiDaS for spatial awareness (placeholder in app.py).
Voice Customization: Add options for language and speech speed in index.html.
Improved Audio Interaction: Resolve autoplay issues with better user prompts.

License
This project is licensed under the MIT License. See the LICENSE file for details (if applicable).
Contact
For issues or contributions, please open an issue on GitHub or contact the project maintainer.
