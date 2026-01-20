🤖 SmartVision AI
Intelligent Multi-Class Object Recognition System

SmartVision AI is an end-to-end Computer Vision application that combines image classification using multiple deep learning models and object detection using YOLOv8.
The system is designed to demonstrate model comparison, real-world inference, and deployment-ready architecture through a clean and professional Streamlit web interface.

📌 Project Overview

Object recognition is a core problem in computer vision with applications across smart cities, retail, security, healthcare, agriculture, and automation.
This project addresses the challenge by building a robust, scalable, and modular vision system capable of:

Classifying images into predefined object categories

Detecting and localizing multiple objects in a single image

Comparing multiple deep learning models

Providing a user-friendly web interface for inference

🎯 Key Features
🖼️ Image Classification

Supports 4 CNN architectures:

VGG16

ResNet50

MobileNetV2

EfficientNetB0

Uses transfer learning with ImageNet pre-trained weights

Displays Top-K predictions with confidence scores

Enables model comparison on the same input image

🎯 Object Detection

Uses YOLOv8 for real-time object detection

Detects multiple objects per image

Displays bounding boxes, class labels, and confidence scores

🖥️ Web Application

Built using Streamlit

Professional, clean UI suitable for production demos

Modular pages:

Home

Classification

Detection

🧠 Models Used
Task	Model	Purpose
Classification	VGG16	Baseline CNN
Classification	ResNet50	Deep residual learning
Classification	MobileNetV2	Lightweight & fast
Classification	EfficientNetB0	Best accuracy-speed balance
Detection	YOLOv8	Real-time multi-object detection
📊 Dataset
Classification Dataset

Cropped object images extracted from a curated COCO subset

Images resized to 224×224

Balanced class distribution

Detection Dataset

YOLO-format dataset with:

images/train, images/val

labels/train, labels/val

Bounding boxes in YOLO normalized format

data.yaml configuration for YOLO training

📁 Project Structure
SmartVisionAI/
│
├── app/
│   └── app.py                  # Streamlit application
│
├── Classification Models Training/
│   └── models/
│       ├── VGG16_best.h5
│       ├── ResNet50_best.keras
│       ├── MobileNetV2_best.keras
│       ├── EfficientNetB0_best.keras
│       ├── yolov8n.pt
│       └── classes.txt
│
├── smartvision_dataset/
│   ├── classification/
│   └── detection/
│
├── requirements.txt
└── README.md

📦 Requirements
streamlit
tensorflow
numpy
pillow
opencv-python
ultralytics

🧪 How It Works
🔹 Classification Pipeline

User uploads an image

Image is preprocessed (resize + ImageNet normalization)

Selected CNN model predicts class probabilities

Top predictions are displayed with confidence

🔹 Detection Pipeline

User uploads an image

YOLOv8 detects multiple objects

Bounding boxes and labels are drawn

Results are displayed with confidence scores

❓ Why Do Predictions Differ Across Models?

Each CNN architecture learns features differently:

Deeper models generalize better

Lightweight models prioritize speed

Prediction variation is expected and acceptable

This project focuses on model comparison, not forcing identical outputs.

🚀 Deployment

The application is deployment-ready and can be hosted on:

Hugging Face Spaces

Local server

Cloud VM

Streamlit ensures fast prototyping and easy sharing.

🏆 Learning Outcomes

Deep Learning with CNNs

Transfer Learning techniques

Object Detection using YOLOv8

Model comparison and evaluation

Dataset preparation in YOLO format

Streamlit application development

Production-style project structuring

👨‍💻 Author

Elansurya K
AI & Data Science Enthusiast
📍 India

📌 Conclusion

SmartVision AI demonstrates a complete computer vision workflow from dataset preparation and model training to inference and deployment.
The project highlights real-world challenges, model trade-offs, and practical deployment considerations, making it suitable for academic evaluation and industry demonstration.
