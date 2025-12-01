# IMAGE-FORGERY-RECOGNITION-USING-DEEP-LEARNING-TECHNIQUES
A DeepFake Detection System using Xception CNN + Raspberry Pi Deployment

🧩 Overview

This project implements an end-to-end DeepFake detection system using Deep Learning. Built using Xception CNN, trained on the FaceForensics++ (FF++) dataset, and deployed on a Raspberry Pi 4, the system can classify faces as REAL or FAKE in images and videos.


The project includes a complete pipeline:

✔ Dataset organization

✔ Frame extraction

✔ Face detection & cropping

✔ Deep learning model training

✔ Real-time inference on videos

✔ Hardware deployment on Raspberry Pi


🚀 Features

Xception-based binary classifier (REAL vs FAKE)

Separable Convolutions → 2.7× faster than standard convs

Automated preprocessing pipeline:

Organizes raw dataset

Extracts frames at intervals

Detects & crops faces with Dlib

High accuracy & AUC score of ~0.878

Fully deployable on Raspberry Pi 4 for low-power, real-time detection

Video inference script overlays prediction labels on frames

Clean and modular Python scripts for easy reuse
