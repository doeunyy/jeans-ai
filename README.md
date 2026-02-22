# 👖 Jeans: AI-Based Photo Editing and Sharing Service for Seniors

### 🏆 **Awards**
- **Personal Excellence Award** — SK Telecom FLY AI Challenger Program  
- **Project Excellence Award** — SK Telecom FLY AI Challenger Program

> ### **TL;DR**
> - Built an **AI-powered photo editing and sharing service for seniors** to improve digital accessibility
> - Led the entire **AI pipeline as the sole AI engineer**, from data collection to deployment
> - Fine-tuned **Whisper for Korean dialect speech**, integrated **YOLO + FaceNet** for face enhancement
> - Deployed a **production-ready FastAPI inference server on AWS** with CI/CD automation

<br>

<img width="2667" height="1500" alt="열정4팀_최종발표-01" src="https://github.com/user-attachments/assets/cd558ebd-a16a-4d89-8687-56213633f301" />

<br>

This repository contains **the AI component of *Jeans***, an AI-powered photo editing and sharing service designed to increase digital accessibility and social participation among seniors.
I served as the **sole AI engineer**, responsible for designing and implementing the **end-to-end machine learning pipeline**, from data preparation and model fine-tuning to API design and cloud deployment.

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Implementation Details](#implementation-details)
5. [Model Training & Deployment](#model-training--deployment)
6. [Impact](#impact)
7. [Team](#team)

---

## Overview

**Jeans** is a senior-friendly photo editing and sharing application powered by multimodal AI. The service enables seniors to edit photos using **voice commands**, automatically detect and enhance faces, and share images easily with family and friends.

The goal of the project was to help older adults overcome technological barriers and participate more actively in digital communication.

I led all AI development, including data preparation, model fine-tuning, API design, and deployment using AWS.

## Key Features

<img width="2667" height="1500" alt="열정4팀_최종발표-08" src="https://github.com/user-attachments/assets/e4a3fd38-1c87-42f3-bbc8-b6fa2f21972f" />


* **Voice-driven photo editing** using Whisper fine-tuned on Korean regional dialect speech
* **Face detection & enhancement** using YOLO and FaceNet
* **Automated photo captioning & tagging** for accessibility
* **FastAPI backend** with scalable REST APIs
* **Real-time inference server** deployed on AWS EC2
* **CI/CD automation** using GitHub Actions
* **User-centered design** based on interviews and usability testing with seniors 

## **System Architecture**

<img width="2667" height="1500" alt="열정4팀_최종발표-28" src="https://github.com/user-attachments/assets/24a72fbf-16f6-434c-a2fd-f71bd162cab4" />

```
User Voice Input
        ↓
Whisper (Fine-tuned) — Korean Dialect ASR
        ↓
Command Parsing & Image Processing Logic
        ↓
YOLO / FaceNet — Face Detection, Cropping, Enhancement
        ↓
FastAPI Inference Server
        ↓
AWS EC2 Deployment + CI/CD
        ↓
Mobile Frontend (Photo Editing / Sharing)
```

This end-to-end flow enables seniors to issue simple voice commands and receive enhanced images instantly.


## Implementation Details

### Dataset Construction

* Collected and cleaned Korean dialect audio dataset
* Labeled command-specific phrases to enable action-triggering
* Applied audio preprocessing (normalization, trimming, silence removal)

### Models

* Whisper fine-tuning
  * Trained on senior & dialectal Korean speech
  * Improved recognition accuracy for age-affected pronunciation
 
* YOLO + FaceNet integration
  * YOLO for coarse face detection
  * FaceNet for identity-level refinement or alignment

* Pipeline logic
  * Map recognized text → edit operations (e.g., crop, brighten, sharpen)

### Backend (FastAPI)

* Designed REST endpoints for image upload, transformation, and retrieval
* Handled model loading, GPU utilization, and async batching
* Built a modular router structure for future expansion


## Model Training & Deployment

* Training
  * PyTorch-based training pipelines
  * Custom DataLoader for speech–command pairs
  * Evaluation pipeline for WER and command classification accuracy

* Deployment
  * AWS EC2 GPU instance
  * FastAPI inference server
  * Nginx + SSL for production deployment

* CI/CD
  * GitHub Actions for automated testing, formatting, and deployment
  * Containerized build process for reproducibility


## Impact

* Conducted **interviews and usability sessions with seniors** to validate real-world needs 
* Improved accessibility by tailoring voice commands to speech patterns common among older adults
* Provided a fully functional prototype with measurable improvements in user satisfaction
* Strengthened expertise in **AI deployment, REST API design, multimodal integration, and end-to-end MLOps**

## **Team**

* **Doeun Kim**: PM, AI Lead - Sole AI Engineer (model development, dataset construction, API design, deployment)
* Jihye Yoo: Frontend Lead
* Junyong Lee: Frontend Engineer
* Soyeon Cha: Backend Lead
* Boseok Park: Backend Engineer
