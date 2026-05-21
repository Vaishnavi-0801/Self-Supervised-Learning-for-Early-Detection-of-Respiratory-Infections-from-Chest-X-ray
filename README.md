# Self-Supervised Learning for Early Detection of Respiratory Infections from Chest X-ray Images

This project presents an AI-assisted medical imaging framework designed for automated chest X-ray analysis using Self-Supervised Learning (SSL) and Deep Learning techniques. The system focuses on the early detection and visualization of respiratory abnormalities from chest radiographs while reducing dependency on heavily labeled medical datasets.

The framework integrates a pretrained ResNet-18 based self-supervised feature extraction model with an interactive web-based dashboard for real-time medical image analysis, abnormality localization, visualization, and automated report generation. Unlike traditional supervised diagnostic systems, the proposed approach learns meaningful radiographic representations from unlabeled chest X-ray images, enabling scalable and resource-efficient medical image processing.

The system performs multiple stages of intelligent analysis including image preprocessing, deep feature extraction, PCA-based feature clustering, heatmap generation, abnormality scoring, and explainable visualization. OpenCV-based image enhancement and heatmap overlays are used to highlight suspicious pulmonary regions, improving interpretability and assisting preliminary radiographic assessment.

An interactive Streamlit dashboard is developed to provide an intuitive user experience for healthcare assistance applications. Users can upload chest X-ray images, visualize abnormality heatmaps, analyze feature-space distributions, manage patient records, and generate downloadable PDF reports automatically. The backend architecture integrates PyTorch, OpenCV, NumPy, Scikit-learn, and ReportLab to support efficient inference, visualization, and reporting workflows.

Key Features:
- Self-Supervised Deep Feature Extraction using ResNet-18
- Automated Chest X-ray Abnormality Analysis
- Heatmap-Based Explainable AI Visualization
- PCA-Based Feature Clustering and Representation Analysis
- Automated Clinical PDF Report Generation
- Streamlit-Based Interactive Dashboard
- Lightweight and Scalable Medical Imaging Framework

This project demonstrates the potential of Self-Supervised Learning for intelligent healthcare systems and clinical decision-support applications by combining deep learning, explainable AI, visualization, and automated medical reporting into a unified platform.

Developed as a research-oriented AI healthcare project for intelligent radiographic image analysis and respiratory infection screening.
