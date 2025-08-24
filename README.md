# Multi-Agent AI Medical Diagnosis System

## Overview
The **Multi-Agent AI Medical Diagnosis System** is designed to assist in medical imaging diagnosis using deep learning and computer vision.  
This project integrates multiple AI agents to analyze, classify, and provide insights into medical images while maintaining a modular backend and interactive frontend.

---

## Features
- Multi-agent system for medical diagnosis  
- Deep Learning (CNN-based models for image classification)  
- Flask backend for API and model inference  
- Streamlit frontend for user interaction  
- Deployment-ready setup for Hugging Face Spaces & Vercel  

---

## Project Structure
```bash
├── app.py                  # Streamlit frontend
├── flask_app.py            # Flask backend API
├── models/                 # Pre-trained and fine-tuned models
├── static/                 # Static files (CSS, JS, images)
├── templates/              # HTML templates for Flask
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## Installation

Clone the repository:
```bash
git clone https://github.com/yourusername/multi-agent-medical-diagnosis.git
cd multi-agent-medical-diagnosis
```

Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate    # For Windows
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Start Flask Backend
```bash
python flask_app.py
```

### 2. Run Streamlit Frontend
```bash
streamlit run app.py
```

---

## Example Workflow
1. Upload a medical image through the Streamlit interface  
2. Image is sent to the Flask backend for preprocessing and inference  
3. AI agents (CNN models, diagnostic rules) analyze the image  
4. Results are displayed with prediction probabilities  

---


## Credits
- **Frontend (Streamlit):** Developed by Ajay Tiwari  
- **Backend (Flask):** Special thanks to my teammate for backend integration  
- **Deployment:** Hugging Face Spaces & Vercel  

---

## License
This project is licensed under the MIT License – feel free to use and improve it.

```bash
MIT License
Copyright (c) 2025 Ajay Tiwari
Permission is hereby granted, free of charge, to any person obtaining a copy of this software...
```
## Author
### Ajay Tiwari
### B.Tech - Computer Science and Engineering (Artificial Intelligence): 2022-26
