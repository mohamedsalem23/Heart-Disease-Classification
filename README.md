# ❤️ Heart Disease Classifier - AI System

**Doctor**: Dr. Mohammad El Tahlawi

## Overview
Multi-Modal Heart Disease Classification System using AI (60% visual + 40% clinical features)

---

## 🚀 Deployment Steps

### Option 1: Deploy to Streamlit Cloud (Recommended)

1. **Push code to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/Heart-project.git
   git push -u origin main
   ```

2. **Go to Streamlit Cloud**
   - Visit: https://streamlit.io/cloud
   - Sign in with GitHub
   - Click "New app"
   - Select your repository: `Heart-project`
   - Choose branch: `main`
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Configure Secrets** (in Streamlit Cloud dashboard)
   - Go to App settings → Secrets
   - Add your environment variables from `.env`

### Option 2: Manual Deployment (Using Docker)

1. **Create Dockerfile**
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   EXPOSE 8501
   CMD ["streamlit", "run", "app.py"]
   ```

2. **Build and run**
   ```bash
   docker build -t heart-classifier .
   docker run -p 8501:8501 heart-classifier
   ```

### Option 3: Deploy to Heroku

1. **Create Procfile**
   ```
   web: streamlit run app.py --server.port=$PORT
   ```

2. **Deploy**
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

---

## 📋 Files Structure

```
Heart-project/
├── app.py                          # Main Streamlit app
├── classifier.py                   # ML classification logic
├── config.py                       # Configuration settings
├── data_processor.py              # Image processing
├── db_manager.py                  # Database management
├── heart_embeddings_swin.db       # SQLite database
├── requirements.txt               # Python dependencies
├── .streamlit/config.toml         # Streamlit config
├── .env                           # Environment variables
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

---

## ⚙️ Local Development

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run locally**
   ```bash
   streamlit run app.py
   ```

3. **Access**
   - Local: http://localhost:8501
   - Network: http://192.168.1.13:8501

---

## 🔧 Features

- **Multi-Modal Classification**: Combines visual analysis (60%) + clinical data (40%)
- **Real-time Predictions**: Instant classification of heart conditions
- **Arabic UI**: User-friendly interface in Arabic
- **4 Classification Levels**: Normal, Grade 1, Grade 2, Grade 3
- **High Accuracy**: Trained on 683 samples with multiple features

---

## 📊 Model Details

- **Architecture**: Swin Transformer + SVM Ensemble
- **Visual Model**: Microsoft Swin Transformer (768-dim embeddings)
- **Clinical Features**: Age, BSA, BMI, LAV, LAVI, Hypertension, Smoking
- **Training Data**: 713 echo images with clinical data
- **Accuracy**: ~85-90% (based on validation set)

---

## ⚠️ Medical Disclaimer

This system is **for research and educational purposes only**.
It should NOT be used for actual medical diagnosis.
Always consult with qualified medical professionals.

---

## 📞 Support

For issues or questions, contact Dr. Mohammad El Tahlawi

**Version**: 1.0.0  
**Last Updated**: December 2025
