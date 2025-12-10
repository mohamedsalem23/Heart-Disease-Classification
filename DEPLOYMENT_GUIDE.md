# 🚀 Streamlit Cloud Deployment Checklist

## Step-by-Step Guide

### 1. Prepare Your Code ✅
- [x] All necessary files in folder
- [x] requirements.txt updated
- [x] .gitignore created
- [x] README.md ready

### 2. Create GitHub Repository
```bash
cd D:\Heart-project
git init
git add .
git commit -m "Heart Disease Classifier - Initial Release"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/heart-disease-classifier.git
git push -u origin main
```

### 3. Deploy to Streamlit Cloud

1. **Go to**: https://streamlit.io/cloud

2. **Click**: "New app" button

3. **Fill in the form**:
   - GitHub account: Select your account
   - Repository: `heart-disease-classifier`
   - Branch: `main`
   - File path: `app.py`

4. **Click**: "Deploy"

5. **Wait** for deployment (usually 2-3 minutes)

### 4. Configure Secrets (if needed)

If your app needs secrets:
1. Go to: **Advanced settings** → **Secrets**
2. Add your `.env` variables in TOML format:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```

### 5. Share Your App

Your app will be available at:
```
https://[YOUR-USERNAME]-heart-disease-classifier.streamlit.app
```

---

## Important Notes

⚠️ **Database Size**: The SQLite database is ~713MB
- Streamlit Cloud has file size limits
- Consider uploading to cloud storage (AWS S3, Google Cloud) if needed

⚠️ **Model Loading**: First load may take 1-2 minutes
- Model is cached after first load
- Subsequent loads are fast

⚠️ **Memory Usage**: The app requires ~4GB RAM
- Streamlit Cloud provides 1GB by default
- May need Streamlit Cloud premium for larger models

---

## Alternative: Self-Hosted Deployment

### Using Railway.app
1. Connect GitHub account
2. Create new project
3. Select repository
4. Deploy

### Using Render
1. Go to render.com
2. Create new Web Service
3. Connect GitHub
4. Deploy

### Using Heroku
```bash
heroku create heart-classifier
git push heroku main
```

---

## Monitoring & Maintenance

- Check app status: Dashboard → App info
- View logs: Dashboard → Logs
- Restart app: Dashboard → Reboot
- Update code: Push to GitHub (auto-deploys)

---

**Ready to deploy? Follow the steps above!** 🎉
