# 🚀 Quick Setup Instructions

## 📁 File Structure

Create this folder structure on your computer:

```
ai-detector-advanced/
├── app.py
├── requirements.txt
├── README.md
├── SETUP_INSTRUCTIONS.txt
└── .streamlit/
    └── config.toml
```

## 📝 Step-by-Step Setup

### 1. Create Main Folder

```bash
mkdir ai-detector-advanced
cd ai-detector-advanced
```

### 2. Save the Files

- Copy the `app.py` code from artifact 1
- Copy the `requirements.txt` code from artifact 2
- Copy the `README.md` code from artifact 3
- Create folder `.streamlit` and save `config.toml` from artifact 4

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

## 🌐 For Streamlit Cloud Deployment

### 1. Upload to GitHub

- Create a new GitHub repository
- Upload all these files to the repository

### 2. Deploy on Streamlit Cloud

- Go to https://share.streamlit.io
- Connect your GitHub account
- Select your repository
- Choose `app.py` as the main file
- Click “Deploy”

## 📱 For iPhone/Mobile Use

### Option 1: Use Streamlit Cloud (Recommended)

1. Deploy to Streamlit Cloud (steps above)
1. Get the public URL (like https://yourapp.streamlit.app)
1. Open the URL on your iPhone browser
1. Add to home screen for easy access

### Option 2: Local Network

1. Run locally with `streamlit run app.py`
1. Note the network URL (usually shows in terminal)
1. Open that URL on your phone (must be on same WiFi)

## 🔧 Troubleshooting

**Common Issues:**

- **Import errors**: Run `pip install -r requirements.txt` again
- **Port busy**: Change port in config.toml or restart terminal
- **Memory issues**: Close other programs, use smaller images
- **Slow processing**: Normal for large images, wait 30-60 seconds

**Performance Tips:**

- Images 500x500 to 1500x1500 pixels work best
- JPG format is fastest to process
- Close other browser tabs for more memory

## ✅ You’re Ready!

Once running, upload any image and get detailed AI detection analysis!

The app will show:

- 🤖 AI Generated / 📸 Real Image / ❓ Uncertain
- Confidence level (High/Medium/Low)
- Detailed technical metrics
- Visual analysis charts
- Interpretation guide