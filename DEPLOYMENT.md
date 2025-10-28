# Deployment Guide

## Quick Deployment to Streamlit Cloud

### Step 1: Repository Setup
Your code is already pushed to: [https://github.com/Uday-2611/OFI_Case_Study.git](https://github.com/Uday-2611/OFI_Case_Study.git)

### Step 2: Deploy to Streamlit Cloud
1. Go to [https://share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub account
4. Select repository: `Uday-2611/OFI_Case_Study`
5. Set main file path to: `app.py`
6. Click "Deploy!"

### Step 3: Access Your App
Your app will be live at: `https://your-app-name.streamlit.app`

## Local Testing
Before deploying, test locally:
```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python -m streamlit run app.py --server.port 8501
```

## Project Structure
```
OFI_Case_Study/
├── app.py                          # Main Streamlit application
├── data_processor.py               # Data processing
├── ml_models.py                    # ML models
├── visualizations.py              # Charts
├── requirements.txt               # Dependencies
├── run_app.bat                    # Windows startup script
└── Case study internship data/    # Data files
```

## Features
- ✅ Predictive delay forecasting (75%+ accuracy)
- ✅ Real-time risk assessment
- ✅ Interactive dashboard (6 modules)
- ✅ Machine learning models
- ✅ Professional dark theme UI
