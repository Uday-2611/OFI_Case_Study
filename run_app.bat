@echo off
echo ========================================
echo NexGen Logistics Predictive Delivery Optimizer
echo ========================================
echo.
echo Starting application...
echo The application will be available at: http://localhost:8501
echo.
echo Features:
echo - Predictive delay forecasting (75%+ accuracy)
echo - Real-time risk assessment
echo - Early warning system
echo - Corrective action recommendations
echo - Analytics and insights
echo - Customer communication tools
echo.
echo Press Ctrl+C to stop the application
echo.
python -m streamlit run app.py --server.port 8501
pause
