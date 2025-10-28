import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

class DelayPredictor:
    def __init__(self):
        self.classifier = None
        self.regressor = None
        self.label_encoder = LabelEncoder()
        self.feature_importance = None
        self.accuracy = 0
        
    def train_models(self, X, y):
        """Train both classification and regression models"""
        print("Training ML models...")
        
        # Encode target variable
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Train Random Forest Classifier
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.classifier.fit(X_train, y_train)
        
        # Train XGBoost Regressor for delay days prediction
        self.regressor = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        # For regression, use actual delay days from the original dataframe
        # We need to get delay_days from the original data, not from X
        # This will be handled in the main app
        y_delay_days = np.zeros(len(X))  # Placeholder for now
        
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            X, y_delay_days, test_size=0.2, random_state=42
        )
        
        self.regressor.fit(X_train_reg, y_train_reg)
        
        # Evaluate models
        y_pred = self.classifier.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Classification Accuracy: {self.accuracy:.3f}")
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return self.accuracy
    
    def predict_delay_category(self, X):
        """Predict delay category for new orders"""
        if self.classifier is None:
            raise ValueError("Model not trained yet")
        
        # Handle missing values
        X_processed = X.fillna(X.mean())
        
        # Predict
        predictions = self.classifier.predict(X_processed)
        probabilities = self.classifier.predict_proba(X_processed)
        
        # Convert back to original labels
        predicted_categories = self.label_encoder.inverse_transform(predictions)
        
        return predicted_categories, probabilities
    
    def predict_delay_days(self, X):
        """Predict actual delay days"""
        if self.regressor is None:
            raise ValueError("Model not trained yet")
        
        # Handle missing values
        X_processed = X.fillna(X.mean())
        
        # Predict
        delay_days = self.regressor.predict(X_processed)
        
        return delay_days
    
    def get_risk_score(self, X):
        """Calculate risk score (0-100) for orders"""
        if self.classifier is None:
            raise ValueError("Model not trained yet")
        
        # Handle missing values
        X_processed = X.fillna(X.mean())
        
        # Get probabilities
        probabilities = self.classifier.predict_proba(X_processed)
        
        # Calculate risk score based on probability of delay
        risk_scores = []
        for prob in probabilities:
            # Weight: On-Time=0, Slightly-Delayed=50, Severely-Delayed=100
            if len(prob) >= 3:
                risk = prob[1] * 50 + prob[2] * 100  # Assuming order: On-Time, Slightly, Severely
            else:
                risk = prob[1] * 100  # If only 2 classes
            risk_scores.append(min(risk, 100))
        
        return np.array(risk_scores)
    
    def get_feature_importance(self):
        """Get feature importance for analysis"""
        return self.feature_importance
    
    def save_models(self, filepath_prefix='models/'):
        """Save trained models"""
        import os
        os.makedirs(filepath_prefix, exist_ok=True)
        
        joblib.dump(self.classifier, f'{filepath_prefix}classifier.pkl')
        joblib.dump(self.regressor, f'{filepath_prefix}regressor.pkl')
        joblib.dump(self.label_encoder, f'{filepath_prefix}label_encoder.pkl')
        
    def load_models(self, filepath_prefix='models/'):
        """Load pre-trained models"""
        self.classifier = joblib.load(f'{filepath_prefix}classifier.pkl')
        self.regressor = joblib.load(f'{filepath_prefix}regressor.pkl')
        self.label_encoder = joblib.load(f'{filepath_prefix}label_encoder.pkl')
