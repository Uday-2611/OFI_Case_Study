import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def load_data(self):
        """Load all CSV files and merge them"""
        print("Loading data files...")
        
        # Load all datasets
        orders = pd.read_csv('Case study internship data/orders.csv')
        delivery_perf = pd.read_csv('Case study internship data/delivery_performance.csv')
        routes = pd.read_csv('Case study internship data/routes_distance.csv')
        vehicles = pd.read_csv('Case study internship data/vehicle_fleet.csv')
        warehouse = pd.read_csv('Case study internship data/warehouse_inventory.csv')
        feedback = pd.read_csv('Case study internship data/customer_feedback.csv')
        costs = pd.read_csv('Case study internship data/cost_breakdown.csv')
        
        # Merge datasets on Order_ID
        df = orders.merge(delivery_perf, on='Order_ID', how='left')
        df = df.merge(routes, on='Order_ID', how='left')
        df = df.merge(costs, on='Order_ID', how='left')
        df = df.merge(feedback, on='Order_ID', how='left')
        
        print(f"Loaded {len(df)} records")
        return df, vehicles, warehouse
    
    def create_features(self, df):
        """Create engineered features for ML model"""
        print("Creating engineered features...")
        
        # Convert dates
        df['Order_Date'] = pd.to_datetime(df['Order_Date'])
        df['Feedback_Date'] = pd.to_datetime(df['Feedback_Date'])
        
        # Create delay categories
        df['delay_days'] = df['Actual_Delivery_Days'] - df['Promised_Delivery_Days']
        df['delay_category'] = df['delay_days'].apply(self._categorize_delay)
        
        # Priority weight
        priority_map = {'Express': 3, 'Standard': 2, 'Economy': 1}
        df['priority_weight'] = df['Priority'].map(priority_map)
        
        # Customer value tier
        df['customer_value_tier'] = pd.cut(df['Order_Value_INR'], 
                                         bins=[0, 1000, 5000, 20000, float('inf')], 
                                         labels=['Low', 'Medium', 'High', 'Premium'])
        
        # Route difficulty score
        df['route_difficulty'] = (
            df['Distance_KM'] / 1000 + 
            df['Traffic_Delay_Minutes'] / 60 + 
            (df['Weather_Impact'] != 'None').astype(int) * 0.5
        )
        
        # Carrier reliability (historical performance)
        carrier_performance = df.groupby('Carrier')['delay_days'].mean()
        df['carrier_reliability'] = df['Carrier'].map(carrier_performance)
        
        # Cost efficiency
        df['cost_per_km'] = df['Delivery_Cost_INR'] / df['Distance_KM']
        df['fuel_efficiency'] = df['Distance_KM'] / df['Fuel_Consumption_L']
        
        # Weather impact
        df['weather_risk'] = (df['Weather_Impact'] != 'None').astype(int)
        
        # Traffic risk
        df['traffic_risk'] = (df['Traffic_Delay_Minutes'] > 30).astype(int)
        
        # Order value categories
        df['high_value'] = (df['Order_Value_INR'] > df['Order_Value_INR'].quantile(0.8)).astype(int)
        
        # International delivery
        df['international'] = df['Destination'].isin(['Dubai', 'Hong Kong', 'Bangkok', 'Singapore']).astype(int)
        
        # Special handling
        df['special_handling'] = (df['Special_Handling'] != 'None').astype(int)
        
        return df
    
    def _categorize_delay(self, delay_days):
        """Categorize delay days"""
        if delay_days <= 1:
            return 'On-Time'
        elif delay_days <= 3:
            return 'Slightly-Delayed'
        else:
            return 'Severely-Delayed'
    
    def prepare_ml_data(self, df):
        """Prepare data for machine learning"""
        print("Preparing ML data...")
        
        # Select features for ML
        feature_cols = [
            'priority_weight', 'Order_Value_INR', 'Distance_KM', 
            'Fuel_Consumption_L', 'Toll_Charges_INR', 'Traffic_Delay_Minutes',
            'route_difficulty', 'carrier_reliability', 'cost_per_km',
            'fuel_efficiency', 'weather_risk', 'traffic_risk', 'high_value',
            'international', 'special_handling'
        ]
        
        # Categorical features
        categorical_cols = ['Customer_Segment', 'Product_Category', 'Origin', 'Destination']
        
        # Encode categorical variables
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                feature_cols.append(f'{col}_encoded')
                self.label_encoders[col] = le
        
        # Handle missing values
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())
        
        # Prepare features and target
        X = df[feature_cols]
        y = df['delay_category']
        
        # Remove rows with missing target
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        self.feature_columns = feature_cols
        
        return X, y, df
    
    def get_processed_data(self):
        """Main method to get fully processed data"""
        df, vehicles, warehouse = self.load_data()
        df = self.create_features(df)
        X, y, df = self.prepare_ml_data(df)
        
        return X, y, df, vehicles, warehouse
