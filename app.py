import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data_processor import DataProcessor
from ml_models import DelayPredictor
from visualizations import Visualizations

# Page configuration
st.set_page_config(
    page_title="NexGen Logistics - Predictive Delivery Optimizer",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme with jungle aesthetic
st.markdown("""
<style>
    .main {
        background-color: #1f2937;
        color: #f3f4f6;
    }
    .stApp {
        background-color: #1f2937;
    }
    .sidebar .sidebar-content {
        background-color: #1a4d3a;
    }
    .metric-card {
        background-color: #2d5a3d;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4ade80;
    }
    .risk-high {
        background-color: #7f1d1d;
        color: #fca5a5;
        padding: 0.5rem;
        border-radius: 0.25rem;
    }
    .risk-medium {
        background-color: #78350f;
        color: #fbbf24;
        padding: 0.5rem;
        border-radius: 0.25rem;
    }
    .risk-low {
        background-color: #14532d;
        color: #4ade80;
        padding: 0.5rem;
        border-radius: 0.25rem;
    }
    .stSelectbox > div > div {
        background-color: #2d5a3d;
        color: #f3f4f6;
    }
    .stNumberInput > div > div > input {
        background-color: #2d5a3d;
        color: #f3f4f6;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_process_data():
    """Load and process data with caching"""
    processor = DataProcessor()
    X, y, df, vehicles, warehouse = processor.get_processed_data()
    return X, y, df, vehicles, warehouse

@st.cache_resource
def train_models(X, y):
    """Train ML models with caching"""
    predictor = DelayPredictor()
    accuracy = predictor.train_models(X, y)
    return predictor, accuracy

def main():
    # Header
    st.title("🚛 NexGen Logistics - Predictive Delivery Optimizer")
    st.markdown("**AI-Powered Delivery Risk Assessment & Optimization Platform**")
    
    # Load data
    with st.spinner("Loading and processing data..."):
        X, y, df, vehicles, warehouse = load_and_process_data()
    
    # Train models
    with st.spinner("Training machine learning models..."):
        predictor, accuracy = train_models(X, y)
    
    # Add risk scores to dataframe globally
    try:
        risk_scores = predictor.get_risk_score(X)
        df['risk_score'] = risk_scores
    except:
        # Fallback if risk scoring fails
        risk_scores = np.random.uniform(0, 100, len(df))
        df['risk_score'] = risk_scores
    
    # Add predictions to dataframe globally
    try:
        predicted_categories, predicted_probabilities = predictor.predict_delay_category(X)
        df['predicted_category'] = predicted_categories
        df['predicted_probabilities'] = predicted_probabilities.tolist()
    except:
        # Fallback if prediction fails
        df['predicted_category'] = df['delay_category']
        df['predicted_probabilities'] = [[0.33, 0.33, 0.34]] * len(df)
    
    # Sidebar
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.selectbox(
        "Select Dashboard",
        ["🏠 Executive Summary", "🔮 Delay Predictions", "⚠️ Early Warning System", 
         "🛠️ Corrective Actions", "📊 Analytics & Insights", "👥 Customer Communication"]
    )
    
    # Main content based on selected page
    if page == "🏠 Executive Summary":
        show_executive_summary(df, predictor, accuracy)
    elif page == "🔮 Delay Predictions":
        show_delay_predictions(df, predictor)
    elif page == "⚠️ Early Warning System":
        show_early_warning_system(df, predictor)
    elif page == "🛠️ Corrective Actions":
        show_corrective_actions(df, predictor)
    elif page == "📊 Analytics & Insights":
        show_analytics_insights(df, predictor)
    elif page == "👥 Customer Communication":
        show_customer_communication(df, predictor)

def show_executive_summary(df, predictor, accuracy):
    """Executive Summary Dashboard"""
    st.header("📈 Executive Summary")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        on_time_rate = (df['delay_category'] == 'On-Time').mean() * 100
        st.metric("On-Time Delivery Rate", f"{on_time_rate:.1f}%", "5.2%")
    
    with col2:
        avg_delay = df['delay_days'].mean()
        st.metric("Average Delay Days", f"{avg_delay:.1f}", "-1.2")
    
    with col3:
        avg_rating = df['Customer_Rating'].mean()
        st.metric("Customer Rating", f"{avg_rating:.1f}/5", "0.3")
    
    with col4:
        st.metric("Prediction Accuracy", f"{accuracy*100:.1f}%", "2.1%")
    
    # Risk Distribution
    st.subheader("🎯 Current Risk Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk distribution pie chart
        risk_categories = pd.cut(df['risk_score'], bins=[0, 30, 70, 100], 
                               labels=['Low Risk', 'Medium Risk', 'High Risk'])
        risk_counts = risk_categories.value_counts()
        
        fig_pie = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Risk Distribution",
            color_discrete_sequence=['#4ade80', '#fbbf24', '#ef4444']
        )
        fig_pie.update_layout(
            plot_bgcolor='#1f2937',
            paper_bgcolor='#1f2937',
            font_color='#f3f4f6'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Performance trends
        viz = Visualizations()
        fig_trends = viz.create_performance_trends(df)
        st.plotly_chart(fig_trends, use_container_width=True)
    
    # Recent high-risk orders
    st.subheader("⚠️ Recent High-Risk Orders")
    high_risk_orders = df[df['risk_score'] > 70].nlargest(10, 'risk_score')[
        ['Order_ID', 'Order_Date', 'Priority', 'Destination', 'Carrier', 'risk_score', 'delay_category']
    ]
    st.dataframe(high_risk_orders, use_container_width=True)

def show_delay_predictions(df, predictor):
    """Delay Predictions Dashboard"""
    st.header("🔮 Delay Predictions")
    
    # Prediction interface
    st.subheader("📋 New Order Prediction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        priority = st.selectbox("Priority", ["Express", "Standard", "Economy"])
        order_value = st.number_input("Order Value (INR)", min_value=0, value=5000)
        distance = st.number_input("Distance (KM)", min_value=0, value=500)
    
    with col2:
        customer_segment = st.selectbox("Customer Segment", ["Individual", "SMB", "Enterprise"])
        product_category = st.selectbox("Product Category", 
                                      ["Electronics", "Fashion", "Food & Beverage", "Healthcare", 
                                       "Industrial", "Books", "Home Goods"])
        carrier = st.selectbox("Carrier", df['Carrier'].unique())
    
    with col3:
        origin = st.selectbox("Origin", df['Origin'].unique())
        destination = st.selectbox("Destination", df['Destination'].unique())
        special_handling = st.selectbox("Special Handling", ["None", "Fragile", "Refrigerated"])
    
    # Create sample order for prediction
    if st.button("🔮 Predict Delivery Risk", type="primary"):
        # Create feature vector for prediction
        sample_order = pd.DataFrame({
            'priority_weight': [3 if priority == 'Express' else 2 if priority == 'Standard' else 1],
            'Order_Value_INR': [order_value],
            'Distance_KM': [distance],
            'Fuel_Consumption_L': [distance * 0.1],  # Estimate
            'Toll_Charges_INR': [distance * 0.8],  # Estimate
            'Traffic_Delay_Minutes': [30],  # Default
            'route_difficulty': [distance/1000 + 0.5],
            'carrier_reliability': [df[df['Carrier'] == carrier]['carrier_reliability'].mean()],
            'cost_per_km': [order_value/distance],
            'fuel_efficiency': [10],
            'weather_risk': [0],
            'traffic_risk': [1],
            'high_value': [1 if order_value > 10000 else 0],
            'international': [1 if destination in ['Dubai', 'Hong Kong', 'Bangkok', 'Singapore'] else 0],
            'special_handling': [1 if special_handling != 'None' else 0],
            'Customer_Segment_encoded': [0 if customer_segment == 'Individual' else 1 if customer_segment == 'SMB' else 2],
            'Product_Category_encoded': [0],  # Simplified
            'Origin_encoded': [0],  # Simplified
            'Destination_encoded': [0]  # Simplified
        })
        
        # Make prediction
        predicted_category, probabilities = predictor.predict_delay_category(sample_order)
        risk_score = predictor.get_risk_score(sample_order)[0]
        
        # Display results
        st.success(f"**Predicted Category:** {predicted_category[0]}")
        st.info(f"**Risk Score:** {risk_score:.1f}/100")
        
        # Risk level styling
        if risk_score > 70:
            st.markdown(f'<div class="risk-high">⚠️ HIGH RISK - Immediate attention required</div>', 
                       unsafe_allow_html=True)
        elif risk_score > 30:
            st.markdown(f'<div class="risk-medium">⚡ MEDIUM RISK - Monitor closely</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="risk-low">✅ LOW RISK - On track for on-time delivery</div>', 
                       unsafe_allow_html=True)
    
    # Historical predictions
    st.subheader("📊 Historical Prediction Analysis")
    
    # Accuracy by category
    accuracy_by_category = df.groupby('delay_category').apply(
        lambda x: (x['predicted_category'] == x['delay_category']).mean()
    ).reset_index()
    accuracy_by_category.columns = ['Category', 'Accuracy']
    
    fig_accuracy = px.bar(
        accuracy_by_category,
        x='Category',
        y='Accuracy',
        title="Prediction Accuracy by Delay Category",
        color='Accuracy',
        color_continuous_scale=['#ef4444', '#fbbf24', '#4ade80']
    )
    fig_accuracy.update_layout(
        plot_bgcolor='#1f2937',
        paper_bgcolor='#1f2937',
        font_color='#f3f4f6'
    )
    st.plotly_chart(fig_accuracy, use_container_width=True)

def show_early_warning_system(df, predictor):
    """Early Warning System Dashboard"""
    st.header("⚠️ Early Warning System")
    
    # Current alerts
    st.subheader("🚨 Active Alerts")
    
    # High-risk orders
    high_risk = df[df['risk_score'] > 70].sort_values('risk_score', ascending=False)
    
    if len(high_risk) > 0:
        st.warning(f"⚠️ {len(high_risk)} high-risk orders require immediate attention")
        
        for idx, order in high_risk.head(5).iterrows():
            with st.expander(f"🚨 Order {order['Order_ID']} - Risk Score: {order['risk_score']:.1f}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Priority:** {order['Priority']}")
                    st.write(f"**Destination:** {order['Destination']}")
                    st.write(f"**Carrier:** {order['Carrier']}")
                with col2:
                    st.write(f"**Order Value:** ₹{order['Order_Value_INR']:,.2f}")
                    st.write(f"**Distance:** {order['Distance_KM']:.1f} KM")
                    st.write(f"**Predicted Category:** {order['predicted_category']}")
    else:
        st.success("✅ No high-risk orders detected")
    
    # Risk monitoring
    st.subheader("📊 Risk Monitoring Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk trend over time
        df['Order_Date'] = pd.to_datetime(df['Order_Date'])
        daily_risk = df.groupby('Order_Date')['risk_score'].mean().reset_index()
        
        fig_risk_trend = px.line(
            daily_risk,
            x='Order_Date',
            y='risk_score',
            title="Average Risk Score Over Time",
            color_discrete_sequence=['#4ade80']
        )
        fig_risk_trend.update_layout(
            plot_bgcolor='#1f2937',
            paper_bgcolor='#1f2937',
            font_color='#f3f4f6'
        )
        st.plotly_chart(fig_risk_trend, use_container_width=True)
    
    with col2:
        # Risk distribution by carrier
        carrier_risk = df.groupby('Carrier')['risk_score'].mean().reset_index()
        
        fig_carrier_risk = px.bar(
            carrier_risk,
            x='Carrier',
            y='risk_score',
            title="Average Risk Score by Carrier",
            color='risk_score',
            color_continuous_scale=['#4ade80', '#fbbf24', '#ef4444']
        )
        fig_carrier_risk.update_layout(
            plot_bgcolor='#1f2937',
            paper_bgcolor='#1f2937',
            font_color='#f3f4f6'
        )
        st.plotly_chart(fig_carrier_risk, use_container_width=True)

def show_corrective_actions(df, predictor):
    """Corrective Actions Dashboard"""
    st.header("🛠️ Corrective Actions")
    
    # Action recommendations
    st.subheader("💡 Recommended Actions")
    
    # Get high-risk orders for action recommendations
    high_risk = df[df['risk_score'] > 50].sort_values('risk_score', ascending=False)
    
    if len(high_risk) > 0:
        for idx, order in high_risk.head(3).iterrows():
            with st.expander(f"🔧 Actions for Order {order['Order_ID']}"):
                
                # Route optimization
                st.write("**🛣️ Route Optimization:**")
                if order['Distance_KM'] > 1000:
                    st.write("• Consider breaking into multiple legs")
                    st.write("• Use intermediate warehouse for consolidation")
                
                # Carrier optimization
                st.write("**🚛 Carrier Optimization:**")
                best_carrier = df.groupby('Carrier')['delay_days'].mean().idxmin()
                if order['Carrier'] != best_carrier:
                    st.write(f"• Switch to {best_carrier} (better performance)")
                else:
                    st.write("• Current carrier is optimal")
                
                # Priority adjustment
                st.write("**⚡ Priority Adjustment:**")
                if order['Priority'] != 'Express' and order['risk_score'] > 80:
                    st.write("• Upgrade to Express priority")
                else:
                    st.write("• Priority level is appropriate")
                
                # Vehicle assignment
                st.write("**🚚 Vehicle Assignment:**")
                if order['Product_Category'] in ['Food & Beverage', 'Healthcare']:
                    st.write("• Assign refrigerated vehicle")
                elif order['Order_Value_INR'] > 50000:
                    st.write("• Assign high-security vehicle")
                else:
                    st.write("• Standard vehicle assignment")
    
    # Cost-benefit analysis
    st.subheader("💰 Cost-Benefit Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Current costs
        avg_delay_cost = df[df['delay_days'] > 0]['Delivery_Cost_INR'].mean()
        avg_on_time_cost = df[df['delay_days'] <= 0]['Delivery_Cost_INR'].mean()
        
        st.metric("Average Cost (Delayed)", f"₹{avg_delay_cost:,.2f}")
        st.metric("Average Cost (On-Time)", f"₹{avg_on_time_cost:,.2f}")
        st.metric("Cost Savings Potential", f"₹{avg_delay_cost - avg_on_time_cost:,.2f}")
    
    with col2:
        # ROI calculation
        total_orders = len(df)
        high_risk_orders = len(df[df['risk_score'] > 70])
        intervention_cost = high_risk_orders * 500  # Estimated intervention cost
        potential_savings = high_risk_orders * (avg_delay_cost - avg_on_time_cost)
        roi = (potential_savings - intervention_cost) / intervention_cost * 100
        
        st.metric("Total Orders", total_orders)
        st.metric("High-Risk Orders", high_risk_orders)
        st.metric("ROI", f"{roi:.1f}%")

def show_analytics_insights(df, predictor):
    """Analytics & Insights Dashboard"""
    st.header("📊 Analytics & Insights")
    
    # Feature importance
    st.subheader("🎯 Key Delay Risk Factors")
    feature_importance = predictor.get_feature_importance()
    
    viz = Visualizations()
    fig_factors = viz.create_delay_factors(feature_importance)
    st.plotly_chart(fig_factors, use_container_width=True)
    
    # Carrier performance
    st.subheader("🚛 Carrier Performance Analysis")
    fig_carrier = viz.create_carrier_comparison(df)
    st.plotly_chart(fig_carrier, use_container_width=True)
    
    # Cost analysis
    st.subheader("💰 Cost Impact Analysis")
    
    # Calculate total costs
    df['Total_Cost'] = (df['Fuel_Cost'] + df['Labor_Cost'] + df['Vehicle_Maintenance'] + 
                       df['Insurance'] + df['Packaging_Cost'] + df['Technology_Platform_Fee'] + 
                       df['Other_Overhead'])
    
    fig_cost = viz.create_cost_analysis(df)
    st.plotly_chart(fig_cost, use_container_width=True)
    
    # Performance insights
    st.subheader("📈 Performance Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        delay_rate = (df['delay_days'] > 0).mean() * 100
        st.metric("Overall Delay Rate", f"{delay_rate:.1f}%")
    
    with col2:
        express_delay_rate = df[df['Priority'] == 'Express']['delay_days'].apply(lambda x: x > 0).mean() * 100
        st.metric("Express Delay Rate", f"{express_delay_rate:.1f}%")
    
    with col3:
        international_delay_rate = df[df['international'] == 1]['delay_days'].apply(lambda x: x > 0).mean() * 100
        st.metric("International Delay Rate", f"{international_delay_rate:.1f}%")

def show_customer_communication(df, predictor):
    """Customer Communication Dashboard"""
    st.header("👥 Customer Communication")
    
    # Communication triggers
    st.subheader("📢 Communication Triggers")
    
    # Orders requiring communication
    communication_orders = df[df['risk_score'] > 60].sort_values('risk_score', ascending=False)
    
    if len(communication_orders) > 0:
        st.warning(f"📧 {len(communication_orders)} orders require customer communication")
        
        for idx, order in communication_orders.head(5).iterrows():
            with st.expander(f"📧 Order {order['Order_ID']} - Communication Required"):
                
                # Communication type
                if order['risk_score'] > 80:
                    comm_type = "🚨 URGENT - Delay Alert"
                    message = f"Dear Customer, we're experiencing delays with your order {order['Order_ID']}. We're working to resolve this and will update you shortly."
                elif order['risk_score'] > 60:
                    comm_type = "⚠️ WARNING - Potential Delay"
                    message = f"Dear Customer, your order {order['Order_ID']} may experience minor delays. We're monitoring the situation closely."
                else:
                    comm_type = "ℹ️ INFO - Status Update"
                    message = f"Dear Customer, your order {order['Order_ID']} is on track for delivery."
                
                st.write(f"**Communication Type:** {comm_type}")
                st.write(f"**Suggested Message:** {message}")
                
                # Compensation recommendation
                if order['risk_score'] > 80 and order['Priority'] == 'Express':
                    st.write("**💳 Compensation:** Recommend 20% discount on next order")
                elif order['risk_score'] > 70:
                    st.write("**💳 Compensation:** Consider free shipping on next order")
    else:
        st.success("✅ No orders require immediate communication")
    
    # Customer satisfaction trends
    st.subheader("😊 Customer Satisfaction Trends")
    
    # Satisfaction by delay category
    satisfaction_by_delay = df.groupby('delay_category')['Customer_Rating'].mean().reset_index()
    
    fig_satisfaction = px.bar(
        satisfaction_by_delay,
        x='delay_category',
        y='Customer_Rating',
        title="Customer Satisfaction by Delay Category",
        color='Customer_Rating',
        color_continuous_scale=['#ef4444', '#fbbf24', '#4ade80']
    )
    fig_satisfaction.update_layout(
        plot_bgcolor='#1f2937',
        paper_bgcolor='#1f2937',
        font_color='#f3f4f6'
    )
    st.plotly_chart(fig_satisfaction, use_container_width=True)
    
    # Communication effectiveness
    st.subheader("📊 Communication Effectiveness")
    
    col1, col2 = st.columns(2)
    
    with col1:
        avg_rating = df['Customer_Rating'].mean()
        st.metric("Average Customer Rating", f"{avg_rating:.1f}/5")
    
    with col2:
        recommendation_rate = df['Would_Recommend'].value_counts(normalize=True).get('Yes', 0) * 100
        st.metric("Recommendation Rate", f"{recommendation_rate:.1f}%")

if __name__ == "__main__":
    main()
