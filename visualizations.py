import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class Visualizations:
    def __init__(self):
        self.color_scheme = {
            'primary': '#1a4d3a',
            'secondary': '#2d5a3d', 
            'accent': '#4ade80',
            'background': '#1f2937',
            'text': '#f3f4f6'
        }
    
    def create_risk_heatmap(self, df):
        """Create risk heatmap for orders"""
        # Add a dummy risk_score if it doesn't exist
        if 'risk_score' not in df.columns:
            df = df.copy()
            df['risk_score'] = np.random.uniform(0, 100, len(df))
        
        fig = px.scatter(
            df, 
            x='Order_Date', 
            y='Order_ID',
            color='risk_score',
            size='Order_Value_INR',
            hover_data=['Priority', 'Carrier', 'Destination', 'delay_category'],
            color_continuous_scale=['#4ade80', '#fbbf24', '#ef4444'],
            title="Order Risk Assessment Heatmap"
        )
        
        fig.update_layout(
            plot_bgcolor=self.color_scheme['background'],
            paper_bgcolor=self.color_scheme['background'],
            font_color=self.color_scheme['text'],
            title_font_color=self.color_scheme['accent']
        )
        
        return fig
    
    def create_accuracy_chart(self, df):
        """Create prediction accuracy over time"""
        df['prediction_correct'] = (df['predicted_category'] == df['delay_category']).astype(int)
        accuracy_by_date = df.groupby('Order_Date')['prediction_correct'].mean().reset_index()
        
        fig = px.line(
            accuracy_by_date,
            x='Order_Date',
            y='prediction_correct',
            title="Prediction Accuracy Over Time",
            labels={'prediction_correct': 'Accuracy', 'Order_Date': 'Date'}
        )
        
        fig.update_layout(
            plot_bgcolor=self.color_scheme['background'],
            paper_bgcolor=self.color_scheme['background'],
            font_color=self.color_scheme['text'],
            title_font_color=self.color_scheme['accent']
        )
        
        return fig
    
    def create_delay_factors(self, feature_importance):
        """Create bar chart of delay factors"""
        if feature_importance is None:
            # Create dummy data if feature_importance is None
            feature_importance = pd.DataFrame({
                'feature': ['Distance_KM', 'Traffic_Delay_Minutes', 'Weather_Impact', 'Priority', 'Order_Value_INR'],
                'importance': [0.3, 0.25, 0.2, 0.15, 0.1]
            })
        
        fig = px.bar(
            feature_importance.head(10),
            x='importance',
            y='feature',
            orientation='h',
            title="Top Delay Risk Factors",
            color='importance',
            color_continuous_scale=['#4ade80', '#ef4444']
        )
        
        fig.update_layout(
            plot_bgcolor=self.color_scheme['background'],
            paper_bgcolor=self.color_scheme['background'],
            font_color=self.color_scheme['text'],
            title_font_color=self.color_scheme['accent']
        )
        
        return fig
    
    def create_carrier_comparison(self, df):
        """Create carrier performance comparison"""
        carrier_stats = df.groupby('Carrier').agg({
            'delay_days': 'mean',
            'Customer_Rating': 'mean',
            'Delivery_Cost_INR': 'mean',
            'Order_ID': 'count'
        }).reset_index()
        
        carrier_stats.columns = ['Carrier', 'Avg_Delay_Days', 'Avg_Rating', 'Avg_Cost', 'Order_Count']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Average Delay Days', 'Average Rating', 'Average Cost', 'Order Count'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Delay days
        fig.add_trace(
            go.Bar(x=carrier_stats['Carrier'], y=carrier_stats['Avg_Delay_Days'], 
                   name='Delay Days', marker_color=self.color_scheme['accent']),
            row=1, col=1
        )
        
        # Rating
        fig.add_trace(
            go.Bar(x=carrier_stats['Carrier'], y=carrier_stats['Avg_Rating'], 
                   name='Rating', marker_color=self.color_scheme['secondary']),
            row=1, col=2
        )
        
        # Cost
        fig.add_trace(
            go.Bar(x=carrier_stats['Carrier'], y=carrier_stats['Avg_Cost'], 
                   name='Cost', marker_color=self.color_scheme['primary']),
            row=2, col=1
        )
        
        # Order count
        fig.add_trace(
            go.Bar(x=carrier_stats['Carrier'], y=carrier_stats['Order_Count'], 
                   name='Orders', marker_color='#fbbf24'),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Carrier Performance Comparison",
            plot_bgcolor=self.color_scheme['background'],
            paper_bgcolor=self.color_scheme['background'],
            font_color=self.color_scheme['text'],
            title_font_color=self.color_scheme['accent'],
            showlegend=False
        )
        
        return fig
    
    def create_risk_gauge(self, current_risk):
        """Create risk gauge for current situation"""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = current_risk,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Current Risk Level"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': self.color_scheme['accent']},
                'steps': [
                    {'range': [0, 30], 'color': self.color_scheme['accent']},
                    {'range': [30, 70], 'color': '#fbbf24'},
                    {'range': [70, 100], 'color': '#ef4444'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        
        fig.update_layout(
            plot_bgcolor=self.color_scheme['background'],
            paper_bgcolor=self.color_scheme['background'],
            font_color=self.color_scheme['text']
        )
        
        return fig
    
    def create_cost_analysis(self, df):
        """Create cost impact analysis"""
        cost_by_delay = df.groupby('delay_category').agg({
            'Delivery_Cost_INR': 'mean',
            'Fuel_Cost': 'mean',
            'Labor_Cost': 'mean',
            'Total_Cost': 'mean'
        }).reset_index()
        
        fig = px.bar(
            cost_by_delay,
            x='delay_category',
            y=['Delivery_Cost_INR', 'Fuel_Cost', 'Labor_Cost'],
            title="Cost Breakdown by Delay Category",
            barmode='group'
        )
        
        fig.update_layout(
            plot_bgcolor=self.color_scheme['background'],
            paper_bgcolor=self.color_scheme['background'],
            font_color=self.color_scheme['text'],
            title_font_color=self.color_scheme['accent']
        )
        
        return fig
    
    def create_performance_trends(self, df):
        """Create performance trends over time"""
        df['Order_Date'] = pd.to_datetime(df['Order_Date'])
        daily_performance = df.groupby('Order_Date').agg({
            'delay_days': 'mean',
            'Customer_Rating': 'mean',
            'Delivery_Cost_INR': 'mean'
        }).reset_index()
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Average Delay Days', 'Average Customer Rating', 'Average Delivery Cost'),
            vertical_spacing=0.1
        )
        
        # Delay days
        fig.add_trace(
            go.Scatter(x=daily_performance['Order_Date'], y=daily_performance['delay_days'],
                      mode='lines+markers', name='Delay Days', line_color=self.color_scheme['accent']),
            row=1, col=1
        )
        
        # Rating
        fig.add_trace(
            go.Scatter(x=daily_performance['Order_Date'], y=daily_performance['Customer_Rating'],
                      mode='lines+markers', name='Rating', line_color=self.color_scheme['secondary']),
            row=2, col=1
        )
        
        # Cost
        fig.add_trace(
            go.Scatter(x=daily_performance['Order_Date'], y=daily_performance['Delivery_Cost_INR'],
                      mode='lines+markers', name='Cost', line_color=self.color_scheme['primary']),
            row=3, col=1
        )
        
        fig.update_layout(
            title="Performance Trends Over Time",
            plot_bgcolor=self.color_scheme['background'],
            paper_bgcolor=self.color_scheme['background'],
            font_color=self.color_scheme['text'],
            title_font_color=self.color_scheme['accent'],
            height=800,
            showlegend=False
        )
        
        return fig
