"""
Employee Attrition Risk Prediction Dashboard
A Professional ML-Powered Analytics Platform for Palo Alto Networks
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os

# Page Configuration
st.set_page_config(
    page_title="Employee Attrition Risk Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main header styling */
    .main-title {
        font-size: 48px;
        font-weight: bold;
        background: linear-gradient(120deg, #1e3a8a 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 20px;
        margin-bottom: 10px;
    }
    
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 18px;
        margin-bottom: 30px;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.3s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-value {
        font-size: 36px;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 14px;
        opacity: 0.9;
    }
    
    /* Risk badges */
    .risk-high {
        background-color: #dc2626;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    
    .risk-medium {
        background-color: #f59e0b;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    
    .risk-low {
        background-color: #10b981;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    
    /* Info box */
    .info-box {
        background-color: #f0f9ff;
        border-left: 4px solid #3b82f6;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    /* Section headers */
    .section-header {
        font-size: 28px;
        font-weight: bold;
        color: #1e293b;
        margin: 30px 0 20px 0;
        padding-bottom: 10px;
        border-bottom: 3px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

# Auto-training functions
def train_and_save_model():
    """Train model from scratch"""
    try:
        from model_training import AttritionPredictor
        
        st.info("🔄 Training machine learning models... This will take 2-3 minutes...")
        progress_bar = st.progress(0)
        
        predictor = AttritionPredictor()
        progress_bar.progress(30)
        
        predictor.prepare_and_train('employee_data.csv')
        progress_bar.progress(100)
        
        st.success("✅ Model training complete!")
        st.balloons()
        return True
    except Exception as e:
        st.error(f"❌ Training failed: {str(e)}")
        st.error("Please check that employee_data.csv exists and is properly formatted.")
        return False

# Load Resources
@st.cache_resource
def load_model_components():
    """Load trained model and components with error handling"""
    try:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return model, scaler, label_encoders, feature_names
    except (FileNotFoundError, ModuleNotFoundError, AttributeError) as e:
        st.warning("⚠️ Model files not found or incompatible. Training new models...")
        
        # Delete old incompatible files
        for file in ['best_model.pkl', 'scaler.pkl', 'label_encoders.pkl', 'feature_names.pkl']:
            if os.path.exists(file):
                try:
                    os.remove(file)
                except:
                    pass
        
        # Clear cache
        st.cache_resource.clear()
        
        # Trigger retraining
        if train_and_save_model():
            # Try loading again
            try:
                model = joblib.load('best_model.pkl')
                scaler = joblib.load('scaler.pkl')
                label_encoders = joblib.load('label_encoders.pkl')
                feature_names = joblib.load('feature_names.pkl')
                return model, scaler, label_encoders, feature_names
            except Exception as e:
                st.error(f"Failed to load models after training: {str(e)}")
                st.stop()
        else:
            st.error("Cannot proceed without trained models.")
            st.stop()

@st.cache_data
def load_data():
    """Load employee data with predictions"""
    try:
        df = pd.read_csv('employee_predictions.csv')
        return df
    except FileNotFoundError:
        try:
            # If predictions don't exist, load raw data
            df = pd.read_csv('employee_data.csv')
            st.warning("Predictions file not found. Using raw data.")
            return df
        except FileNotFoundError:
            st.error("⚠️ Data file not found. Please ensure employee_data.csv exists.")
            st.stop()

# Helper Functions
def get_risk_badge(risk_category):
    """Return HTML badge for risk category"""
    if risk_category == 'High':
        return '<span class="risk-high">🔴 HIGH RISK</span>'
    elif risk_category == 'Medium':
        return '<span class="risk-medium">🟡 MEDIUM RISK</span>'
    else:
        return '<span class="risk-low">🟢 LOW RISK</span>'

def create_gauge_chart(value, title):
    """Create a gauge chart for risk score"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24}},
        delta={'reference': 30, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#d1fae5'},
                {'range': [30, 60], 'color': '#fef3c7'},
                {'range': [60, 100], 'color': '#fee2e2'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 60
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="white",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def show_dashboard(df):
    """Main dashboard view"""
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    total_employees = len(df)
    high_risk = len(df[df['RiskCategory'] == 'High']) if 'RiskCategory' in df.columns else 0
    medium_risk = len(df[df['RiskCategory'] == 'Medium']) if 'RiskCategory' in df.columns else 0
    avg_risk = df['AttritionRisk'].mean() if 'AttritionRisk' in df.columns else 0
    
    with col1:
        pct_high = (high_risk / total_employees * 100) if total_employees > 0 else 0
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
            <div class="metric-label">Total Employees</div>
            <div class="metric-value">{total_employees}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        pct_high = (high_risk / total_employees * 100) if total_employees > 0 else 0
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <div class="metric-label">High Risk</div>
            <div class="metric-value">{high_risk}</div>
            <div class="metric-label">{pct_high:.1f}% of workforce</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        pct_medium = (medium_risk / total_employees * 100) if total_employees > 0 else 0
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);">
            <div class="metric-label">Medium Risk</div>
            <div class="metric-value">{medium_risk}</div>
            <div class="metric-label">{pct_medium:.1f}% of workforce</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);">
            <div class="metric-label">Average Risk Score</div>
            <div class="metric-value">{avg_risk:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Only show charts if we have risk data
    if 'RiskCategory' in df.columns and 'AttritionRisk' in df.columns:
        # Charts Row
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="section-header">📊 Risk Distribution</div>', unsafe_allow_html=True)
            
            risk_counts = df['RiskCategory'].value_counts()
            colors = {'High': '#dc2626', 'Medium': '#f59e0b', 'Low': '#10b981'}
            
            fig = go.Figure(data=[go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                hole=0.4,
                marker_colors=[colors.get(cat, '#3b82f6') for cat in risk_counts.index],
                textinfo='label+percent',
                textfont_size=14,
                pull=[0.1 if cat == 'High' else 0 for cat in risk_counts.index]
            )])
            
            fig.update_layout(
                showlegend=True,
                height=400,
                margin=dict(t=20, b=20, l=20, r=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<div class="section-header">📈 Risk Score Distribution</div>', unsafe_allow_html=True)
            
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=df['AttritionRisk'],
                nbinsx=30,
                marker_color='#3b82f6',
                opacity=0.7,
                name='Risk Distribution'
            ))
            
            fig.add_vline(x=0.3, line_dash="dash", line_color="green", annotation_text="Low Risk Threshold")
            fig.add_vline(x=0.6, line_dash="dash", line_color="red", annotation_text="High Risk Threshold")
            
            fig.update_layout(
                xaxis_title="Attrition Risk Probability",
                yaxis_title="Number of Employees",
                height=400,
                showlegend=False,
                margin=dict(t=20, b=50, l=50, r=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk by Department
        if 'Department' in df.columns:
            st.markdown('<div class="section-header">🏢 Risk by Department</div>', unsafe_allow_html=True)
            
            dept_risk = df.groupby('Department').agg({
                'AttritionRisk': 'mean',
                'EmployeeID': 'count'
            }).reset_index()
            dept_risk.columns = ['Department', 'Average Risk', 'Employee Count']
            dept_risk = dept_risk.sort_values('Average Risk', ascending=False)
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=dept_risk['Department'],
                y=dept_risk['Average Risk'],
                marker_color=dept_risk['Average Risk'],
                marker_colorscale='RdYlGn_r',
                text=[f"{val:.1%}" for val in dept_risk['Average Risk']],
                textposition='outside',
                name='Average Risk'
            ))
            
            fig.update_layout(
                xaxis_title="Department",
                yaxis_title="Average Attrition Risk",
                height=400,
                showlegend=False,
                yaxis_tickformat='.0%'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # High Risk Employees Table
        st.markdown('<div class="section-header">🚨 Top 10 Highest Risk Employees</div>', unsafe_allow_html=True)
        
        display_cols = ['EmployeeID', 'Department', 'JobRole', 'YearsAtCompany', 'AttritionRisk', 'RiskCategory']
        available_cols = [col for col in display_cols if col in df.columns]
        
        high_risk_df = df.nlargest(10, 'AttritionRisk')[available_cols].copy()
        high_risk_df['AttritionRisk'] = high_risk_df['AttritionRisk'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(high_risk_df, use_container_width=True, hide_index=True)
    else:
        st.info("📊 Risk predictions will be generated after model training completes.")

def show_department_analytics(df):
    """Department-level analytics view"""
    
    st.markdown('<div class="section-header">🏢 Department-Level Risk Analysis</div>', unsafe_allow_html=True)
    
    if 'Department' not in df.columns:
        st.warning("Department data not available.")
        return
    
    # Department selector
    departments = ['All Departments'] + sorted(df['Department'].unique().tolist())
    selected_dept = st.selectbox("Select Department", departments)
    
    if selected_dept != 'All Departments':
        df_filtered = df[df['Department'] == selected_dept]
    else:
        df_filtered = df
    
    # Department Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Employees", len(df_filtered))
    
    with col2:
        high_risk_count = len(df_filtered[df_filtered['RiskCategory'] == 'High']) if 'RiskCategory' in df.columns else 0
        st.metric("High Risk", high_risk_count, delta=f"{high_risk_count/len(df_filtered)*100:.1f}%" if len(df_filtered) > 0 else "0%")
    
    with col3:
        avg_risk = df_filtered['AttritionRisk'].mean() if 'AttritionRisk' in df.columns else 0
        st.metric("Average Risk", f"{avg_risk:.1%}")
    
    with col4:
        avg_tenure = df_filtered['YearsAtCompany'].mean() if 'YearsAtCompany' in df.columns else 0
        st.metric("Avg Tenure (Years)", f"{avg_tenure:.1f}")
    
    st.markdown("---")
    
    if 'RiskCategory' in df.columns:
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Risk Category Breakdown")
            
            risk_breakdown = df_filtered['RiskCategory'].value_counts()
            colors = ['#10b981', '#f59e0b', '#dc2626']
            
            fig = go.Figure(data=[go.Bar(
                x=risk_breakdown.index,
                y=risk_breakdown.values,
                marker_color=colors[:len(risk_breakdown)],
                text=risk_breakdown.values,
                textposition='outside'
            )])
            
            fig.update_layout(
                xaxis_title="Risk Category",
                yaxis_title="Number of Employees",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Risk by Job Role")
            
            if 'JobRole' in df.columns:
                role_risk = df_filtered.groupby('JobRole')['AttritionRisk'].mean().sort_values(ascending=False).head(10)
                
                fig = go.Figure(data=[go.Bar(
                    y=role_risk.index,
                    x=role_risk.values,
                    orientation='h',
                    marker_color=role_risk.values,
                    marker_colorscale='RdYlGn_r',
                    text=[f"{val:.1%}" for val in role_risk.values],
                    textposition='outside'
                )])
                
                fig.update_layout(
                    xaxis_title="Average Risk",
                    yaxis_title="Job Role",
                    height=400,
                    xaxis_tickformat='.0%'
                )
                
                st.plotly_chart(fig, use_container_width=True)

def show_employee_profile(df):
    """Individual employee profile view"""
    
    st.markdown('<div class="section-header">👤 Employee Risk Profile</div>', unsafe_allow_html=True)
    
    if 'EmployeeID' not in df.columns:
        st.warning("Employee ID data not available.")
        return
    
    # Employee selector
    employee_id = st.selectbox("Select Employee ID", df['EmployeeID'].tolist())
    
    # Get employee data
    employee = df[df['EmployeeID'] == employee_id].iloc[0]
    
    st.markdown("---")
    
    # Employee Header
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.markdown(f"### Employee #{employee['EmployeeID']}")
        if 'Department' in employee:
            st.markdown(f"**Department:** {employee['Department']}")
        if 'JobRole' in employee:
            st.markdown(f"**Job Role:** {employee['JobRole']}")
    
    with col2:
        if 'Age' in employee:
            st.markdown(f"**Age:** {employee['Age']}")
        if 'Gender' in employee:
            st.markdown(f"**Gender:** {employee['Gender']}")
        if 'YearsAtCompany' in employee:
            st.markdown(f"**Years at Company:** {employee['YearsAtCompany']}")
    
    with col3:
        if 'RiskCategory' in employee:
            risk_badge = get_risk_badge(employee['RiskCategory'])
            st.markdown(risk_badge, unsafe_allow_html=True)
        if 'AttritionRisk' in employee:
            st.markdown(f"**Score:** {employee['AttritionRisk']:.1%}")
    
    st.markdown("---")
    
    # Risk Gauge
    if 'AttritionRisk' in employee:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            fig = create_gauge_chart(employee['AttritionRisk'], "Attrition Risk")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Employee Metrics")
            
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                if 'MonthlyIncome' in employee:
                    st.metric("Monthly Income", f"${employee['MonthlyIncome']:,.0f}")
                if 'JobSatisfaction' in employee:
                    st.metric("Job Satisfaction", f"{employee['JobSatisfaction']}/4")
            
            with metric_col2:
                if 'YearsInCurrentRole' in employee:
                    st.metric("Years in Role", f"{employee['YearsInCurrentRole']}")
                if 'WorkLifeBalance' in employee:
                    st.metric("Work-Life Balance", f"{employee['WorkLifeBalance']}/4")
            
            with metric_col3:
                if 'YearsSinceLastPromotion' in employee:
                    st.metric("Last Promotion", f"{employee['YearsSinceLastPromotion']} yrs ago")
                if 'PerformanceRating' in employee:
                    st.metric("Performance", f"{employee.get('PerformanceRating', 'N/A')}")

def show_insights_actions(df):
    """Insights and actionable recommendations"""
    
    st.markdown('<div class="section-header">💡 Strategic Insights & Action Plan</div>', unsafe_allow_html=True)
    
    if 'RiskCategory' not in df.columns or 'AttritionRisk' not in df.columns:
        st.info("Insights will be available after risk predictions are generated.")
        return
    
    # Executive Summary
    st.markdown("### 📊 Executive Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_high_risk = len(df[df['RiskCategory'] == 'High'])
        st.metric("Priority Actions", total_high_risk, help="Employees requiring immediate intervention")
    
    with col2:
        avg_cost_per_employee = 75000
        potential_cost = total_high_risk * avg_cost_per_employee
        st.metric("Potential Cost at Risk", f"${potential_cost:,.0f}", 
                 help="Estimated replacement cost for high-risk employees")
    
    with col3:
        retention_potential = int(total_high_risk * 0.7)
        st.metric("Retention Potential", retention_potential,
                 help="Employees that could be retained with intervention")
    
    st.markdown("---")
    
    # Priority Employees
    st.markdown("### 🚨 Priority Intervention List")
    
    display_cols = ['EmployeeID', 'Department', 'JobRole', 'YearsAtCompany', 
                   'JobSatisfaction', 'WorkLifeBalance', 'AttritionRisk']
    available_cols = [col for col in display_cols if col in df.columns]
    
    priority_df = df[df['RiskCategory'] == 'High'].nlargest(20, 'AttritionRisk')[available_cols].copy()
    
    if 'AttritionRisk' in priority_df.columns:
        priority_df['AttritionRisk'] = priority_df['AttritionRisk'].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(priority_df, use_container_width=True, hide_index=True)

def show_model_performance():
    """Model performance metrics"""
    
    st.markdown('<div class="section-header">📊 Model Performance Analysis</div>', unsafe_allow_html=True)
    
    try:
        model_comparison = pd.read_csv('model_comparison.csv')
        
        st.markdown("### 🎯 Model Comparison")
        st.dataframe(model_comparison, use_container_width=True, hide_index=True)
        
        best_model = model_comparison.loc[model_comparison['ROC-AUC'].idxmax(), 'Model']
        best_score = model_comparison['ROC-AUC'].max()
        
        st.success(f"🏆 **Best Performing Model:** {best_model} (ROC-AUC: {best_score:.4f})")
        
    except FileNotFoundError:
        st.info("Model performance metrics will be available after training completes.")

def main():
    """Main application function"""
    
    # Load data and model
    df = load_data()
    
    try:
        model, scaler, label_encoders, feature_names = load_model_components()
    except:
        st.error("Unable to load models. Please refresh the page.")
        st.stop()
    
    # Header
    st.markdown('<h1 class="main-title">🎯 Employee Attrition Risk Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Predictive Workforce Intelligence Platform | Palo Alto Networks</p>', unsafe_allow_html=True)
    
    # Sidebar Navigation
    st.sidebar.title("🎛️ Navigation")
    
    page = st.sidebar.radio(
        "📊 Dashboard Sections",
        ["🏠 Dashboard Overview", "📈 Department Analytics", "👤 Employee Profile", 
         "💡 Insights & Actions", "📊 Model Performance"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Quick Stats")
    st.sidebar.metric("Total Employees", len(df))
    
    if 'RiskCategory' in df.columns:
        st.sidebar.metric("High Risk", len(df[df['RiskCategory'] == 'High']))
    if 'AttritionRisk' in df.columns:
        st.sidebar.metric("Avg Risk Score", f"{df['AttritionRisk'].mean():.1%}")
    
    # Page Routing
    if page == "🏠 Dashboard Overview":
        show_dashboard(df)
    elif page == "📈 Department Analytics":
        show_department_analytics(df)
    elif page == "👤 Employee Profile":
        show_employee_profile(df)
    elif page == "💡 Insights & Actions":
        show_insights_actions(df)
    elif page == "📊 Model Performance":
        show_model_performance()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #64748b; padding: 20px;'>
        <p><strong>Employee Attrition Risk Prediction System</strong></p>
        <p>Powered by Machine Learning | Built for Palo Alto Networks</p>
        <p>© 2024 All Rights Reserved</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
