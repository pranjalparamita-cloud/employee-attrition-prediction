# app.py - UPDATED VERSION

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
    
    /* Logo box */
    .logo-box {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Load Resources
@st.cache_resource
def load_model_components():
    """Load trained model and components"""
    try:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return model, scaler, label_encoders, feature_names
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please run model_training.py first!")
        st.info("Run: `python model_training.py` in your terminal")
        st.stop()

@st.cache_data
def load_data():
    """Load employee data with predictions"""
    try:
        df = pd.read_csv('employee_predictions.csv')
        return df
    except FileNotFoundError:
        st.error("⚠️ Predictions file not found. Please run model_training.py first!")
        st.info("Run: `python model_training.py` in your terminal")
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

# Main Application
def main():
    # Load data
    df = load_data()
    model, scaler, label_encoders, feature_names = load_model_components()
    
    # Header
    st.markdown('<h1 class="main-title">🎯 Employee Attrition Risk Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Predictive Workforce Intelligence Platform | Palo Alto Networks</p>', unsafe_allow_html=True)
    
    # Sidebar Navigation - FIXED VERSION
    st.sidebar.markdown("""
    <div class="logo-box">
        🏢 Palo Alto Networks
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "📊 Navigation",
        ["🏠 Dashboard Overview", "📈 Department Analytics", "👤 Employee Profile", 
         "💡 Insights & Actions", "📊 Model Performance"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Quick Stats")
    st.sidebar.metric("Total Employees", len(df))
    st.sidebar.metric("High Risk", len(df[df['RiskCategory'] == 'High']))
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

def show_dashboard(df):
    """Main dashboard view"""
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_employees = len(df)
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
            <div class="metric-label">Total Employees</div>
            <div class="metric-value">{total_employees}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        high_risk = len(df[df['RiskCategory'] == 'High'])
        pct_high = (high_risk / total_employees * 100)
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <div class="metric-label">High Risk</div>
            <div class="metric-value">{high_risk}</div>
            <div class="metric-label">{pct_high:.1f}% of workforce</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        medium_risk = len(df[df['RiskCategory'] == 'Medium'])
        pct_medium = (medium_risk / total_employees * 100)
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);">
            <div class="metric-label">Medium Risk</div>
            <div class="metric-value">{medium_risk}</div>
            <div class="metric-label">{pct_medium:.1f}% of workforce</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_risk = df['AttritionRisk'].mean()
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);">
            <div class="metric-label">Average Risk Score</div>
            <div class="metric-value">{avg_risk:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
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
            marker_colors=[colors[cat] for cat in risk_counts.index],
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
    
    high_risk_df = df.nlargest(10, 'AttritionRisk')[
        ['EmployeeID', 'Department', 'JobRole', 'YearsAtCompany', 'AttritionRisk', 'RiskCategory']
    ].copy()
    
    high_risk_df['AttritionRisk'] = high_risk_df['AttritionRisk'].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(
        high_risk_df,
        use_container_width=True,
        hide_index=True
    )

def show_department_analytics(df):
    """Department-level analytics view"""
    
    st.markdown('<div class="section-header">🏢 Department-Level Risk Analysis</div>', unsafe_allow_html=True)
    
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
        high_risk_count = len(df_filtered[df_filtered['RiskCategory'] == 'High'])
        st.metric("High Risk", high_risk_count, delta=f"{high_risk_count/len(df_filtered)*100:.1f}%")
    
    with col3:
        avg_risk = df_filtered['AttritionRisk'].mean()
        st.metric("Average Risk", f"{avg_risk:.1%}")
    
    with col4:
        avg_tenure = df_filtered['YearsAtCompany'].mean()
        st.metric("Avg Tenure (Years)", f"{avg_tenure:.1f}")
    
    st.markdown("---")
    
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
    
    # Detailed Department Table
    st.subheader("📋 Detailed Statistics by Job Role")
    
    role_stats = df_filtered.groupby('JobRole').agg({
        'EmployeeID': 'count',
        'AttritionRisk': ['mean', 'min', 'max'],
        'YearsAtCompany': 'mean',
        'MonthlyIncome': 'mean'
    }).round(2)
    
    role_stats.columns = ['Count', 'Avg Risk', 'Min Risk', 'Max Risk', 'Avg Tenure', 'Avg Salary']
    role_stats = role_stats.sort_values('Avg Risk', ascending=False)
    
    st.dataframe(role_stats, use_container_width=True)

def show_employee_profile(df):
    """Individual employee profile view"""
    
    st.markdown('<div class="section-header">👤 Employee Risk Profile</div>', unsafe_allow_html=True)
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        dept_filter = st.selectbox(
            "Department",
            ['All'] + sorted(df['Department'].unique().tolist())
        )
    
    with col2:
        risk_filter = st.selectbox(
            "Risk Category",
            ['All', 'High', 'Medium', 'Low']
        )
    
    with col3:
        role_filter = st.selectbox(
            "Job Role",
            ['All'] + sorted(df['JobRole'].unique().tolist())
        )
    
    # Apply filters
    df_filtered = df.copy()
    if dept_filter != 'All':
        df_filtered = df_filtered[df_filtered['Department'] == dept_filter]
    if risk_filter != 'All':
        df_filtered = df_filtered[df_filtered['RiskCategory'] == risk_filter]
    if role_filter != 'All':
        df_filtered = df_filtered[df_filtered['JobRole'] == role_filter]
    
    # Employee selector
    employee_id = st.selectbox(
        "Select Employee ID",
        df_filtered['EmployeeID'].tolist()
    )
    
    # Get employee data
    employee = df[df['EmployeeID'] == employee_id].iloc[0]
    
    st.markdown("---")
    
    # Employee Header
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.markdown(f"### Employee #{employee['EmployeeID']}")
        st.markdown(f"**Department:** {employee['Department']}")
        st.markdown(f"**Job Role:** {employee['JobRole']}")
    
    with col2:
        st.markdown(f"**Age:** {employee['Age']}")
        st.markdown(f"**Gender:** {employee['Gender']}")
        st.markdown(f"**Years at Company:** {employee['YearsAtCompany']}")
    
    with col3:
        risk_badge = get_risk_badge(employee['RiskCategory'])
        st.markdown(risk_badge, unsafe_allow_html=True)
        st.markdown(f"**Score:** {employee['AttritionRisk']:.1%}")
    
    st.markdown("---")
    
    # Risk Gauge
    col1, col2 = st.columns([1, 2])
    
    with col1:
        fig = create_gauge_chart(employee['AttritionRisk'], "Attrition Risk")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📊 Employee Metrics")
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric("Monthly Income", f"${employee['MonthlyIncome']:,.0f}")
            st.metric("Job Satisfaction", f"{employee['JobSatisfaction']}/4")
        
        with metric_col2:
            st.metric("Years in Role", f"{employee['YearsInCurrentRole']}")
            st.metric("Work-Life Balance", f"{employee['WorkLifeBalance']}/4")
        
        with metric_col3:
            st.metric("Last Promotion", f"{employee['YearsSinceLastPromotion']} yrs ago")
            st.metric("Performance", f"{employee.get('PerformanceRating', 'N/A')}")
    
    # Risk Factors
    st.markdown("#### 🎯 Key Risk Factors")
    
    risk_factors = []
    
    if employee['YearsSinceLastPromotion'] > 3:
        risk_factors.append("⚠️ **Long promotion gap** - No promotion in 3+ years")
    
    if employee['JobSatisfaction'] < 3:
        risk_factors.append("⚠️ **Low job satisfaction** - Below average satisfaction score")
    
    if employee['WorkLifeBalance'] < 3:
        risk_factors.append("⚠️ **Poor work-life balance** - Needs improvement")
    
    if employee['OverTime'] == 'Yes':
        risk_factors.append("⚠️ **Overtime work** - Regularly working extra hours")
    
    if employee.get('PercentSalaryHike', 15) < 12:
        risk_factors.append("⚠️ **Below average salary hike** - Recent raise below company average")
    
    if employee['DistanceFromHome'] > 20:
        risk_factors.append("⚠️ **Long commute** - Distance from home > 20 miles")
    
    if len(risk_factors) > 0:
        for factor in risk_factors:
            st.markdown(f'<div class="info-box">{factor}</div>', unsafe_allow_html=True)
    else:
        st.success("✅ No major risk factors identified for this employee")
    
    # Recommendations
    st.markdown("#### 💡 Recommended Actions")
    
    if employee['AttritionRisk'] > 0.6:
        st.error("""
        **IMMEDIATE ACTION REQUIRED:**
        - Schedule urgent one-on-one meeting with manager
        - Review compensation and benefits package
        - Discuss career development opportunities
        - Consider retention bonus or incentives
        - Explore flexible work arrangements
        """)
    elif employee['AttritionRisk'] > 0.3:
        st.warning("""
        **PROACTIVE ENGAGEMENT NEEDED:**
        - Schedule regular check-in meetings
        - Provide professional development opportunities
        - Review workload and work-life balance
        - Recognize recent achievements
        - Discuss career progression path
        """)
    else:
        st.info("""
        **MAINTAIN ENGAGEMENT:**
        - Continue regular performance reviews
        - Provide growth opportunities
        - Maintain competitive compensation
        - Foster positive work environment
        """)

def show_insights_actions(df):
    """Insights and actionable recommendations"""
    
    st.markdown('<div class="section-header">💡 Strategic Insights & Action Plan</div>', unsafe_allow_html=True)
    
    # Executive Summary
    st.markdown("### 📊 Executive Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_high_risk = len(df[df['RiskCategory'] == 'High'])
        st.metric("Priority Actions", total_high_risk, help="Employees requiring immediate intervention")
    
    with col2:
        avg_cost_per_employee = 75000  # Estimated replacement cost
        potential_cost = total_high_risk * avg_cost_per_employee
        st.metric("Potential Cost at Risk", f"${potential_cost:,.0f}", 
                 help="Estimated replacement cost for high-risk employees")
    
    with col3:
        retention_potential = int(total_high_risk * 0.7)  # 70% retention assumption
        st.metric("Retention Potential", retention_potential,
                 help="Employees that could be retained with intervention")
    
    st.markdown("---")
    
    # Priority Employees
    st.markdown("### 🚨 Priority Intervention List")
    
    priority_df = df[df['RiskCategory'] == 'High'].nlargest(20, 'AttritionRisk')[[
        'EmployeeID', 'Department', 'JobRole', 'YearsAtCompany', 
        'JobSatisfaction', 'WorkLifeBalance', 'AttritionRisk'
    ]].copy()
    
    priority_df['Action Required'] = '🔴 URGENT'
    priority_df['AttritionRisk'] = priority_df['AttritionRisk'].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(priority_df, use_container_width=True, hide_index=True)
    
    # Department-Specific Actions
    st.markdown("### 🎯 Department-Specific Action Plans")
    
    dept_risk = df.groupby('Department').agg({
        'AttritionRisk': 'mean',
        'EmployeeID': 'count'
    }).reset_index()
    dept_risk.columns = ['Department', 'AvgRisk', 'Count']
    dept_risk = dept_risk.sort_values('AvgRisk', ascending=False)
    
    for _, row in dept_risk.head(3).iterrows():
        dept = row['Department']
        risk = row['AvgRisk']
        
        with st.expander(f"🏢 {dept} - Risk Score: {risk:.1%}"):
            dept_employees = df[df['Department'] == dept]
            high_risk_count = len(dept_employees[dept_employees['RiskCategory'] == 'High'])
            
            st.markdown(f"""
            **Department Statistics:**
            - Total Employees: {row['Count']}
            - High Risk Employees: {high_risk_count}
            - Average Risk Score: {risk:.1%}
            """)
            
            st.markdown("**Common Risk Factors:**")
            
            factors = []
            if (dept_employees['YearsSinceLastPromotion'] > 3).sum() > len(dept_employees) * 0.3:
                factors.append("- Promotion delays affecting 30%+ of department")
            if (dept_employees['JobSatisfaction'] < 3).sum() > len(dept_employees) * 0.3:
                factors.append("- Low job satisfaction is prevalent")
            if (dept_employees['WorkLifeBalance'] < 3).sum() > len(dept_employees) * 0.3:
                factors.append("- Work-life balance concerns widespread")
            if (dept_employees['OverTime'] == 'Yes').sum() > len(dept_employees) * 0.4:
                factors.append("- High overtime requirements")
            
            for factor in factors:
                st.markdown(factor)
            
            st.markdown("**Recommended Actions:**")
            st.markdown("""
            1. Conduct department-wide engagement survey
            2. Review and accelerate promotion timelines
            3. Implement workload balancing initiatives
            4. Establish mentorship programs
            5. Increase recognition and rewards
            """)
    
    # General Recommendations
    st.markdown("### 📋 Organization-Wide Recommendations")
    
    tab1, tab2, tab3 = st.tabs(["Immediate (0-30 days)", "Short-term (1-3 months)", "Long-term (3-12 months)"])
    
    with tab1:
        st.markdown("""
        #### Immediate Actions (0-30 Days)
        
        1. **Emergency Retention Meetings**
           - Schedule one-on-one discussions with all high-risk employees
           - Understand specific concerns and challenges
           - Document feedback and action items
        
        2. **Compensation Review**
           - Fast-track salary reviews for high-risk, high-performers
           - Consider retention bonuses for critical roles
           - Benchmark against market rates
        
        3. **Workload Assessment**
           - Review overtime requirements
           - Redistribute work where possible
           - Hire temporary support if needed
        
        4. **Quick Wins**
           - Implement flexible work arrangements
           - Provide immediate recognition
           - Address any urgent workplace concerns
        """)
    
    with tab2:
        st.markdown("""
        #### Short-term Actions (1-3 Months)
        
        1. **Career Development Programs**
           - Launch personalized development plans
           - Identify promotion-ready employees
           - Create clear career progression paths
        
        2. **Manager Training**
           - Train managers on retention strategies
           - Improve one-on-one meeting quality
           - Enhance feedback mechanisms
        
        3. **Work Environment Improvements**
           - Address work-life balance issues
           - Improve team collaboration
           - Enhance workplace amenities
        
        4. **Recognition Programs**
           - Implement peer recognition system
           - Celebrate achievements regularly
           - Tie recognition to career growth
        """)
    
    with tab3:
        st.markdown("""
        #### Long-term Actions (3-12 Months)
        
        1. **Cultural Transformation**
           - Build employee-centric culture
           - Enhance organizational values alignment
           - Improve internal communication
        
        2. **Systematic Improvements**
           - Redesign promotion and compensation frameworks
           - Implement continuous feedback systems
           - Establish employee development infrastructure
        
        3. **Predictive Monitoring**
           - Deploy real-time risk monitoring
           - Automate early warning alerts
           - Integrate with HRIS systems
        
        4. **Strategic Workforce Planning**
           - Build succession planning processes
           - Develop talent pipelines
           - Create knowledge transfer programs
        """)
    
    # ROI Calculator
    st.markdown("### 💰 ROI Projection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Cost of Attrition")
        replacement_cost = st.slider("Average Replacement Cost per Employee ($)", 
                                     30000, 150000, 75000, 5000)
        
        high_risk_count = len(df[df['RiskCategory'] == 'High'])
        total_cost_at_risk = high_risk_count * replacement_cost
        
        st.metric("Total Cost at Risk", f"${total_cost_at_risk:,.0f}")
    
    with col2:
        st.markdown("#### Projected Savings")
        retention_rate = st.slider("Expected Retention Rate (%)", 0, 100, 70, 5)
        
        employees_retained = int(high_risk_count * (retention_rate / 100))
        cost_savings = employees_retained * replacement_cost
        
        st.metric("Projected Annual Savings", f"${cost_savings:,.0f}")
        st.metric("Employees Retained", employees_retained)

def show_model_performance():
    """Model performance metrics"""
    
    st.markdown('<div class="section-header">📊 Model Performance Analysis</div>', unsafe_allow_html=True)
    
    # Load model comparison
    try:
        model_comparison = pd.read_csv('model_comparison.csv')
        
        st.markdown("### 🎯 Model Comparison")
        
        # Display comparison table
        st.dataframe(model_comparison, use_container_width=True, hide_index=True)
        
        # Best model highlight
        best_model = model_comparison.loc[model_comparison['ROC-AUC'].idxmax(), 'Model']
        best_score = model_comparison['ROC-AUC'].max()
        
        st.success(f"🏆 **Best Performing Model:** {best_model} (ROC-AUC: {best_score:.4f})")
        
        # Metrics visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Performance Metrics Comparison")
            
            metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
            
            fig = go.Figure()
            
            for metric in metrics_to_plot:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=model_comparison['Model'],
                    y=model_comparison[metric],
                    text=model_comparison[metric].round(3),
                    textposition='outside'
                ))
            
            fig.update_layout(
                barmode='group',
                height=500,
                xaxis_title="Model",
                yaxis_title="Score",
                yaxis_range=[0, 1.1]
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### ROC-AUC Comparison")
            
            fig = go.Figure(data=[go.Bar(
                x=model_comparison['Model'],
                y=model_comparison['ROC-AUC'],
                marker_color=model_comparison['ROC-AUC'],
                marker_colorscale='Viridis',
                text=[f"{val:.4f}" for val in model_comparison['ROC-AUC']],
                textposition='outside'
            )])
            
            fig.update_layout(
                height=500,
                xaxis_title="Model",
                yaxis_title="ROC-AUC Score",
                showlegend=False,
                yaxis_range=[0, 1.1]
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature Importance
        try:
            model = joblib.load('best_model.pkl')
            feature_names = joblib.load('feature_names.pkl')
            
            if hasattr(model, 'feature_importances_'):
                st.markdown("### 📈 Feature Importance Analysis")
                
                importances = model.feature_importances_
                feature_importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False).head(20)
                
                fig = go.Figure(data=[go.Bar(
                    y=feature_importance_df['Feature'],
                    x=feature_importance_df['Importance'],
                    orientation='h',
                    marker_color=feature_importance_df['Importance'],
                    marker_colorscale='Viridis'
                )])
                
                fig.update_layout(
                    title="Top 20 Most Important Features",
                    xaxis_title="Importance Score",
                    yaxis_title="Feature",
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"""
                **Key Insight:** {feature_importance_df.iloc[0]['Feature']} is the most influential 
                factor in predicting employee attrition, contributing {feature_importance_df.iloc[0]['Importance']:.2%} 
                to the model's predictions.
                """)
        except:
            st.warning("Feature importance analysis not available for this model type.")
        
    except FileNotFoundError:
        st.error("Model comparison file not found. Please run model_training.py first.")
    
    # Model Information
    st.markdown("### ℹ️ Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Model Type:** Ensemble Learning (Random Forest / XGBoost)
        
        **Training Data:**
        - SMOTE applied for class balancing
        - 80-20 train-test split
        - Stratified sampling
        
        **Preprocessing:**
        - Label encoding for binary variables
        - One-hot encoding for categorical variables
        - Standard scaling for numerical features
        """)
    
    with col2:
        st.markdown("""
        **Evaluation Metrics:**
        - **Accuracy:** Overall correctness
        - **Precision:** False positive control
        - **Recall:** Ability to find all attrition cases
        - **F1-Score:** Balance of precision and recall
        - **ROC-AUC:** Overall discrimination ability
        
        **Model Update Frequency:** Recommended quarterly
        """)

# Footer
def add_footer():
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
    add_footer()