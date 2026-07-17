import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from dotenv import load_dotenv
from risk_analyzer import analyze_employee_risk
from llm_advisor import generate_action_plan
import requests
import urllib.parse
import datetime

# Load environment variables
load_dotenv()

# 1. PAGE SETUP (Must be the first Streamlit command executed)
st.set_page_config(
    page_title="Attrition Tracker",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for user profile
if 'user_profile' not in st.session_state:
    st.session_state['user_profile'] = None

# OAuth Google Config
try:
    CLIENT_ID = st.secrets["google"]["client_id"]
    CLIENT_SECRET = st.secrets["google"]["client_secret"]
    REDIRECT_URI = st.secrets["google"]["redirect_uri"]
except Exception:
    CLIENT_ID = CLIENT_SECRET = REDIRECT_URI = ""

# OAuth GitHub Config
try:
    GH_CLIENT_ID = st.secrets["github"]["client_id"]
    GH_CLIENT_SECRET = st.secrets["github"]["client_secret"]
    GH_REDIRECT_URI = st.secrets["github"]["redirect_uri"]
except Exception:
    GH_CLIENT_ID = GH_CLIENT_SECRET = GH_REDIRECT_URI = ""

# Helper to read streamlit secrets safely (even if secrets.toml is missing)
def get_secret(key, default=""):
    try:
        if key in st.secrets:
            return st.secrets[key]
        # Fallback to nested keys (e.g. GOOGLE_CLIENT_ID -> st.secrets["google"]["client_id"])
        parts = key.lower().split("_", 1)
        if len(parts) == 2:
            section, subkey = parts
            if section in st.secrets:
                section_val = st.secrets[section]
                if isinstance(section_val, dict) and subkey in section_val:
                    return section_val[subkey]
                # In some streamlit versions it might act like an attr/dict
                elif hasattr(section_val, "get"):
                    return section_val.get(subkey, default)
        return default
    except Exception:
        return default

def get_google_login_url():
    if not CLIENT_ID: return "#"
    params = {"client_id": CLIENT_ID, "response_type": "code", "redirect_uri": REDIRECT_URI, "scope": "openid email profile", "state": "google"}
    return f"https://accounts.google.com/o/oauth2/v2/auth?{urllib.parse.urlencode(params)}"

def get_github_login_url():
    if not GH_CLIENT_ID: return "#"
    params = {"client_id": GH_CLIENT_ID, "redirect_uri": GH_REDIRECT_URI, "scope": "read:user user:email", "state": "github"}
    return f"https://github.com/login/oauth/authorize?{urllib.parse.urlencode(params)}"

# 2. OAUTH CALLBACK & LOGOUT HANDLER
@st.dialog("👤 User Profile")
def show_profile_popup():
    if st.session_state.get('user_profile'):
        user = st.session_state['user_profile']
        st.markdown(f"**Name:** {user.get('name', 'N/A')}")
        st.markdown(f"**Email:** {user.get('email', 'N/A')}")
        st.markdown(f"**Login Time:** {user.get('login_time', 'N/A')}")


if "action" in st.query_params:
    if st.query_params["action"] == "logout":
        st.session_state['user_profile'] = None
        st.query_params.clear()
        st.rerun()

if "code" in st.query_params and "state" in st.query_params:
    auth_code = st.query_params["code"]
    auth_state = st.query_params["state"]

    if auth_state == "google":
        token_url = "https://oauth2.googleapis.com/token"
        token_data = {
            "code": auth_code,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "redirect_uri": REDIRECT_URI,
            "grant_type": "authorization_code",
        }
        try:
            token_res = requests.post(token_url, data=token_data)
            if token_res.status_code == 200:
                access_token = token_res.json().get("access_token")
                user_res = requests.get("https://www.googleapis.com/oauth2/v1/userinfo", headers={"Authorization": f"Bearer {access_token}"})
                if user_res.status_code == 200:
                    user_data = user_res.json()
                    st.session_state['user_profile'] = {
                        "name": user_data.get("name") or "Authenticated User",
                        "email": user_data.get("email") or "No email provided",
                        "picture": user_data.get("picture"),
                        "login_time": datetime.datetime.now().strftime("%d %b %Y, %I:%M %p")
                    }
        except Exception as e:
            st.error(f"Google authentication failed: {str(e)}")

    elif auth_state == "github":
        # GitHub Token Exchange
        gh_token_url = "https://github.com/login/oauth/access_token"
        gh_headers = {"Accept": "application/json"}
        gh_data = {"client_id": GH_CLIENT_ID, "client_secret": GH_CLIENT_SECRET, "code": auth_code, "redirect_uri": GH_REDIRECT_URI}
        try:
            gh_token_res = requests.post(gh_token_url, headers=gh_headers, data=gh_data)
            if gh_token_res.status_code == 200:
                gh_access_token = gh_token_res.json().get("access_token")
                gh_user_res = requests.get("https://api.github.com/user", headers={"Authorization": f"Bearer {gh_access_token}"})
                if gh_user_res.status_code == 200:
                    gh_user_data = gh_user_res.json()
                    st.session_state['user_profile'] = {
                        "name": gh_user_data.get("name") or gh_user_data.get("login") or "Authenticated User",
                        "email": gh_user_data.get("email") or "No email provided",
                        "picture": gh_user_data.get("avatar_url"),
                        "login_time": datetime.datetime.now().strftime("%d %b %Y, %I:%M %p")
                    }
        except Exception as e:
            st.error(f"GitHub authentication failed: {str(e)}")

    st.query_params.clear()
    st.rerun()

if 'is_authenticated' not in st.session_state:
    st.session_state.is_authenticated = False

@st.cache_resource
def get_mock_model_and_explainer(df):
    features = ['Age', 'MonthlyIncome', 'DistanceFromHome', 'JobSatisfaction']
    available_features = [f for f in features if f in df.columns]
    if not available_features:
        available_features = df.select_dtypes(include=[np.number]).columns.tolist()[:4]
        
    X = df[available_features].fillna(0)
    y = df['Predicted_Risk_Percentage'] >= 50
    
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X, y)
    
    explainer = shap.TreeExplainer(model)
    return model, explainer, available_features

# 2. CUSTOM CSS FOR PREMIUM DARK THEME
st.markdown("""
<style>
    /* Global Background and Layout */
    .stApp {
        background: radial-gradient(circle at top left, #1E293B, #0F172A 40%, #020617);
        color: #F8FAFC;
    }
    
    /* Smooth Scrollbar for a premium feel */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0F172A; 
    }
    ::-webkit-scrollbar-thumb {
        background: #334155; 
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #475569; 
    }
    
    /* Typography Overrides */
    h1, h2, h3, h4, h5, h6 {
        color: #F8FAFC !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    
    /* Gradient Text for Main Title */
    .premium-title {
        background: linear-gradient(90deg, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
    }
    
    /* Glassmorphism Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(11, 17, 32, 0.85) !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Sidebar Typography & Labels */
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] p {
        color: #CBD5E1 !important;
    }
    
    [data-testid="stSidebar"] label {
        color: #94A3B8 !important;
        font-size: 12px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        font-weight: 600 !important;
        margin-bottom: 6px !important;
    }
    
    /* Multiselect Styling (Pills & Dropdown) */
    .stMultiSelect [data-baseweb="tag"] {
        background-color: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #F8FAFC !important;
        border-radius: 6px !important;
    }
    .stMultiSelect [data-baseweb="tag"] span {
        color: #F8FAFC !important;
    }
    
    /* Slider Customization */
    .stSlider [data-baseweb="slider"] > div > div {
        background: linear-gradient(90deg, #38BDF8, #2563EB) !important;
    }
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: #0F172A !important;
        border: 2px solid #38BDF8 !important;
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.4) !important;
        transition: transform 0.1s ease;
    }
    .stSlider [data-baseweb="slider"] [role="slider"]:hover {
        transform: scale(1.15);
    }
    
    /* Glassmorphic KPI Cards with Micro-animations */
    .kpi-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 28px 24px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    /* Subtle glowing top border effect */
    .kpi-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.5), transparent);
        opacity: 0.5;
    }
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.3);
        border-color: rgba(255, 255, 255, 0.1);
        background: rgba(30, 41, 59, 0.6);
    }
    .kpi-title {
        color: #94A3B8;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .kpi-value {
        color: #F8FAFC;
        font-size: 42px;
        font-weight: 800;
        line-height: 1;
        background: linear-gradient(180deg, #FFFFFF, #CBD5E1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* High-Risk KPI Card Styling */
    .kpi-card-warning {
        background: rgba(69, 10, 10, 0.2);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 28px 24px;
        border: 1px solid rgba(248, 113, 113, 0.2);
        box-shadow: 0 8px 32px 0 rgba(220, 38, 38, 0.15);
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }
    .kpi-card-warning::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(248, 113, 113, 0.8), transparent);
        opacity: 0.6;
    }
    .kpi-card-warning:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px 0 rgba(220, 38, 38, 0.3);
        border-color: rgba(248, 113, 113, 0.4);
        background: rgba(69, 10, 10, 0.4);
    }
    .kpi-title-warning {
        color: #FCA5A5;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .kpi-value-warning {
        color: #F87171;
        font-size: 42px;
        font-weight: 800;
        line-height: 1;
        background: linear-gradient(180deg, #FEE2E2, #FCA5A5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* DataFrame Container (Glassmorphic) */
    div[data-testid="stDataFrame"] {
        background: rgba(30, 41, 59, 0.3);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.15);
    }
    
    /* Hide top header bar in streamlit */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    
    /* Tooltip / Hint styling override */
    .stTooltipIcon {
        color: #64748B !important;
    }
    
    /* Fade-in Animation for main content */
    .stApp > header + div {
        animation: fadeIn 0.8s ease-out forwards;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

@st.dialog("Employee Risk Factor Diagnostic", width="large")
def show_risk_dialog(dataframe):
    st.markdown("Enter an Employee ID to analyze why they might leave or stay.")
    raw_emp_id = st.text_input("Enter Employee ID (e.g., EMP-1001)")
    emp_id = ""
    if raw_emp_id:
        clean_id = raw_emp_id.strip()
        if clean_id.isdigit():
            emp_id = f"EMP-{clean_id}"
        else:
            emp_id = clean_id.upper()
    
    if emp_id:
        factors = analyze_employee_risk(emp_id, dataframe)
        
        if factors and (factors['risk'] or factors['retention']):
            st.success(f"Analysis complete for {emp_id}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔻 Attrition Factors")
                if factors['risk']:
                    for factor in factors['risk']:
                        st.markdown(f"""
                        <div style="background-color: rgba(239, 68, 68, 0.15); border-left: 4px solid #EF4444; padding: 12px; border-radius: 4px; margin-bottom: 10px;">
                            <h4 style="margin: 0; color: #FCA5A5; font-size: 14px;">{factor['reason']} ({factor['percentage']}% Impact)</h4>
                            <p style="margin: 5px 0 0 0; font-size: 13px; color: #F8FAFC;">{factor['detail']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No significant risk factors found.")
                    
            with col2:
                st.subheader("🟢 Retention Factors")
                if factors['retention']:
                    for factor in factors['retention']:
                        st.markdown(f"""
                        <div style="background-color: rgba(16, 185, 129, 0.15); border-left: 4px solid #10B981; padding: 12px; border-radius: 4px; margin-bottom: 10px;">
                            <h4 style="margin: 0; color: #6EE7B7; font-size: 14px;">{factor['reason']} ({factor['percentage']}% Impact)</h4>
                            <p style="margin: 5px 0 0 0; font-size: 13px; color: #F8FAFC;">{factor['detail']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No significant retention factors found.")
            

                    
        else:
            st.warning("Employee ID not found or risk is too low to analyze.")

# 3. DATA LOADING AND CACHING
@st.cache_data
def load_data():
    # Load dataset
    df = pd.read_csv('data/Updated_HR_Data.csv')
    
    # Create derived Employee ID (emp-1001, emp-1002, etc.)
    df['Employee ID'] = [f"EMP-{i:04d}" for i in range(1001, 1001 + len(df))]
    
    # Derive Department from one-hot columns
    def get_dept(row):
        if row.get('Department_Research & Development') == True or row.get('Department_Research & Development') == 1:
            return 'Research & Development'
        elif row.get('Department_Sales') == True or row.get('Department_Sales') == 1:
            return 'Sales'
        else:
            return 'Human Resources'
            
    df['Department'] = df.apply(get_dept, axis=1)
    
    # Derive Job Role from one-hot columns
    job_role_cols = [c for c in df.columns if c.startswith('JobRole_')]
    def get_job_role(row):
        for col in job_role_cols:
            if row.get(col) == True or row.get(col) == 1:
                return col.replace('JobRole_', '')
        return 'Healthcare Representative'
        
    df['Job Role'] = df.apply(get_job_role, axis=1)
    
    # Map Actual Attrition to Yes/No
    df['Actual Attrition'] = df['Actual_Attrition'].map({1: 'Yes', 0: 'No'})
    
    return df

# Initialize Data
try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading CSV file: {e}")
    st.info("Please make sure the file is located at `data/Updated_HR_Data.csv` relative to the workspace root.")
    st.stop()

@st.dialog("✨ AI Retention Strategy Plan", width="large")
def show_ai_strategy_dialog(emp_id, dataframe):
    # Resolve search input format (e.g. "1001" -> "EMP-1001")
    resolved_id = emp_id.strip()
    if resolved_id.isdigit():
        resolved_id = f"EMP-{int(resolved_id):04d}"
    elif not resolved_id.upper().startswith("EMP-"):
        resolved_id = f"EMP-{resolved_id}"
        
    emp_rows = dataframe[dataframe['Employee ID'] == resolved_id]
    if emp_rows.empty:
        st.warning(f"Could not find employee '{emp_id}' in the database.")
        return
        
    risk_score = emp_rows['Predicted_Risk_Percentage'].values[0]
    
    if risk_score >= 50:
        with st.spinner(f"Consulting AI HR Advisor for {resolved_id}..."):
            # Get factors for the selected employee
            factors = analyze_employee_risk(resolved_id, dataframe)
            if factors and (factors.get('risk') or factors.get('retention')):
                risk_list = factors.get('risk', [])
                retention_list = factors.get('retention', [])
                
                action_plan = generate_action_plan(risk_list, retention_list)
                if action_plan:
                    st.markdown(action_plan)
                else:
                    st.error("Failed to generate strategy.")
            else:
                st.warning("Could not extract factors for this employee.")
    else:
        st.success("This employee is currently in the Low-Risk zone. No immediate retention strategy is required. Great job keeping them engaged!")
        st.balloons()

# 4. SIDEBAR (FIXED NAVIGATION & FILTERS)
# CSS Fixes for sidebar and main top gap
st.markdown("""
<style>
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0B1120;
        border-right: 1px solid #334155;
    }
    
    /* Remove top gap for enterprise look */
    .block-container {
        padding-top: 2rem !important;
    }
    header[data-testid="stHeader"] {
        display: none !important;
    }
    .stApp {
        margin-top: -50px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Header & Logo
st.sidebar.markdown(
    """
    <div style='display: flex; flex-direction: column; align-items: center; padding-top: 15px; padding-bottom: 25px;'>
        <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 8px;'>
            <svg width="28" height="32" viewBox="0 0 24 28" fill="none" xmlns="http://www.w3.org/2000/svg">
                <!-- Outer silver rim -->
                <path d="M12 1L2 4.5V13.5C2 20.3 6.3 26.1 12 28C17.7 26.1 22 20.3 22 13.5V4.5L12 1Z" fill="#94A3B8"/>
                <!-- Darker blue left side -->
                <path d="M12 2.5V26.2C7.3 24.5 3.5 19.3 3.5 13.5V5.8L12 2.5Z" fill="#3B82F6"/>
                <!-- Lighter blue right side -->
                <path d="M12 2.5V26.2C16.7 24.5 20.5 19.3 20.5 13.5V5.8L12 2.5Z" fill="#38BDF8"/>
            </svg>
            <span style='color: #F8FAFC; font-weight: 800; font-size: 21px; font-family: "Inter", sans-serif; letter-spacing: 0.5px; white-space: nowrap;'>Attrition Tracker</span>
        </div>
        <div style='color: #CBD5E1; font-size: 13px; font-weight: 500; font-family: "Inter", sans-serif;'>Attrition Tracker Intelligence</div>
    </div>
    """, 
    unsafe_allow_html=True
)

st.sidebar.markdown("""
<style>
/* Reduce the gap between elements in the sidebar for a tighter layout */
[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
    gap: 0.5rem !important;
}
/* Ensure the filter header is tight */
h3[style*="🎯 Dashboard Filters"] {
    margin-bottom: 5px !important;
    margin-top: 5px !important;
}
</style>
""", unsafe_allow_html=True)


raw_search_id = st.sidebar.text_input("🔍 Search by ID (e.g. 1001)")
search_employee_id = ""
if raw_search_id:
    clean_id = raw_search_id.strip()
    if clean_id.isdigit():
        search_employee_id = f"EMP-{clean_id}"
    else:
        search_employee_id = clean_id.upper()

# Filters
st.sidebar.markdown("<h3 style='margin-bottom: 15px; font-size: 16px; color: #38BDF8;'>🎯 Dashboard Filters</h3>", unsafe_allow_html=True)

departments = sorted(df['Department'].unique())
selected_depts = st.sidebar.multiselect("Select Departments", options=departments, default=[])

job_roles = sorted(df['Job Role'].unique())
selected_roles = st.sidebar.multiselect("Select Job Roles", options=job_roles, default=[])

attrition_status = st.sidebar.selectbox("Actual Attrition Status", options=["All", "Yes", "No"], index=0)

# Apply filters
filtered_df = df.copy()

if search_employee_id:
    filtered_df = filtered_df[filtered_df['Employee ID'].astype(str).str.contains(search_employee_id.strip(), case=False, na=False)]
else:
    if selected_depts:
        filtered_df = filtered_df[filtered_df['Department'].isin(selected_depts)]

    if selected_roles:
        filtered_df = filtered_df[filtered_df['Job Role'].isin(selected_roles)]

    if attrition_status != "All":
        filtered_df = filtered_df[filtered_df['Actual Attrition'] == attrition_status]

# Resolve and validate selected employee ID in session state
filtered_options = filtered_df['Employee ID'].tolist() if not filtered_df.empty else []
if "selected_emp_id" not in st.session_state or st.session_state.selected_emp_id not in filtered_options:
    st.session_state.selected_emp_id = filtered_options[0] if filtered_options else None

st.sidebar.markdown("---")
st.sidebar.subheader("RETENTION IDEA WITH AI")
if st.sidebar.button("✨ Generate AI Retention Strategy", use_container_width=True):
    if not search_employee_id.strip():
        st.sidebar.warning("Please search an Employee ID first.")
    else:
        show_ai_strategy_dialog(search_employee_id, df)

# Generate OAuth2 urls
google_auth_url = get_google_login_url()
github_auth_url = get_github_login_url()

user_profile = st.session_state.get("user_profile")

with st.sidebar:
    if user_profile:
        col1, col2, col3 = st.columns(3)
        with col1:
            # Native button to trigger popup smoothly without full reload
            if st.button("👤", use_container_width=True, help="User Profile"):
                show_profile_popup()
        with col2:
            # HTML to restore the exact Google Logo styling
            st.markdown("""
            <div style="display: flex; justify-content: center; align-items: center; background-color: #1E293B; border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px; height: 40px; opacity: 0.7; width: 100%;">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" width="20" height="20">
                    <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>
                    <path fill="#4285F4" d="M46.5 24c0-1.61-.15-3.16-.42-4.69H24v8.87h12.66c-.55 2.94-2.21 5.43-4.7 7.09l7.3 5.66c4.27-3.93 6.74-9.72 6.74-16.92z"/>
                    <path fill="#FBBC05" d="M10.54 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.98-6.19z"/>
                    <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.3-5.66c-2.03 1.36-4.63 2.17-8.59 2.17-6.26 0-11.57-4.22-13.46-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
                </svg>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            # HTML to restore the exact GitHub Logo styling
            st.markdown("""
            <div style="display: flex; justify-content: center; align-items: center; background-color: #1E293B; border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px; height: 40px; opacity: 0.7; width: 100%;">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="#F8FAFC">
                    <path d="M12 0C5.37 0 0 5.37 0 12c0 5.3 3.438 9.8 8.205 11.385.6.11.82-.26.82-.577v-2.234c-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.43.372.82 1.102.82 2.222v3.293c0 .319.22.694.825.576C20.565 21.795 24 17.3 24 12c0-6.63-5.37-12-12-12z"/>
                </svg>
            </div>
            """, unsafe_allow_html=True)

        # Sleek Divider
        st.markdown("<hr style='margin: 16px 0; border: none; border-top: 1px solid rgba(255, 255, 255, 0.15); width: 100%;'>", unsafe_allow_html=True)

        # Original HTML Sign Out button to strictly prevent inheriting the large metric card CSS
        st.markdown("""
        <a href="?action=logout" target="_self" style="display: flex; justify-content: center; align-items: center; background-color: #EF4444; color: #F8FAFC; border-radius: 8px; height: 40px; width: 100%; text-decoration: none; font-weight: bold; font-family: sans-serif; font-size: 15px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">🚪 Sign Out</a>
        """, unsafe_allow_html=True)
    else:
        profile_button_html = '<a href="#" target="_self" style="flex: 1; display: flex; justify-content: center; align-items: center; background-color: #1E293B; border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px; height: 40px; text-decoration: none; color: #F8FAFC; font-weight: bold; font-family: sans-serif; font-size: 16px; transition: border-color 0.3s;">P</a>'
        
        google_logo_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" width="20" height="20">
            <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>
            <path fill="#4285F4" d="M46.5 24c0-1.61-.15-3.16-.42-4.69H24v8.87h12.66c-.55 2.94-2.21 5.43-4.7 7.09l7.3 5.66c4.27-3.93 6.74-9.72 6.74-16.92z"/>
            <path fill="#FBBC05" d="M10.54 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 24 0 24c0 3.88.92 7.54 2.56 10.78l7.98-6.19z"/>
            <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.3-5.66c-2.03 1.36-4.63 2.17-8.59 2.17-6.26 0-11.57-4.22-13.46-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
        </svg>"""

        github_logo_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="#F8FAFC">
            <path d="M12 0C5.37 0 0 5.37 0 12c0 5.3 3.438 9.8 8.205 11.385.6.11.82-.26.82-.577v-2.234c-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.43.372.82 1.102.82 2.222v3.293c0 .319.22.694.825.576C20.565 21.795 24 17.3 24 12c0-6.63-5.37-12-12-12z"/>
        </svg>"""

        sidebar_html = f"""
<div style="display: flex; gap: 12px; justify-content: space-between; align-items: center; width: 100%; margin-top: 10px;">
{profile_button_html}
<a href="{google_auth_url}" target="_self" style="flex: 1; display: flex; justify-content: center; align-items: center; background-color: #1E293B; border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px; height: 40px; text-decoration: none; transition: border-color 0.3s;">
{google_logo_svg}
</a>
<a href="{github_auth_url}" target="_self" style="flex: 1; display: flex; justify-content: center; align-items: center; background-color: #1E293B; border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px; height: 40px; text-decoration: none; transition: border-color 0.3s;">
{github_logo_svg}
</a>
</div>
"""
        st.markdown(sidebar_html, unsafe_allow_html=True)

@st.dialog("Total Active Employees", width="large")
def show_active_employees(data_df):
    st.markdown("<p style='color: #94A3B8; font-size: 14px; margin-bottom: 15px;'>Showing all active employees based on current filters.</p>", unsafe_allow_html=True)
    exclude_terms = ['attrition', 'risk', 'category']
    display_cols = [col for col in data_df.columns if not any(term in col.lower() for term in exclude_terms)]
    if 'Employee ID' in display_cols:
        display_cols.insert(0, display_cols.pop(display_cols.index('Employee ID')))
    st.dataframe(data_df[display_cols], use_container_width=True, hide_index=True, height=400)

@st.dialog("Average Attrition Risk", width="large")
def show_risk_details(data_df):
    st.markdown("<p style='color: #94A3B8; font-size: 14px; margin-bottom: 15px;'>Showing all employees sorted by attrition risk.</p>", unsafe_allow_html=True)
    exclude_terms = ['attrition', 'risk', 'category']
    display_cols = [col for col in data_df.columns if not any(term in col.lower() for term in exclude_terms)]
    if 'Employee ID' in display_cols:
        display_cols.insert(0, display_cols.pop(display_cols.index('Employee ID')))
    display_cols.append('Predicted_Risk_Percentage')
    display_df = data_df[display_cols].sort_values(by='Predicted_Risk_Percentage', ascending=False)
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)

@st.dialog("High-Risk Threat Profiles", width="large")
def show_high_risk_profiles(data_df):
    st.markdown("<p style='color: #EF4444; font-size: 14px; margin-bottom: 15px;'>Showing ONLY high-risk employees (Risk &ge; 50%).</p>", unsafe_allow_html=True)
    exclude_terms = ['attrition', 'risk', 'category']
    display_cols = [col for col in data_df.columns if not any(term in col.lower() for term in exclude_terms)]
    if 'Employee ID' in display_cols:
        display_cols.insert(0, display_cols.pop(display_cols.index('Employee ID')))
    display_cols.append('Predicted_Risk_Percentage')
    display_df = data_df[data_df['Predicted_Risk_Percentage'] >= 50][display_cols].sort_values(by='Predicted_Risk_Percentage', ascending=False)
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)

@st.dialog("All Employee Data", width="large")
def show_all_employee_data(data_df):
    st.markdown("<p style='color: #94A3B8; font-size: 14px; margin-bottom: 15px;'>Showing all available employee data based on current filters.</p>", unsafe_allow_html=True)
    display_cols = ['Employee ID', 'Age', 'Department', 'Job Role', 'MonthlyIncome', 'Predicted_Risk_Percentage']
    display_cols = [c for c in display_cols if c in data_df.columns]
    st.dataframe(data_df[display_cols], use_container_width=True, hide_index=True, height=400)

# DASHBOARD MAIN HEADER
st.markdown(
    """
    <div style='margin-bottom: 25px;'>
        <h1 class='premium-title' style='margin-top: 0px; margin-bottom: 5px; font-size: 36px;'>Attrition Tracker</h1>
        <p style='color: #94A3B8; font-size: 16px; margin: 0;'>Predicting churn to build stronger teams.</p>
    </div>
    """, 
    unsafe_allow_html=True
)

if filtered_df.empty:
    if search_employee_id:
        st.warning("⚠️ Employee ID not found!")
    else:
        st.warning("⚠️ No employee profiles match the selected filters. Please adjust your criteria in the sidebar.")
    st.stop()
    
total_employees = len(filtered_df)
avg_risk = filtered_df['Predicted_Risk_Percentage'].mean()
high_risk_count = len(filtered_df[filtered_df['Predicted_Risk_Percentage'] >= 50])

avg_risk_str = f"{avg_risk:.1f}%" if not np.isnan(avg_risk) else "0.0%"

st.markdown("""
<style>
/* --- METRIC CARDS (PRIMARY BUTTONS) --- */
div[data-testid="stButton"] button[kind="primary"] {
    background-color: #1E293B !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    height: 120px !important;
    width: 100% !important;
    padding: 0 !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover {
    border-color: #2563EB !important;
}
div[data-testid="stButton"] button[kind="primary"] p {
    font-size: 38px !important; 
    font-weight: 700 !important;
    color: #F8FAFC !important;
    white-space: pre-line !important;
    margin: 0 !important;
    line-height: 1.5 !important;
    text-align: center !important;
}
div[data-testid="stButton"] button[kind="primary"] p::first-line {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #94A3B8 !important;
    letter-spacing: 1.5px !important;
}

/* --- RED THEME FOR 3RD METRIC CARD --- */
div[data-testid="column"]:nth-of-type(3) div[data-testid="stButton"] button[kind="primary"],
div[data-testid="stColumn"]:nth-of-type(3) div[data-testid="stButton"] button[kind="primary"] {
    background-color: rgba(220, 38, 38, 0.1) !important;
    border: 1px solid rgba(239, 68, 68, 0.4) !important;
}
div[data-testid="column"]:nth-of-type(3) div[data-testid="stButton"] button[kind="primary"]:hover,
div[data-testid="stColumn"]:nth-of-type(3) div[data-testid="stButton"] button[kind="primary"]:hover {
    border-color: #EF4444 !important;
    box-shadow: 0 4px 12px rgba(239, 68, 68, 0.2) !important;
}
div[data-testid="column"]:nth-of-type(3) div[data-testid="stButton"] button[kind="primary"] p::first-line,
div[data-testid="stColumn"]:nth-of-type(3) div[data-testid="stButton"] button[kind="primary"] p::first-line {
    color: #FCA5A5 !important;
}
div[data-testid="column"]:nth-of-type(3) div[data-testid="stButton"] button[kind="primary"] p,
div[data-testid="stColumn"]:nth-of-type(3) div[data-testid="stButton"] button[kind="primary"] p {
    color: #FEF2F2 !important;
}

/* --- NORMAL BUTTONS (SECONDARY - LIKE ANALYZE & SIDEBAR) --- */
div[data-testid="stButton"] button[kind="secondary"] {
    height: auto !important;
    padding: 0.6rem 1.2rem !important;
    border-radius: 8px !important;
    background-color: #1E293B !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    display: inline-flex !important;
    justify-content: center !important;
    align-items: center !important;
    color: #F8FAFC !important;
    transition: all 0.3s ease !important;
}
div[data-testid="stButton"] button[kind="secondary"]:hover {
    border-color: #2563EB !important;
    background-color: #1e293b !important;
}
div[data-testid="stButton"] button[kind="secondary"] p {
    font-size: 16px !important;
    font-weight: 500 !important;
    color: #F8FAFC !important;
    white-space: normal !important;
    margin: 0 !important;
    line-height: normal !important;
}
div[data-testid="stButton"] button[kind="secondary"] p::first-line {
    font-size: 16px !important;
    color: #F8FAFC !important;
    letter-spacing: normal !important;
}
</style>
""", unsafe_allow_html=True)

if st.session_state.get('user_profile'):
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button(f"TOTAL ACTIVE EMPLOYEES\n{total_employees:,}", key="kpi_total_btn", type="primary", use_container_width=True):
            show_active_employees(filtered_df)

    with col2:
        if st.button(f"AVERAGE ATTRITION RISK\n{avg_risk_str}", key="kpi_risk_btn", type="primary", use_container_width=True):
            show_risk_details(filtered_df)

    with col3:
        if st.button(f"⚠️ HIGH-RISK THREAT PROFILES\n{high_risk_count}", key="kpi_high_risk_btn", type="primary", use_container_width=True):
            show_high_risk_profiles(filtered_df)


    import plotly.express as px

    st.markdown("### 📊 Retention Insights")
    chart_col1, chart_col2 = st.columns(2, gap="large")
    with chart_col1:
        st.markdown("##### Attrition Risk by Department")
        dept_risk = filtered_df.groupby('Department')['Predicted_Risk_Percentage'].mean().reset_index()
        dept_risk = dept_risk.sort_values(by='Predicted_Risk_Percentage', ascending=False)
        fig1 = px.bar(dept_risk, x='Department', y='Predicted_Risk_Percentage', color_discrete_sequence=['#2563EB'])
        fig1.update_layout(height=500, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        fig1.update_traces(marker_line_width=0)
        st.plotly_chart(fig1, use_container_width=True)
    with chart_col2:
        st.markdown("##### Average Attrition Risk by Job Role")
        role_risk = filtered_df.groupby('Job Role')['Predicted_Risk_Percentage'].mean().reset_index()
        role_risk = role_risk.sort_values(by='Predicted_Risk_Percentage', ascending=True)
        
        fig2 = px.bar(
            role_risk,
            x='Predicted_Risk_Percentage',
            y='Job Role',
            orientation='h',
            color_discrete_sequence=['#2563EB']
        )
        fig2.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#F8FAFC', family='Inter, sans-serif'),
            xaxis=dict(showgrid=False),
            yaxis=dict(title='')
        )
        st.plotly_chart(fig2, use_container_width=True)
        
    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("🧠 Deep Dive: Individual Attrition Risk Explanation")
    st.markdown("<p style='color: #94A3B8; font-size: 14px; margin-bottom: 15px;'>Select an employee to understand the key factors driving their predicted attrition score.</p>", unsafe_allow_html=True)
    if not filtered_df.empty:
        selected_emp = st.selectbox("Select Employee ID", options=filtered_df['Employee ID'].tolist(), key="selected_emp_id")
        if selected_emp:
            try:
                model, explainer, features = get_mock_model_and_explainer(df)
                emp_idx = df[df['Employee ID'] == selected_emp].index[0]
                emp_row = df.loc[emp_idx, features].fillna(0)
                shap_vals = explainer.shap_values(emp_row.to_frame().T)
                
                if isinstance(explainer.expected_value, (list, np.ndarray)):
                    exp_val = explainer.expected_value[1]
                else:
                    exp_val = explainer.expected_value
                    
                if isinstance(shap_vals, list):
                    sv_array = shap_vals[1][0, :]
                else:
                    sv_array = shap_vals[0, :, 1] if len(shap_vals.shape) == 3 else shap_vals[0, :]
                    
                import plotly.graph_objects as go
                shap_df = pd.DataFrame({'Feature': features, 'SHAP Value': sv_array, 'Feature Value': emp_row.to_frame().T.iloc[0].values})
                shap_df['Abs_SHAP'] = shap_df['SHAP Value'].abs()
                shap_df = shap_df.sort_values(by='Abs_SHAP', ascending=True).tail(5)
                shap_df['Color'] = shap_df['SHAP Value'].apply(lambda x: '#EF4444' if x > 0 else '#10B981')
                shap_df['Label'] = shap_df.apply(lambda row: f"{row['Feature']} = {row['Feature Value']}", axis=1)
                
                fig = go.Figure(go.Bar(
                    x=shap_df['SHAP Value'], y=shap_df['Label'], orientation='h',
                    marker_color=shap_df['Color'], text=shap_df['SHAP Value'].apply(lambda x: f"{x:+.3f}"),
                    textposition='outside', textfont=dict(color='#F8FAFC')
                ))
                fig.update_layout(
                    title='Top Factors Influencing Attrition Risk', title_font=dict(color='#F8FAFC', size=14),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#F8FAFC', family='Inter, sans-serif'),
                    xaxis=dict(showgrid=True, gridcolor='#1E293B', zeroline=True, zerolinecolor='#475569', title='SHAP Value (Impact)'),
                    yaxis=dict(showgrid=False, zeroline=False, title=''), margin=dict(l=10, r=40, t=40, b=30), height=280
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating SHAP explanation: {e}")
    else:
        st.info("No employees match the current filters.")
            
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("🔍 Diagnostic Reason Analyzer")
    if st.button("Analyze Specific Employee Risk"):
        show_risk_dialog(filtered_df)
    st.markdown("### Active Threat Grid")
    st.markdown("The grid displays profiles sorted by highest attrition risk. Low-risk profiles are highlighted in blue/green, while high-risk profiles (>= 50%) require immediate management action.")

    grid_df = filtered_df[['Employee ID', 'Job Role', 'Age', 'MonthlyIncome', 'Actual Attrition', 'Predicted_Risk_Percentage']].copy()
    grid_df = grid_df.sort_values(by='Predicted_Risk_Percentage', ascending=False)
    grid_df.columns = ['Employee ID', 'Job Role', 'Age', 'Monthly Income ($)', 'Actual Attrition', 'Attrition Risk (%)']

    def style_risk_cell(val):
        if val >= 50: return 'background-color: rgba(239, 68, 68, 0.2); color: #EF4444; font-weight: bold; border-left: 3px solid #EF4444;'
        elif val >= 50: return 'background-color: rgba(249, 115, 22, 0.15); color: #F97316;'
        elif val >= 25: return 'background-color: rgba(37, 99, 235, 0.12); color: #3B82F6;'
        else: return 'background-color: rgba(16, 185, 129, 0.12); color: #10B981;'
        
    styled_grid = grid_df.style.map(style_risk_cell, subset=['Attrition Risk (%)']).format({'Monthly Income ($)': '${:,.0f}', 'Attrition Risk (%)': '{:.2f}%'})

    st.dataframe(styled_grid, use_container_width=True, height=400, hide_index=True)
else:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("SECURE PORTAL\n🔒", type="primary", use_container_width=True, key="lock1")
    with col2:
        st.button("COMPANY ANALYTICS\n🔒", type="primary", use_container_width=True, key="lock2")
    with col3:
        st.button("ACCESS RESTRICTED\n🔒", type="primary", use_container_width=True, key="lock3")
