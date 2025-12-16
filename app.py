# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import os
import ssl
import json
import smtplib

from email.message import EmailMessage

from utils import load_dataset
from chatbot import chatbot_response
from alerts import high_missed_districts

#------------------------

#==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Vaccination Dashboard",
    layout="centered"
)

# ==================================================
# SESSION STATE INIT
# ==================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "page" not in st.session_state:
    st.session_state.page = "Login"

# ==================================================
# FILES & CONFIG
# ==================================================
USERS_FILE = "users.json"

SENDER_EMAIL = "yourgmail@gmail.com"        # 🔴 CHANGE
SENDER_PASSWORD = "your_app_password"       # 🔴 CHANGE
ADMIN_NOTIFY_EMAIL = "yourgmail@gmail.com"  # 🔴 CHANGE

# ==================================================
# USER STORAGE HELPERS
# ==================================================


def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)


def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)


# ==================================================
# EMAIL (FAIL-SAFE)
# ==================================================
def send_signup_email(username, email):
    try:
        msg = EmailMessage()
        msg["Subject"] = "New Signup – Vaccination Dashboard"
        msg["From"] = SENDER_EMAIL
        msg["To"] = ADMIN_NOTIFY_EMAIL

        msg.set_content(
            f"""
            New user signed up.

            Username: {username}
            Email: {email}
            """
        )

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(
            "smtp.gmail.com",
            465,
            context=context
        ) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)

    except Exception as e:
        st.warning("Signup successful, but email notification failed.")
        print("Email error:", e)

# ==================================================
# LOGIN PAGE
# ==================================================


def login_page():
    st.title(" Login ")

    img = os.path.join("assets", "login_banner.png")
    if os.path.exists(img):
        st.image(img, width=450)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login", use_container_width=True):
            users = load_users()
            if username in users and users[username]["password"] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid username or password")

# ==================================================
# SIGNUP PAGE
# ==================================================


def signup_page():
    st.title(" Sign Up ")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm = st.text_input("Confirm Password", type="password")

        if st.button("Create Account", use_container_width=True):
            users = load_users()

            if not username or not email or not password:
                st.error("All fields are required")
            elif password != confirm:
                st.error("Passwords do not match")
            elif username in users:
                st.error("Username already exists")
            else:
                users[username] = {
                    "password": password,
                    "email": email,
                    "role": "viewer"
                }
                save_users(users)
                send_signup_email(username, email)

                st.success("Account created successfully")
                st.session_state.page = "Login"
                st.rerun()


# ==================================================
# AUTH GATE
# ==================================================
if not st.session_state.logged_in:
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Login Page"):
            st.session_state.page = "Login"

    with col2:
        if st.button("Signup Page"):
            st.session_state.page = "Signup"

    st.markdown("---")

    if st.session_state.page == "Login":
        login_page()
    else:
        signup_page()

    st.stop()

# ==================================================
# AFTER LOGIN (DASHBOARD PLACEHOLDER)
# ==================================================
st.sidebar.success(f"Logged in as {st.session_state.username}")

if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.page = "Login"
    st.rerun()

# ==================================================
# DASHBOARD CONTENT (REPLACE WITH YOUR REAL DASHBOARD)
# ==================================================


st.info("Login & Signup system is working correctly.")



# --------------------------------------------------
# PAGE CONFIG (ONLY ONCE)
# --------------------------------------------------




##--------------------------------------------------## CONFIG
## --------------------------------------------------
st.set_page_config(
    page_title="Vaccination Monitoring Dashboard",
    page_icon="💉",
    layout="wide"
)

px.defaults.template = "plotly_white"





# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
DATA_PATH = "data/vaccination_dataset_10000_enhanced.xlsx"
df = load_dataset(DATA_PATH)

# --------------------------------------------------
# SIDEBAR: NAVIGATION + FILTERS
# --------------------------------------------------
st.sidebar.title(" Navigation ")


page = st.sidebar.selectbox(
    "Select Page",
    [
        "Dashboard",
        "India Map",
        "Analytics",
        "PHC View",
        "Alerts",
        "Predictions",
        "Chatbot"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader(" Filters")

state = st.sidebar.selectbox("State", ["All"] + sorted(df["state"].unique()))
district = st.sidebar.selectbox(
    "District",
    ["All"] + sorted(df[df["state"] == state]["district"].unique())
    if state != "All" else ["All"]
)
vaccine = st.sidebar.selectbox("Vaccine", ["All"] + sorted(df["vaccine"].unique()))
status = st.sidebar.selectbox("Status", ["All", "completed", "pending", "missed"])

if st.sidebar.button(" 🚪Logout  "):
    st.session_state.logged_in = False
    st.rerun()
# --------------------------------------------------
# APPLY FILTERS (GLOBAL)
# --------------------------------------------------
df_f = df.copy()

if state != "All":
    df_f = df_f[df_f["state"] == state]
if district != "All":
    df_f = df_f[df_f["district"] == district]
if vaccine != "All":
    df_f = df_f[df_f["vaccine"] == vaccine]
if status != "All":
    df_f = df_f[df_f["status"] == status]

# ==================================================
# DASHBOARD
# ==================================================
if page == "Dashboard":
    st.title(" VACCINATION MONITORING DASHBOARD ")

    # KPI METRICS
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(" Children", df_f["child_id"].nunique())
    c2.metric("Completed", (df_f["status"] == "completed").sum())
    c3.metric(" Pending", (df_f["status"] == "pending").sum())
    c4.metric("Missed", (df_f["status"] == "missed").sum())

    # MONTHLY TREND
    st.markdown("## Monthly Vaccination Trend ")
    ts = (
        df_f.groupby(pd.Grouper(key="last_due_date", freq="M"))
        .size()
        .reset_index(name="count")
    )
    st.plotly_chart(
        px.line(ts, x="last_due_date", y="count", markers=True),
        use_container_width=True,
        key="dashboard_trend"
    )

    # PIE + TOP STATES
    st.markdown("##    Status & State Comparison ")
    col1, col2 = st.columns(2)

    with col1:
        status_df = df_f.groupby("status").size().reset_index(name="count")
        st.plotly_chart(
            px.pie(status_df, names="status", values="count", hole=0.4),
            use_container_width=True,
            key="dashboard_status_pie"
        )

    with col2:
        state_bar = (
            df_f.groupby("state")
            .size()
            .reset_index(name="records")
            .sort_values("records", ascending=False)
            .head(10)
        )
        st.plotly_chart(
            px.bar(state_bar, x="records", y="state", orientation="h"),
            use_container_width=True,
            key="dashboard_top_states"
        )

    # STATE STATUS COMPARISON
    st.markdown("## State-wise Status Comparison ")
    state_status = (
        df_f.groupby(["state", "status"])
        .size()
        .reset_index(name="count")
    )
    st.plotly_chart(
        px.bar(
            state_status,
            x="count",
            y="state",
            color="status",
            orientation="h",
            barmode="stack"
        ),
        use_container_width=True,
        key="dashboard_state_status"
    )

    # DISTRICT COMPARISON
    st.markdown("## Top Districts")
    district_df = (
        df_f.groupby("district")
        .size()
        .reset_index(name="records")
        .sort_values("records", ascending=False)
        .head(15)
    )
    st.plotly_chart(
        px.bar(district_df, x="records", y="district", orientation="h"),
        use_container_width=True,
        key="dashboard_districts"
    )
    
    
    st.markdown(" Provider Type Analysis ")

    col_p1, col_p2 = st.columns(2)

    # 1️⃣ Provider-wise total records
    with col_p1:
        provider_df = (
            df_f.groupby("provider_type")
            .size()
            .reset_index(name="records")
        )

        fig_provider_bar = px.bar(
            provider_df,
            x="records",
            y="provider_type",
            orientation="h",
            title="Vaccinations by Provider Type"
        )

        st.plotly_chart(
            fig_provider_bar,
            use_container_width=True,
            key="dashboard_provider_total"
        )
        
            # -----------------------------
    # DOSE-WISE PERSON ANALYSIS
    # -----------------------------
    st.markdown("### Dose-wise Coverage (Unique Persons)")

    dose_df = (
        df_f.groupby("dose_number")["child_id"]
        .nunique()
        .reset_index(name="persons")
        .sort_values("dose_number")
    )

    # -----------------------------
    # KPI CARDS FOR DOSES
    # -----------------------------
    dose_cols = st.columns(len(dose_df))

    for i, row in dose_df.iterrows():
        dose_cols[i].metric(
            f"Dose {int(row['dose_number'])}",
            int(row["persons"])
        )

    # -----------------------------
    # BAR CHART FOR DOSES
    # -----------------------------
    fig_dose = px.bar(
        dose_df,
        x="dose_number",
        y="persons",
        text="persons",
        title="Number of Persons by Dose Number"
    )

    fig_dose.update_traces(textposition="outside")

    st.plotly_chart(
        fig_dose,
        use_container_width=True,
        key="dashboard_dose_distribution"
    )


    # 2️ Provider-wise status distribution
    with col_p2:
        provider_status_df = (
            df_f.groupby(["provider_type", "status"])
            .size()
            .reset_index(name="count")
        )

        fig_provider_status = px.bar(
            provider_status_df,
            x="count",
            y="provider_type",
            color="status",
            orientation="h",
            barmode="stack",
            title="Provider-wise Status Distribution"
        )

        st.plotly_chart(
            fig_provider_status,
            use_container_width=True,
            key="dashboard_provider_status"
        )

    # VACCINE COMPARISON
    st.markdown("### Vaccine-wise Comparison")
    vaccine_df = (
        df_f.groupby(["vaccine", "status"])
        .size()
        .reset_index(name="count")
    )
    st.plotly_chart(
        px.bar(vaccine_df, x="vaccine", y="count", color="status", barmode="group"),
        use_container_width=True,
        key="dashboard_vaccine"
    )
    
        # -----------------------------
    # DROPOUT RATE ANALYSIS
    # -----------------------------
    st.markdown("###  Dose Dropout Rate Analysis")

    dose_persons = (
        df_f.groupby("dose_number")["child_id"]
        .nunique()
        .to_dict()
    )

    d1 = dose_persons.get(1, 0)
    d2 = dose_persons.get(2, 0)
    d3 = dose_persons.get(3, 0)

    dropout_1_2 = ((d1 - d2) / d1 * 100) if d1 else 0
    dropout_2_3 = ((d2 - d3) / d2 * 100) if d2 else 0

    c1, c2 = st.columns(2)
    c1.metric("Dropout Dose 1 → 2", f"{dropout_1_2:.1f}%")
    c2.metric("Dropout Dose 2 → 3", f"{dropout_2_3:.1f}%")

    # -----------------------------
    # DOSE COMPLETION FUNNEL
    # -----------------------------
    st.markdown("###  Dose Completion Funnel")

    funnel_df = pd.DataFrame({
        "Dose": ["Dose 1", "Dose 2", "Dose 3"],
        "Persons": [d1, d2, d3]
    })

    fig_funnel = px.funnel(
        funnel_df,
        x="Persons",
        y="Dose",
        title="Dose Completion Funnel"
    )

    st.plotly_chart(
        fig_funnel,
        use_container_width=True,
        key="dashboard_dose_funnel"
    )

    # -----------------------------
    # DOSE-WISE PROVIDER COMPARISON
    # -----------------------------
    st.markdown("###  Dose-wise Provider Comparison")

    dose_provider_df = (
        df_f.groupby(["dose_number", "provider_type"])["child_id"]
        .nunique()
        .reset_index(name="persons")
    )

    fig_dose_provider = px.bar(
        dose_provider_df,
        x="dose_number",
        y="persons",
        color="provider_type",
        barmode="group",
        title="Dose-wise Persons by Provider Type"
    )

    st.plotly_chart(
        fig_dose_provider,
        use_container_width=True,
        key="dashboard_dose_provider"
    )

        # -----------------------------
    # DOSE-WISE DISTRICT RANKING
    # -----------------------------
    st.markdown("###  Top Districts by Dose Coverage")

    selected_dose = st.selectbox(
        "Select Dose Number",
        sorted(df_f["dose_number"].unique()),
        key="dashboard_dose_selector"
    )

    district_dose_df = (
        df_f[df_f["dose_number"] == selected_dose]
        .groupby("district")["child_id"]
        .nunique()
        .reset_index(name="persons")
        .sort_values("persons", ascending=False)
        .head(10)
    )

    fig_district_dose = px.bar(
        district_dose_df,
        x="persons",
        y="district",
        orientation="h",
        title=f"Top Districts for Dose {selected_dose}"
    )

    st.plotly_chart(
        fig_district_dose,
        use_container_width=True,
        key="dashboard_dose_district"
    )
        # -----------------------------
    # DROPOUT HEATMAP (DISTRICT × DOSE)
    # -----------------------------
    st.markdown("###  Dropout Heatmap (District × Dose)")

    heat_dose_df = (
        df_f.groupby(["district", "dose_number"])["child_id"]
        .nunique()
        .reset_index(name="persons")
    )

    fig_heat_dose = px.density_heatmap(
        heat_dose_df,
        x="dose_number",
        y="district",
        z="persons",
        color_continuous_scale="Reds",
        title="District-wise Dose Coverage Heatmap"
    )

    st.plotly_chart(
        fig_heat_dose,
        use_container_width=True,
        key="dashboard_dose_heatmap"
    )

        # -----------------------------
    # COHORT SURVIVAL CURVE
    # -----------------------------
    st.markdown("###  Cohort Survival Curve (Dose Retention)")

    survival_df = pd.DataFrame({
        "Dose": ["Dose 1", "Dose 2", "Dose 3"],
        "Retention (%)": [
            100,
            (d2 / d1 * 100) if d1 else 0,
            (d3 / d1 * 100) if d1 else 0
        ]
    })

    fig_survival = px.line(
        survival_df,
        x="Dose",
        y="Retention (%)",
        markers=True,
        title="Cohort Retention Across Doses"
    )

    st.plotly_chart(
        fig_survival,
        use_container_width=True,
        key="dashboard_survival_curve"
    )
        # -----------------------------
    # BASE DOSE COUNTS (REQUIRED)
    # -----------------------------
    



# ==================================================
# INDIA MAP
# ==================================================
elif page == "India Map":
    st.title(" State-wise Vaccination Map")

    state_points = pd.DataFrame({
        "state": [
            "Maharashtra","Karnataka","Tamil Nadu","Kerala",
            "Uttar Pradesh","Bihar","West Bengal","Gujarat","Rajasthan"
        ],
        "lat": [19.0,12.97,13.08,10.9,26.8,25.6,22.6,23.0,27.0],
        "lon": [72.5,77.6,78.7,76.3,80.9,85.1,88.3,72.6,74.6]
    })

    state_counts = df_f.groupby("state").size().reset_index(name="records")
    map_df = state_points.merge(state_counts, on="state", how="left").fillna(0)

    st.plotly_chart(
        px.scatter_mapbox(
            map_df,
            lat="lat",
            lon="lon",
            size="records",
            color="records",
            hover_name="state",
            zoom=3.5,
            center={"lat": 22.97, "lon": 78.65},
            mapbox_style="open-street-map"
        ),
        use_container_width=True,
        key="india_map"
    )

# ==================================================
# ANALYTICS
# ==================================================
elif page == "Analytics":
    st.title(" Analytical Insights")

    heat_df = (
        df_f.groupby(["district", "status"])
        .size()
        .reset_index(name="count")
    )
    st.plotly_chart(
        px.density_heatmap(
            heat_df,
            x="status",
            y="district",
            z="count",
            color_continuous_scale="Reds"
        ),
        use_container_width=True,
        key="analytics_heatmap"
    )

    funnel_df = df_f.groupby("status").size().reset_index(name="count")
    st.plotly_chart(
        px.funnel(funnel_df, x="count", y="status"),
        use_container_width=True,
        key="analytics_funnel"
    )

# ==================================================
# PHC VIEW
# ==================================================
elif page == "PHC View":
    st.title(" PHC Performance")

    phc_table = (
        df_f.groupby("phc_code")
        .agg(
            total=("status","count"),
            missed=("status", lambda x: (x=="missed").sum())
        )
        .reset_index()
        .sort_values("missed", ascending=False)
    )
    st.dataframe(phc_table, use_container_width=True)

    phc_bar = (
        df_f.groupby(["phc_code","status"])
        .size()
        .reset_index(name="count")
    )
    st.plotly_chart(
        px.bar(phc_bar, x="count", y="phc_code", color="status", orientation="h"),
        use_container_width=True,
        key="phc_bar"
    )

# ==================================================
# ALERTS
# ==================================================
elif page == "Alerts":
    st.title(" High Risk Districts")

    alerts = high_missed_districts(df_f)
    if alerts.empty:
        st.success(" No high-risk districts")
    else:
        st.warning("⚠ Attention required")
        st.dataframe(alerts, use_container_width=True)

# ==================================================
# PREDICTIONS
# ==================================================
elif page == "Predictions":
    st.title(" Missed Vaccination Risk")

    with open("models/missed_model.pkl","rb") as f:
        saved = pickle.load(f)

    model = saved["model"]
    features = saved["features"]

    sample = df_f.sample(1)
    X = pd.get_dummies(
        sample[["age","dose_number","state","provider_type"]],
        drop_first=True
    ).reindex(columns=features, fill_value=0)

    risk = model.predict_proba(X)[0][1]

    st.markdown(f"""
    **State:** {sample['state'].values[0]}  
    **Dose:** {int(sample['dose_number'].values[0])}  
    **Provider:** {sample['provider_type'].values[0]}

    ###  Missed Risk Probability
    **{risk*100:.2f}%**
    """)

# ==================================================
# CHATBOT
# ==================================================
elif page == "Chatbot":
    st.title("  Vaccination Assistant ")

    q = st.text_input("Ask about vaccination data")
    if q:
        st.info(chatbot_response(df_f, q))
        
        # -----------------------------
    # AUTO DROPOUT INSIGHTS
    # -----------------------------
    st.markdown("###  Auto Dropout Insights")
    dose_persons = (
        df_f.groupby("dose_number")["child_id"]
        .nunique()
        .to_dict()
    )

    d1 = dose_persons.get(1, 0)
    d2 = dose_persons.get(2, 0)
    d3 = dose_persons.get(3, 0)
    insights = []

    if d1 and d2:
        insights.append(
            f"🔹 Dropout from Dose 1 to Dose 2 is **{((d1-d2)/d1)*100:.1f}%**."
        )

    if d2 and d3:
        insights.append(
            f"🔹 Dropout from Dose 2 to Dose 3 is **{((d2-d3)/d2)*100:.1f}%**."
        )

    worst_district = (
        df_f.groupby(["district", "dose_number"])["child_id"]
        .nunique()
        .reset_index()
        .pivot(index="district", columns="dose_number", values="child_id")
        .assign(dropout=lambda x: (x[1] - x.get(3, 0)))
        .sort_values("dropout", ascending=False)
        .head(1)
    )

    if not worst_district.empty:
        insights.append(
            f"🔹 **{worst_district.index[0]}** shows the highest dose dropout."
        )

    for i in insights:
        st.info(i)

    # -----------------------------
    # EXPORT DOSE ANALYSIS PDF
    # -----------------------------
    st.markdown("### 📄 Export Dose Analysis Report")

    if st.button("📄 Download Dose Analysis PDF"):
        dose_insights = [
            f"Dose 1 persons: {d1}",
            f"Dose 2 persons: {d2}",
            f"Dose 3 persons: {d3}",
            f"Dropout 1→2: {((d1-d2)/d1*100):.1f}%" if d1 else "N/A",
            f"Dropout 2→3: {((d2-d3)/d2*100):.1f}%" if d2 else "N/A"
        ]

        # --- PDF GENERATION FUNCTION ---
        def generate_pdf(insights, filename="report.pdf"):
            from fpdf import FPDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Dose Analysis Report", ln=True, align="C")
            pdf.ln(10)
            for line in insights:
                pdf.cell(200, 10, txt=line, ln=True)
            output_path = os.path.join(".", filename)
            pdf.output(output_path)
            return output_path

        file_path = generate_pdf(
            dose_insights,
            filename="dose_analysis_report.pdf"
        )

        with open(file_path, "rb") as f:
            st.download_button(
                label="⬇️ Download PDF",
                data=f,
                file_name="dose_analysis_report.pdf",
                mime="application/pdf"
            )

