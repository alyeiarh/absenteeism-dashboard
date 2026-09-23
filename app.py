import collections
import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------
# 1. DASHBOARD CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="ABSENTEEISM MANAGEMENT SYSTEM DASHBOARD",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    html, body, [data-testid="stAppViewContainer"], .main {
        overflow: hidden !important;
        height: 100vh !important;
        width: 100vw !important;
    }

    .stApp {
        background: radial-gradient(circle at 75% 15%, #1E0B36 0%, #0D0722 50%, #050311 100%) !important;
        color: #F8FAFC;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .block-container {
        padding-top: 0.3rem !important;
        padding-bottom: 0.1rem !important;
        padding-left: 0.6rem !important;
        padding-right: 0.6rem !important;
        max-width: 100% !important;
    }

    /* KPI CARDS STYLING */
    .kpi-card {
        background: linear-gradient(145deg, rgba(35, 18, 62, 0.9) 0%, rgba(15, 10, 32, 0.95) 100%);
        border: 1.5px solid #00E5FF;
        border-radius: 10px;
        padding: 10px 14px;
        box-shadow: 0 0 10px rgba(0, 229, 255, 0.3), inset 0 0 6px rgba(0, 229, 255, 0.12);
        backdrop-filter: blur(14px);
        min-height: 100px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .kpi-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .kpi-title {
        font-size: 12px;
        color: #CBD5E1;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .status-dot {
        height: 8px;
        width: 8px;
        background-color: #10B981;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 6px #10B981;
    }

    .kpi-value {
        font-size: 22px;
        font-weight: 900;
        color: #00E5FF;
        text-shadow: 0 0 8px rgba(0, 229, 255, 0.8);
        margin: 2px 0;
    }

    .kpi-insight {
        font-size: 12px;
        color: #94A3B8;
        line-height: 1.25;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        padding-top: 4px;
        margin-top: 2px;
    }

    .kpi-insight b {
        color: #10B981;
    }

    /* CARD CONTAINER (white border box) */
    div[data-testid="stVerticalBlockBorderWrapper"],
    div[class*="st-key-card_"] {
        background: linear-gradient(150deg, rgba(32, 17, 58, 0.75) 0%, rgba(13, 8, 28, 0.95) 100%) !important;
        border: 1.2px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 10px !important;
        padding: 8px 10px !important;
        box-shadow: 0 0 10px rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(12px);
        margin-bottom: 0px !important;
    }

    .card-title {
        font-size: 14px;
        font-weight: 800;
        color: #00E5FF;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        padding-bottom: 3px;
        border-bottom: 1.2px solid rgba(0, 229, 255, 0.3);
        text-shadow: 0 0 6px rgba(0, 229, 255, 0.5);
        margin-bottom: 4px;
        line-height: 1.2;
    }

    .layman-insight {
        font-size: 12px;
        color: #F1F5F9;
        background: rgba(255, 255, 255, 0.05);
        padding: 5px 8px;
        border-radius: 5px;
        border-left: 3px solid #00E5FF;
        margin-top: -5px;
        line-height: 1.25;
    }
    .layman-insight b {
        color: #00E5FF;
        font-weight: 700;
    }

    /* PREDICTION ITEM STYLING */
    .inference-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 10px;
        margin-bottom: 6px;
        background: rgba(255, 255, 255, 0.04);
        border-radius: 5px;
        border-left: 3px solid #FF2A85;
        font-size: 13px;
        font-weight: 600;
    }
    .badge-count {
        background: linear-gradient(135deg, #FF2A85 0%, #FF7300 100%);
        color: #FFFFFF;
        padding: 2px 8px;
        border-radius: 8px;
        font-weight: 800;
        font-size: 10.5px;
        box-shadow: 0 0 6px rgba(255, 42, 133, 0.5);
    }

    #MainMenu, footer, header {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# 2. DATA & MODEL LOADING
# ---------------------------------------------------------
@st.cache_data
def load_data():
    file_path = "demo_data.csv"  # synthetic demo data (original dataset is private)
    if not os.path.exists(file_path):
        st.error(f"❌ Dataset `{file_path}` not found!")
        st.stop()
    df = pd.read_csv(file_path)
    if "Work load" in df.columns:
        df["Work load"] = (
            df["Work load"].astype(str).str.replace(",", "").astype(float)
        )
    return df


@st.cache_resource
def load_model_and_features():
    model_file = (
        "knn_model.pkl"
        if os.path.exists("knn_model.pkl")
        else "knn_model.joblib"
    )
    features_file = (
        "model_features.pkl"
        if os.path.exists("model_features.pkl")
        else "model_features.joblib"
    )

    if not os.path.exists(model_file) or not os.path.exists(features_file):
        st.error("❌ Model files missing!")
        st.stop()

    knn_model = joblib.load(model_file)
    model_features = joblib.load(features_file)

    return knn_model, model_features


df = load_data()
best_knn, feature_cols = load_model_and_features()

month_map = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}

# NEW DATA
newData_a = [[11, 8, 2, 1, 180, 51, 18, 42, 205917, 92, 0, 1, 0, 1, 0, 0, 89, 170, 31]]
newData_b = [[22, 7, 6, 1, 361, 52, 3, 28, 239554, 97, 0, 1, 1, 1, 0, 4, 80, 172, 27]]
newData_c = [[15, 12, 3, 2, 160, 12, 14, 34, 280549, 98, 0, 1, 2, 1, 0, 0, 95, 196, 25]]

batch_inputs = np.vstack([newData_a, newData_b, newData_c])
if batch_inputs.shape[1] == len(feature_cols):
    batch_df = pd.DataFrame(batch_inputs, columns=feature_cols)
else:
    batch_df = pd.DataFrame(
        [df[col].median() for col in feature_cols]
    ).T.reindex(index=range(3), method="ffill")

predictions = best_knn.predict(batch_df)
pred_counts = collections.Counter(predictions)

time_slip_today = pred_counts.get(1, 0)
mc_today = pred_counts.get(2, 0)
abnormal_today = pred_counts.get(3, 0)


def apply_lively_theme(fig, x_label, y_label):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=5, r=5, t=5, b=5),
        font=dict(color="#CBD5E1", size=9),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=8, color="#CBD5E1"),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            title=dict(text=x_label, font=dict(size=11, color="#00E5FF")),
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="rgba(255,255,255,0.12)",
        ),
        yaxis=dict(
            title=dict(text=y_label, font=dict(size=11, color="#00E5FF")),
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
        ),
    )
    return fig


# ---------------------------------------------------------
# 3. HEADER & KPI CARDS ROW
# ---------------------------------------------------------
st.markdown(
    """
<div style="text-align: center; margin-bottom: 8px;">
    <h1 style="margin:0; font-weight: 900; color: #00E5FF; text-shadow: 0 0 16px rgba(0,229,255,0.7); letter-spacing: 1px; font-size: 20px;">
        ABSENTEEISM MANAGEMENT SYSTEM DASHBOARD
    </h1>
    <div style="font-size: 11px; color: #94A3B8; margin-top: 2px;">
        Demo version running on synthetic data &middot; the original hackathon dataset is private
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# 1. Target Met Metric
avg_target = df["Hit target"].mean() if "Hit target" in df else 0

# 2. Highest Absence Age Range Metric
if "Age" in df.columns:
    df["Age Group"] = pd.cut(
        df["Age"],
        bins=[18, 30, 40, 50, 65],
        labels=["18-30 Yrs", "31-40 Yrs", "41-50 Yrs", "51+ Yrs"],
    )
    top_age_group = (
        df.groupby("Age Group", observed=False)["Absenteeism time in hours"]
        .sum()
        .idxmax()
    )
    top_age_hrs = (
        df.groupby("Age Group", observed=False)["Absenteeism time in hours"]
        .sum()
        .max()
    )
    avg_age_val = df[df["Age Group"] == top_age_group]["Age"].mean()
else:
    top_age_group, top_age_hrs, avg_age_val = "31-40 Yrs", 0, 36.5

# 3. Drinkers vs Non-Drinkers Absence Hours
drinker_total_hrs = df[df["Social drinker"] == 1][
    "Absenteeism time in hours"
].sum()
nondrinker_total_hrs = df[df["Social drinker"] == 0][
    "Absenteeism time in hours"
].sum()
total_hrs = drinker_total_hrs + nondrinker_total_hrs

drinker_hrs_pct = (drinker_total_hrs / total_hrs) * 100
nondrinker_hrs_pct = (nondrinker_total_hrs / total_hrs) * 100

# 4. Peak Absence Month Metric
if "Month of absence" in df.columns and "Absenteeism time in hours" in df.columns:
    m_df = df[df["Month of absence"] > 0]
    m_sum = m_df.groupby("Month of absence")["Absenteeism time in hours"].sum()
    peak_m_num = m_sum.idxmax()
    peak_m_name = month_map.get(peak_m_num, "N/A")
    peak_m_hrs = m_sum.max()
else:
    peak_m_name, peak_m_hrs = "N/A", 0

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(
        f"""
    <div class="kpi-card">
        <div class="kpi-header">
            <span class="kpi-title">AVG HIT TARGET</span>
        </div>
        <div class="kpi-value">{avg_target:.1f}%</div>
        <div class="kpi-insight">Overall productivity achievement rate.</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with k2:
    st.markdown(
        f"""
    <div class="kpi-card">
        <div class="kpi-header">
            <span class="kpi-title">HIGHEST ABSENCE AGE RANGE</span>
        </div>
        <div class="kpi-value">{top_age_group} ({avg_age_val:.1f} yrs)</div>
        <div class="kpi-insight">Highest total loss with <b>{top_age_hrs:.0f} hrs</b> recorded.</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

avg_absence_hrs = df["Absenteeism time in hours"].mean()
with k3:
    st.markdown(
        f"""
    <div class="kpi-card">
        <div class="kpi-header">
            <span class="kpi-title">AVG ABSENCE PER INCIDENT</span>
        </div>
        <div class="kpi-value">{avg_absence_hrs:.1f} hrs</div>
        <div class="kpi-insight">Average duration of hours lost per absence incident.</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with k4:
    st.markdown(
        f"""
    <div class="kpi-card">
        <div class="kpi-header">
            <span class="kpi-title">PEAK ABSENCE MONTH</span>
        </div>
        <div class="kpi-value">{peak_m_name}</div>
        <div class="kpi-insight">Highest monthly impact with <b>{peak_m_hrs:.0f} total hrs</b> lost.</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 4. MAIN DASHBOARD LAYOUT 
# ---------------------------------------------------------
left_col, mid_col, right_col = st.columns([1.25, 1.25, 1.0])

with left_col:
    # 1. MONTHLY ABSENCE TREND (SUM OF HOURS)
    with st.container(border=True, key="card_1"):
        st.markdown('<div class="card-title">📅 MONTHLY ABSENCE TREND</div>', unsafe_allow_html=True)
        if "Month of absence" in df.columns and "Absenteeism time in hours" in df.columns:
            month_df = df[df["Month of absence"] > 0].copy()
            month_df["Month Name"] = month_df["Month of absence"].map(month_map)
            month_order = list(month_map.values())
            summary_month = (
                month_df.groupby(["Month of absence", "Month Name"])["Absenteeism time in hours"]
                .sum()
                .reset_index(name="Total Absence Hours")
                .sort_values("Month of absence")
            )

            fig_month = px.area(
                summary_month,
                x="Month Name",
                y="Total Absence Hours",
                height=205,
                category_orders={"Month Name": month_order},
            )
            fig_month.update_traces(
                name="Absence Hours",
                line_color="#00E5FF",
                line_width=2,
                fillcolor="rgba(0, 229, 255, 0.2)",
                mode="lines+markers",
                marker=dict(size=4, color="#00E5FF"),
            )
            fig_month = apply_lively_theme(fig_month, "Month", "Total Absence Hours")
            st.plotly_chart(fig_month, width="stretch")

            st.markdown(
                f'<div class="layman-insight"><b>Insight:</b> Absence hours spikes in <b>{peak_m_name}</b> </div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='margin-bottom: 6px;'></div>", unsafe_allow_html=True)

    # 2. WORKLOAD VS TARGET MET
    with st.container(border=True, key="card_2"):
        st.markdown('<div class="card-title">⚖️ WORKLOAD VS TARGET MET</div>', unsafe_allow_html=True)
        if "Month of absence" in df.columns and "Work load" in df.columns:
            m_work = (
                df[df["Month of absence"] > 0]
                .groupby("Month of absence")
                .agg({"Work load": "mean", "Hit target": "mean"})
                .reset_index()
            )
            m_work["Month Name"] = m_work["Month of absence"].map(month_map)

            THRESHOLD = 94.0

            bar_colors = [
                "rgba(16, 185, 129, 0.85)" if val >= THRESHOLD else "rgba(255, 42, 133, 0.85)"
                for val in m_work["Hit target"]
            ]

            marker_colors = [
                "#00E5FF" if val >= THRESHOLD else "#FFE600"
                for val in m_work["Hit target"]
            ]
            marker_sizes = [
                5 if val >= THRESHOLD else 8
                for val in m_work["Hit target"]
            ]

            fig_work = go.Figure()

            fig_work.add_trace(
                go.Bar(
                    x=m_work["Month Name"],
                    y=m_work["Work load"],
                    name="Workload",
                    marker_color=bar_colors,
                    yaxis="y",
                )
            )

            fig_work.add_trace(
                go.Scatter(
                    x=m_work["Month Name"],
                    y=m_work["Hit target"],
                    name="Target Hit %",
                    mode="lines+markers",
                    line=dict(color="#00E5FF", width=2),
                    marker=dict(
                        size=marker_sizes,
                        color=marker_colors,
                        line=dict(width=1.5, color="#FFFFFF")
                    ),
                    yaxis="y2",
                )
            )

            fig_work.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=5, r=5, t=5, b=5),
                font=dict(color="#CBD5E1", size=8.5),
                height=205,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(size=12, color="#FFFFFF"),
                    bgcolor="rgba(0,0,0,0)",
                ),
                xaxis=dict(title=dict(text="Month", font=dict(size=11, color="#00E5FF")), showgrid=False),
                yaxis=dict(title=dict(text="Workload", font=dict(size=11, color="#00E5FF")), showgrid=False),
                yaxis2=dict(
                    title=dict(text="Target %", font=dict(size=11, color="#00E5FF")),
                    overlaying="y",
                    side="right",
                    showgrid=False,
                ),
            )
            st.plotly_chart(fig_work, width="stretch")

            low_months = m_work[m_work["Hit target"] < THRESHOLD]["Month Name"].tolist()
            low_months_str = ", ".join(low_months) if low_months else "None"
            st.markdown(
                f'<div class="layman-insight"><b>Insight:</b> Avg Target drops (&lt;{THRESHOLD:.0f}%) in <b>{low_months_str}</b>.</div>',
                unsafe_allow_html=True,
            )


# ================= LAJUR TENGAH =================
with mid_col:
    # 1. AGE VS BMI HEALTH TREND
    with st.container(border=True, key="card_3"):
        st.markdown('<div class="card-title">🏥 AGE VS BMI HEALTH TREND</div>', unsafe_allow_html=True)
        fig_bmi = px.scatter(
            df,
            x="Age",
            y="Body mass index",
            trendline="ols",
            trendline_color_override="#10B981",
            height=205,
            color_discrete_sequence=["#A855F7"],
        )
        fig_bmi.data[0].name = "BMI"
        if len(fig_bmi.data) > 1:
            fig_bmi.data[1].name = "Trend"

        fig_bmi.add_hline(
            y=25,
            line_dash="dot",
            line_color="#FF7300",
            annotation_text="BMI 25 Limit",
            annotation_position="bottom right",
        )
        fig_bmi = apply_lively_theme(fig_bmi, "Age (Years)", "BMI Score")
        fig_bmi.update_traces(marker=dict(size=4.5, opacity=0.8))
        st.plotly_chart(fig_bmi, width="stretch")

        st.markdown(
            '<div class="layman-insight"><b>Insight:</b> Older employees show higher BMI trends, leading to more MCs.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom: 6px;'></div>", unsafe_allow_html=True)

    # 2. COMMUTE DISTANCE PIE CHART (DONUT)
    with st.container(border=True, key="card_4"):
        st.markdown('<div class="card-title">🚗 COMMUTE DISTANCE SHARE</div>', unsafe_allow_html=True)
        if "Distance from Residence to Work" in df.columns:
            dist_df = df.copy()
            bins = [0, 15, 35, 100]
            labels = ["Near (<15km)", "Mid (15-35km)", "Far (>35km)"]
            dist_df["Distance Bracket"] = pd.cut(
                dist_df["Distance from Residence to Work"],
                bins=bins,
                labels=labels,
            )
            dist_summary = (
                dist_df.groupby("Distance Bracket", observed=False)[
                    "Absenteeism time in hours"
                ]
                .sum()
                .reset_index()
            )

            top_dist_bracket = dist_summary.loc[dist_summary["Absenteeism time in hours"].idxmax()]["Distance Bracket"]

            fig_dist_pie = px.pie(
                dist_summary,
                names="Distance Bracket",
                values="Absenteeism time in hours",
                color="Distance Bracket",
                color_discrete_sequence=["#10B981", "#FF7300", "#FF2A85"],
                hole=0.45,
                height=205,
            )
            fig_dist_pie.update_layout(
                margin=dict(l=5, r=5, t=5, b=5),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.18,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=12, color="#CBD5E1"),
                    bgcolor="rgba(0,0,0,0)",
                ),
            )
            fig_dist_pie.update_traces(
                textposition="inside",
                textinfo="percent",
                insidetextfont=dict(size=9, color="#FFFFFF"),
            )
            st.plotly_chart(fig_dist_pie, width="stretch")

            st.markdown(
                f'<div class="layman-insight"><b>Insight:</b> <b>{top_dist_bracket}</b> account for the largest share absence hours.</div>',
                unsafe_allow_html=True,
            )


with right_col:
    # 1. AI PREDICTION
    with st.container(border=True, key="card_5"):
        st.markdown('<div class="card-title">🤖 TODAY\'S PREDICTION RESULTS</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
        <div style="display: flex; flex-direction: column; gap: 7px; padding: 6px 0;">
            <div class="inference-item"><span>Time Slip (Short Leave)</span><span class="badge-count">{time_slip_today}</span></div>
            <div class="inference-item"><span>Medical Leave (MC)</span><span class="badge-count">{mc_today}</span></div>
            <div class="inference-item"><span>Abnormal Absence</span><span class="badge-count">{abnormal_today}</span></div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)

# 2. ABSENCE CATEGORY BREAKDOWN
    with st.container(border=True, key="card_6"):
        st.markdown(
            '<div class="card-title">📊 ABSENCE CATEGORY BREAKDOWN</div>',
            unsafe_allow_html=True,
        )

        if "Absenteeism time in hours" in df.columns:

            def categorize_by_hours(h):
                if h == 1:
                    return "Time Slip"
                elif h == 2:
                    return "MC"
                else:
                    return "Abnormal"

            cat_df = df.copy()
            cat_df["Category"] = cat_df["Absenteeism time in hours"].apply(
                categorize_by_hours
            )

            cat_summary = (
                cat_df.groupby("Category")
                .size()
                .reset_index(name="Number of Employees")
            )

            category_order = ["Time Slip", "MC", "Abnormal"]
            cat_summary["Category"] = pd.Categorical(
                cat_summary["Category"],
                categories=category_order,
                ordered=True,
            )
            cat_summary = cat_summary.sort_values("Category")

            top_cat = cat_summary.loc[
                cat_summary["Number of Employees"].idxmax()
            ]["Category"]

            fig_cat = px.bar(
                cat_summary,
                x="Category",
                y="Number of Employees",
                color="Category",
                color_discrete_sequence=["#00E5FF", "#A855F7", "#FF2A85"],
                height=280,
            )

            fig_cat = apply_lively_theme(
                fig_cat, "Category", "Number of Employees"
            )

            fig_cat.update_layout(
                legend=dict(
                    font=dict(size=12, color="#CBD5E1") 
                )
            )

            fig_cat.update_traces(width=0.4)

            st.plotly_chart(fig_cat, width="stretch")

            st.markdown(
                f'<div class="layman-insight"><b>Insight:</b> <b>{top_cat}</b> recorded the highest number of employees ({cat_summary["Number of Employees"].max()}).</div>',
                unsafe_allow_html=True,
            )
