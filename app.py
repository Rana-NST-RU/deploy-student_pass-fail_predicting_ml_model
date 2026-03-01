import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import io

from sklearn.metrics import confusion_matrix

# ─────────────────────────────────────────────
# PDF table extraction (optional)
# ─────────────────────────────────────────────
CAMELOT_AVAILABLE = False
try:
    import camelot
    CAMELOT_AVAILABLE = True
except Exception:
    pass

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Intelligent Learning Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif !important; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
h1, h2, h3 { letter-spacing: -0.02em; font-weight: 700; }
hr { opacity: 0.10; }

.metric-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(168,85,247,0.08));
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 14px;
    padding: 18px 20px;
    text-align: center;
    margin-bottom: 8px;
}
.metric-card .val { font-size: 2rem; font-weight: 700; color: #a78bfa; }
.metric-card .lbl { font-size: 0.82rem; opacity: 0.65; margin-top: 4px;
                    letter-spacing: 0.04em; text-transform: uppercase; }

.result-pass {
    background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(5,150,105,0.08));
    border: 1.5px solid rgba(16,185,129,0.4);
    border-radius: 16px; padding: 22px 26px; margin-bottom: 14px;
}
.result-fail {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(220,38,38,0.08));
    border: 1.5px solid rgba(239,68,68,0.4);
    border-radius: 16px; padding: 22px 26px; margin-bottom: 14px;
}
.result-pass .headline { color: #10b981; font-size: 1.8rem; font-weight: 700; }
.result-fail .headline { color: #ef4444; font-size: 1.8rem; font-weight: 700; }
.prob-bar-wrap { background: rgba(255,255,255,0.07); border-radius: 99px; height:10px; margin:8px 0 4px; }
.prob-bar-inner { height:10px; border-radius:99px; }

.info-card {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 16px 18px;
    background: rgba(255,255,255,0.02);
    margin-bottom: 10px;
}
.small-note { font-size: 0.85rem; opacity: 0.6; }

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.55rem 2rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
# Model uses only 3 behavioral features (no total_score)
MODEL_FEATURE_COLS = [
    "weekly_self_study_hours",
    "attendance_percentage",
    "class_participation",
]
INDIVIDUAL_COLS = MODEL_FEATURE_COLS
MODELS_DIR = "models"
MODEL_FILES = {
    "model":  os.path.join(MODELS_DIR, "model.pkl"),
    "scaler": os.path.join(MODELS_DIR, "scaler.pkl"),
    "poly":   os.path.join(MODELS_DIR, "poly.pkl"),
    "kmeans": os.path.join(MODELS_DIR, "kmeans.pkl"),
    "meta":   os.path.join(MODELS_DIR, "meta.pkl"),
}

plt.style.use("dark_background")
PALETTE = ["#6366f1", "#a78bfa", "#10b981", "#f59e0b", "#ef4444", "#06b6d4"]


# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pretrained_model():
    if not all(os.path.exists(v) for v in MODEL_FILES.values()):
        return None
    return {
        "model":  joblib.load(MODEL_FILES["model"]),
        "scaler": joblib.load(MODEL_FILES["scaler"]),
        "poly":   joblib.load(MODEL_FILES["poly"]),
        "kmeans": joblib.load(MODEL_FILES["kmeans"]),
        "meta":   joblib.load(MODEL_FILES["meta"]),
    }


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def validate_dataset(df: pd.DataFrame) -> list:
    return [f"Missing required column: `{col}`" for col in INDIVIDUAL_COLS if col not in df.columns]


def predict_bundle(bundle, X_df: pd.DataFrame):
    """Run prediction using pre-trained model (3 behavioral features)."""
    Xc = X_df.copy()
    for col in MODEL_FEATURE_COLS:
        Xc[col] = pd.to_numeric(Xc[col], errors="coerce").fillna(0.0)
    X_s      = bundle["scaler"].transform(Xc[MODEL_FEATURE_COLS])
    X_p      = bundle["poly"].transform(X_s)
    probas   = bundle["model"].predict_proba(X_p)[:, 1]
    preds    = (probas >= 0.5).astype(int)
    clusters = bundle["kmeans"].predict(X_s)
    return probas, preds, clusters


def student_recommendations(row: pd.Series) -> list:
    recs = []
    if float(row.get("weekly_self_study_hours", 20)) < 15:
        recs.append("📚 Increase weekly self-study to **15–20 hours** with a consistent schedule.")
    if float(row.get("attendance_percentage", 100)) < 80:
        recs.append("🏫 Aim for **80%+ attendance** to reduce missed concepts.")
    if float(row.get("class_participation", 10)) < 5:
        recs.append("🙋 Improve participation: ask questions + attempt in-class problems.")
    if not recs:
        recs.append("🌟 Keep it up! Maintain consistency and add spaced revision weekly.")
    return recs


def render_prediction_card(proba: float, pred: int, cluster: int):
    pct       = proba * 100
    bar_color = "#10b981" if pred == 1 else "#ef4444"
    label     = "PASS ✅" if pred == 1 else "FAIL ❌"
    card_cls  = "result-pass" if pred == 1 else "result-fail"
    st.markdown(f"""
    <div class="{card_cls}">
        <div class="headline">{label}</div>
        <div style="margin-top:10px; font-size:0.95rem; opacity:0.75;">Pass Probability</div>
        <div style="font-size:1.5rem; font-weight:700; color:{bar_color};">{pct:.1f}%</div>
        <div class="prob-bar-wrap">
            <div class="prob-bar-inner" style="width:{pct:.1f}%; background:{bar_color};"></div>
        </div>
        <div class="small-note" style="margin-top:6px;">Performance cluster: {cluster}</div>
    </div>
    """, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    return normalize_columns(pd.read_csv(file))


@st.cache_data(show_spinner=False)
def load_pdf_file(file) -> pd.DataFrame:
    if not CAMELOT_AVAILABLE:
        raise RuntimeError("Camelot not installed. Run: pip install camelot-py[cv] pypdf")
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.getbuffer())
        tmp_path = tmp.name
    try:
        tables = camelot.read_pdf(tmp_path, pages="all", flavor="lattice")
    except Exception:
        tables = camelot.read_pdf(tmp_path, pages="all", flavor="stream")
    if tables is None or tables.n == 0:
        raise RuntimeError("No tables found in PDF.")
    best = max(tables, key=lambda t: t.df.shape[0] * t.df.shape[1]).df
    best.columns = best.iloc[0]
    best = best.iloc[1:].reset_index(drop=True)
    return normalize_columns(best)


# ─────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────
def mpl_hist(ax, data, title, xlabel, color="#6366f1"):
    ax.hist(data.dropna(), bins=25, color=color, alpha=0.85, edgecolor="none", rwidth=0.9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel(xlabel, fontsize=8, alpha=0.7)
    ax.set_ylabel("Count", fontsize=8, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def mpl_bar(ax, labels, values, title, colors=None):
    cols = colors or PALETTE[:len(labels)]
    bars = ax.bar(labels, values, color=cols, alpha=0.9, edgecolor="none", width=0.55)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{int(val):,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def mpl_pie(ax, labels, values, title, colors=None):
    cols = colors or PALETTE[:len(labels)]
    _, _, autotexts = ax.pie(
        values, labels=labels, autopct="%1.1f%%", startangle=90,
        colors=cols, pctdistance=0.82,
        wedgeprops={"linewidth": 2, "edgecolor": "#1a1a2e"},
        textprops={"fontsize": 10}
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)


def mpl_box(ax, df, col, title):
    groups, labels = [], []
    for c in sorted(df["cluster"].unique()):
        groups.append(pd.to_numeric(df[df["cluster"] == c][col], errors="coerce").dropna().values)
        labels.append(f"C{c}")
    bp = ax.boxplot(groups, labels=labels, patch_artist=True,
                    medianprops={"color": "white", "linewidth": 2})
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "scored_df" not in st.session_state:
    st.session_state.scored_df = None
if "file_fp" not in st.session_state:
    st.session_state.file_fp = None


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
bundle = load_pretrained_model()

if bundle:
    meta = bundle["meta"]
else:
    meta = {}


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Learning Analytics")
    st.markdown("<div class='small-note'>Student Performance Prediction System</div>", unsafe_allow_html=True)
    st.divider()

    mode = st.radio(
        "Navigate to",
        ["👤 Individual Predictor", "📁 CSV / PDF Batch Predictor"],
        label_visibility="collapsed",
    )

    st.divider()
    if bundle:
        st.success(
            f"✅ Model ready\n\n"
            f"**{meta.get('best_model_name', 'RandomForest')}** "
            f"· F1: {meta.get('f1', 0):.3f}"
        )
    else:
        st.error("❌ No model found.\nContact administrator.")

    st.divider()
    st.markdown("<div class='small-note'>Trained with SMOTE + RandomForest on 1M student records.</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────
st.markdown("# 📊 Intelligent Learning Analytics")
st.markdown(
    "<div class='small-note'>AI-powered student pass/fail prediction — powered by SMOTE-balanced RandomForest</div>",
    unsafe_allow_html=True
)
st.divider()

# Gate: model must be loaded
if bundle is None:
    st.error("❌ Pre-trained model not found. Please contact the system administrator.")
    st.stop()


# ═══════════════════════════════════════════════════════════════
# PAGE 1 — INDIVIDUAL PREDICTOR  (shown first)
# ═══════════════════════════════════════════════════════════════
if mode == "👤 Individual Predictor":
    st.markdown("### 👤 Individual Student Prediction")
    st.markdown(
        "<div class='small-note'>Enter a student's details to instantly predict their pass/fail result.</div><br>",
        unsafe_allow_html=True
    )

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("#### 📝 Student Details")
        study_hours   = st.slider("📚 Weekly Self-Study Hours",    0,  40,  10, key="ind_study")
        attendance    = st.slider("🏫 Attendance Percentage (%)",  0, 100,  80, key="ind_att")
        participation = st.slider("🙋 Class Participation (0–10)", 0,  10,   5, key="ind_part")
        st.markdown("")
        predict_btn = st.button("🔮 Predict Result", type="primary", use_container_width=True)

    with right:
        if predict_btn:
            inp = pd.DataFrame([{
                "weekly_self_study_hours": study_hours,
                "attendance_percentage":   attendance,
                "class_participation":     participation,
            }])
            probas, preds, clusters = predict_bundle(bundle, inp)
            proba   = float(probas[0])
            pred    = int(preds[0])
            cluster = int(clusters[0])

            st.markdown("#### 🎯 Result")
            render_prediction_card(proba, pred, cluster)

            st.markdown("#### 📋 Recommendations")
            row_ps = pd.Series({
                "weekly_self_study_hours": study_hours,
                "attendance_percentage":   attendance,
                "class_participation":     participation,
            })
            recs = student_recommendations(row_ps)
            if pred == 1:
                st.success("Great performance — keep the momentum!")
            else:
                st.warning("At risk of failing. Key areas to improve:")
            for r in recs:
                st.write(f"• {r}")

            # Position chart
            st.markdown("#### 📍 Position vs Targets")
            fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="#0e1117")
            ax.set_facecolor("#0e1117")
            ax.scatter([study_hours], [attendance], marker="*", s=380,
                       color="#f59e0b", zorder=5, label="This student")
            ax.axhline(80, color="#10b981", linestyle="--", alpha=0.6,
                       linewidth=1.5, label="80% attendance target")
            ax.axvline(15, color="#6366f1", linestyle="--", alpha=0.6,
                       linewidth=1.5, label="15h study target")
            ax.set_xlim(0, 42)
            ax.set_ylim(0, 105)
            ax.set_xlabel("Weekly Self-Study Hours", fontsize=9, alpha=0.7)
            ax.set_ylabel("Attendance %", fontsize=9, alpha=0.7)
            ax.set_title("Student Position vs Targets", fontsize=11, fontweight="bold")
            ax.legend(fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.markdown("""
            <div class="info-card" style="text-align:center; padding:60px 24px;">
            <div style="font-size:3rem; margin-bottom:14px;">🎓</div>
            <div style="font-size:1.05rem; opacity:0.7;">
                Adjust the sliders on the left<br>and click <b>Predict Result</b>.
            </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 2 — CSV / PDF BATCH PREDICTOR
# ═══════════════════════════════════════════════════════════════
elif mode == "📁 CSV / PDF Batch Predictor":
    st.markdown("### 📁 Batch Predictor — CSV / PDF Upload")
    st.markdown(
        "<div class='small-note'>Upload a file of student records — get predictions for everyone, "
        "search by Student ID, filter by result, and download as CSV.</div><br>",
        unsafe_allow_html=True
    )

    colA, colB = st.columns([2, 1], gap="large")
    with colA:
        uploaded_batch = st.file_uploader(
            "Upload CSV or PDF file with student records",
            type=["csv", "pdf"],
            key="batch_upload"
        )
    with colB:
        st.markdown("""
        <div class="info-card">
        <b>Required columns</b><br><br>
        • weekly_self_study_hours<br>
        • attendance_percentage<br>
        • class_participation<br><br>
        <span class="small-note">Optional: total_score, grade, student_id</span>
        </div>
        """, unsafe_allow_html=True)

    if uploaded_batch is None:
        st.info("👆 Upload a CSV or PDF file to run batch predictions.")
        st.stop()

    # Load file
    try:
        with st.spinner("Reading file..."):
            if uploaded_batch.name.lower().endswith(".csv"):
                df_batch = load_csv(uploaded_batch)
            else:
                df_batch = load_pdf_file(uploaded_batch)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        if not CAMELOT_AVAILABLE and uploaded_batch.name.lower().endswith(".pdf"):
            st.info("To enable PDF parsing: `pip install camelot-py[cv] pypdf`")
        st.stop()

    issues = validate_dataset(df_batch)
    if issues:
        st.error("Validation failed:")
        for it in issues:
            st.write(f"• {it}")
        st.stop()

    # Convert to numeric
    for col in INDIVIDUAL_COLS:
        df_batch[col] = pd.to_numeric(df_batch[col], errors="coerce")
    df_batch = df_batch.dropna(subset=INDIVIDUAL_COLS).copy()

    # Run predictions (cache by file fingerprint)
    fp = f"{uploaded_batch.name}-{uploaded_batch.size}"
    if st.session_state.file_fp != fp:
        with st.spinner(f"Running predictions on {len(df_batch):,} students..."):
            probas, preds, clusters = predict_bundle(bundle, df_batch)
            df_result = df_batch.copy()
            df_result["pass_probability_%"] = (probas * 100).round(1)
            df_result["prediction"]          = ["PASS" if p == 1 else "FAIL" for p in preds]
            df_result["cluster"]             = clusters
            st.session_state.scored_df = df_result
            st.session_state.file_fp   = fp

    df_result  = st.session_state.scored_df
    preds_arr  = (df_result["prediction"] == "PASS").astype(int).values
    probas_arr = df_result["pass_probability_%"].values / 100

    # ── SUMMARY METRICS ──────────────────────────
    pass_c   = int(preds_arr.sum())
    fail_c   = int(len(preds_arr) - pass_c)
    avg_prob = float(probas_arr.mean() * 100)

    st.divider()
    m1, m2, m3, m4 = st.columns(4, gap="medium")
    m1.markdown(f"""<div class="metric-card"><div class="val">{len(df_result):,}</div><div class="lbl">Total Students</div></div>""", unsafe_allow_html=True)
    m2.markdown(f"""<div class="metric-card"><div class="val">{pass_c:,}</div><div class="lbl">Predicted Pass</div></div>""", unsafe_allow_html=True)
    m3.markdown(f"""<div class="metric-card"><div class="val">{fail_c:,}</div><div class="lbl">Predicted Fail</div></div>""", unsafe_allow_html=True)
    m4.markdown(f"""<div class="metric-card"><div class="val">{avg_prob:.1f}%</div><div class="lbl">Avg Pass Prob</div></div>""", unsafe_allow_html=True)
    st.markdown("")

    # ── TABS ──────────────────────────────────────
    tSearch, tCharts, tTable = st.tabs(["🔍 Search by Student ID", "📈 Charts", "📋 Full Results"])

    # ── Tab 1: Search by Student ID ──────────────
    with tSearch:
        st.markdown("### 🔍 Search by Student ID")
        has_id = "student_id" in df_result.columns

        if not has_id:
            st.info("`student_id` column not present in your file. Add it to enable this feature.")
        else:
            s_col, f_col = st.columns([2, 1], gap="medium")
            with s_col:
                query_id = st.text_input("Enter Student ID", placeholder="e.g. 1042", key="search_id")
            with f_col:
                filter_result = st.selectbox("Filter by result", ["All", "PASS", "FAIL"], key="filter_res")

            # Apply filters
            view = df_result.copy()
            if filter_result != "All":
                view = view[view["prediction"] == filter_result]

            if query_id.strip():
                view = view[view["student_id"].astype(str) == query_id.strip()]

                if view.empty:
                    st.error(
                        f"No student with ID **{query_id.strip()}** found"
                        + (f" in **{filter_result}** group" if filter_result != "All" else "") + "."
                    )
                else:
                    row         = view.iloc[0]
                    proba_val   = float(row["pass_probability_%"]) / 100
                    pred_val    = 1 if row["prediction"] == "PASS" else 0
                    cluster_val = int(row["cluster"])

                    c1, c2 = st.columns([1, 1], gap="large")
                    with c1:
                        render_prediction_card(proba_val, pred_val, cluster_val)
                        st.markdown("**Input values used for prediction:**")
                        st.dataframe(
                            pd.DataFrame([{c: row[c] for c in INDIVIDUAL_COLS if c in row}]),
                            use_container_width=True
                        )
                    with c2:
                        st.markdown("**📋 Recommendations**")
                        recs = student_recommendations(row)
                        if pred_val == 1:
                            st.success("Predicted PASS — maintain consistency!")
                        else:
                            st.warning("Predicted FAIL — focus on these areas:")
                        for r in recs:
                            st.write(f"• {r}")
            else:
                # No specific ID typed — show filtered list
                n = len(view)
                group_label = f" · {filter_result}" if filter_result != "All" else ""
                st.markdown(f"Showing **{n:,}** students{group_label}")
                disp_cols = (
                    ["student_id", "prediction", "pass_probability_%", "cluster"] + INDIVIDUAL_COLS
                )
                disp_cols = [c for c in disp_cols if c in view.columns]
                st.dataframe(view[disp_cols].head(200), use_container_width=True)
                if n > 200:
                    st.caption(f"Showing first 200 of {n:,}. Use the Full Results tab to see all.")

    # ── Tab 2: Charts ──────────────────────────────
    with tCharts:
        st.markdown("### 📈 Prediction Analytics")

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5, 3.5), facecolor="#0e1117")
            ax.set_facecolor("#0e1117")
            mpl_pie(ax, ["Pass", "Fail"], [pass_c, fail_c],
                    "Pass / Fail Split", colors=["#10b981", "#ef4444"])
            plt.tight_layout(); st.pyplot(fig); plt.close()
        with c2:
            fig2, ax2 = plt.subplots(figsize=(5, 3.5), facecolor="#0e1117")
            ax2.set_facecolor("#0e1117")
            ax2.hist(probas_arr * 100, bins=30, color="#6366f1", alpha=0.85, edgecolor="none")
            ax2.axvline(50, color="#f59e0b", linestyle="--", linewidth=1.5, label="50% threshold")
            ax2.set_title("Pass Probability Distribution", fontsize=11, fontweight="bold")
            ax2.set_xlabel("Pass Probability (%)", fontsize=9, alpha=0.7)
            ax2.legend(fontsize=8)
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            plt.tight_layout(); st.pyplot(fig2); plt.close()

        st.markdown("")

        # Feature histograms
        fig3, axes3 = plt.subplots(1, 3, figsize=(15, 3.5), facecolor="#0e1117")
        for ax, col, color in zip(axes3, INDIVIDUAL_COLS, PALETTE):
            ax.set_facecolor("#0e1117")
            mpl_hist(ax, df_result[col], col.replace("_", " ").title(), col, color=color)
        plt.tight_layout(); st.pyplot(fig3); plt.close()

        st.markdown("")

        # Cluster distribution
        cc = df_result["cluster"].value_counts().sort_index()
        fig4, ax4 = plt.subplots(figsize=(6, 3.5), facecolor="#0e1117")
        ax4.set_facecolor("#0e1117")
        mpl_bar(ax4, [f"Cluster {i}" for i in cc.index], cc.values.tolist(),
                "Student Cluster Distribution", colors=PALETTE)
        plt.tight_layout(); st.pyplot(fig4); plt.close()

    # ── Tab 3: Full Results Table + Download ───────
    with tTable:
        st.markdown(f"#### 📋 All Predictions — {len(df_result):,} students")

        fc1, fc2 = st.columns([3, 1], gap="medium")
        with fc2:
            tbl_filter = st.selectbox("Filter", ["All", "PASS", "FAIL"], key="tbl_filter")

        view_tbl = df_result if tbl_filter == "All" else df_result[df_result["prediction"] == tbl_filter]
        st.dataframe(view_tbl, use_container_width=True)

        csv_buf = io.BytesIO()
        view_tbl.to_csv(csv_buf, index=False)
        csv_buf.seek(0)
        st.download_button(
            label=f"⬇️ Download {tbl_filter} Results as CSV",
            data=csv_buf,
            file_name=f"predictions_{tbl_filter.lower()}.csv",
            mime="text/csv",
            type="primary",
        )