import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.impute import KNNImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, StandardScaler
import io
import random

# ========================================================
# 1. INITIALIZATION & UI/UX BOUTIQUE THEMING
# ========================================================
st.set_page_config(page_title="Enterprise Cleansing Matrix", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght=300;400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: #FAF9F6;
    }
    .metric-card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(41, 37, 36, 0.04);
        border: 1px solid #E7E5E4;
        text-align: center;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #292524;
    }
    .metric-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #78716C;
        margin-bottom: 4px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ Enterprise Data Cleansing, Feature Engineering & Leakage Guard Engine")
st.markdown("*A production-grade algorithmic pre-processing pipeline featuring SVD Low-Rank Imputation, LOF Density Outlier Analysis, Regularized Target Encoding, and Polynomial Interaction Generation.*")
st.markdown("---")

# Helper function to generate a multi-type complex messy dataset for demo purposes
def generate_complex_messy_data():
    np.random.seed(101)
    n_samples = 250
    
    # Numerical base signals
    age = np.random.normal(42, 10, n_samples)
    risk_score = (age * 0.4) + np.random.normal(15, 5, n_samples)
    balance = (risk_score * 3500) + np.random.normal(20000, 30000, n_samples)
    collinear_balance = balance * 1.85 + np.random.normal(0, 5, n_samples) # Exact multicollinearity
    zero_var = np.ones(n_samples) * 44.0
    
    # Categorical features with high cardinality & text messiness
    categories = ['Tier-A', 'Tier-B', 'Tier-C', 'Tier-D', 'Unknown_Status', 'Pending_Review', 'Legacy_Hold']
    user_segment = [random.choice(categories) for _ in range(n_samples)]
    
    # Explicit target variable for leakage/encoding testing
    target_probability = np.clip((balance / 300000) + (risk_score / 60) + np.random.normal(0, 0.1, n_samples), 0, 1)
    target = (target_probability > 0.55).astype(int)
    
    df = pd.DataFrame({
        "Age": age,
        "Risk_Score": risk_score,
        "Account_Balance_USD": balance,
        "Collinear_Asset_Proxy": collinear_balance,
        "System_Static_Code": zero_var,
        "Corporate_Tier_Class": user_segment,
        "Conversion_Target": target
    })
    
    # Introduce explicit NaN states
    for col in ["Age", "Risk_Score", "Account_Balance_USD"]:
        df.loc[df.sample(frac=0.06).index, col] = np.nan
        
    # Inject spatial anomalies (Inliers that look normal globally but are anomalous locally)
    df.loc[12, "Age"] = 21.0
    df.loc[12, "Account_Balance_USD"] = 280000.0 # Highly abnormal for a 21-year-old in this covariance matrix
    
    return df

# ========================================================
# 2. INTERACTIVE SIDEBAR PIPELINE CONTROLS
# ========================================================
st.sidebar.header("⚙️ Execution Architecture Matrix")
data_source = st.sidebar.radio("Data Ingestion Stream:", ["Simulate High-Cardinality Messy Pipeline Data", "Upload Production CSV Target File"])

if data_source == "Upload Production CSV Target File":
    uploaded_file = st.sidebar.file_uploader("Upload target dataset", type=["csv"])
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
    else:
        st.info("Awaiting file. Running simulated matrix array baseline.")
        raw_df = generate_complex_messy_data()
else:
    raw_df = generate_complex_messy_data()

cleaned_df = raw_df.copy()
executed_pipeline_steps = []

# Dynamic Target Variable Assignment for Advanced Encoders
numerical_cols_initial = raw_df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols_initial = raw_df.select_dtypes(exclude=[np.number]).columns.tolist()

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Target Mapping Matrix")
target_variable = st.sidebar.selectbox("Identify Pipeline Target Label (y):", raw_df.columns.tolist(), index=len(raw_df.columns)-1)

st.sidebar.markdown("---")
st.sidebar.subheader("1. Low-Rank Matrix Imputation")
impute_strategy = st.sidebar.selectbox("Algorithmic Imputation Vector:", ["SVD Iterative Low-Rank Matrix Completion", "KNN Multi-Variable Network", "Drop Rows"])

st.sidebar.markdown("---")
st.sidebar.subheader("2. Regularized Feature Encoding")
encoding_strategy = st.sidebar.selectbox("High-Cardinality Categorical Path:", ["Regularized M-Estimate Target Encoding", "One-Hot Matrix Expansion"])

st.sidebar.markdown("---")
st.sidebar.subheader("3. Spatial Anomaly Extraction")
outlier_engine = st.sidebar.selectbox("Anomaly Isolation Architecture:", ["Local Outlier Factor (LOF Density Path)", "Isolation Forest (Global Splits)"])
outlier_fraction = st.sidebar.slider("Algorithmic Outlier Threshold (%)", 1, 15, 4) / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("4. Automated Mathematical Feature Generation")
generate_features = st.sidebar.checkbox("Compute Interaction & Non-Linear Features", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("5. Distribution Scaling")
scaler_engine = st.sidebar.selectbox("Feature Regularization Scaler:", ["RobustScaler (IQR Adaptive)", "StandardScaler (Z-Score)"])


# ========================================================
# 3. CORE ALGORITHMIC PROCESSING LAYER
# ========================================================
initial_rows, initial_cols = raw_df.shape
missing_cells_count = raw_df.isna().sum().sum()

# --- STEP 1: VARIANCE & CONSTANT PRUNING ---
constant_features = [col for col in cleaned_df.select_dtypes(include=[np.number]).columns if cleaned_df[col].std() == 0]
if constant_features:
    cleaned_df = cleaned_df.drop(columns=constant_features)
    executed_pipeline_steps.append(f"# Prune Zero-Variance features\ncleaned_df = cleaned_df.drop(columns={constant_features})")

# --- STEP 2: REGULARIZED HIGH-CARDINALITY TARGET ENCODING ---
encoded_categories_logged = []
if categorical_cols_initial and target_variable in cleaned_df.columns:
    if encoding_strategy == "Regularized M-Estimate Target Encoding" and cleaned_df[target_variable].nunique() >= 2:
        global_target_mean = cleaned_df[target_variable].mean()
        m_smoothing = 10.0 # Suppresses encoding variance/leakage
        
        for cat_col in categorical_cols_initial:
            if cat_col == target_variable:
                continue
            stats = cleaned_df.groupby(cat_col)[target_variable].agg(['count', 'mean'])
            smoothed_vals = (stats['count'] * stats['mean'] + m_smoothing * global_target_mean) / (stats['count'] + m_smoothing)
            
            cleaned_df[f"{cat_col}_Target_Encoded"] = cleaned_df[cat_col].map(smoothed_vals).fillna(global_target_mean)
            encoded_categories_logged.append(cat_col)
            cleaned_df = cleaned_df.drop(columns=[cat_col])
            
        executed_pipeline_steps.append(f"# Regularized M-Estimate Target Encoding Matrix Generation\nglobal_mean = cleaned_df['{target_variable}'].mean()\nfor col in {categorical_cols_initial}:\n    stats = cleaned_df.groupby(col)['{target_variable}'].agg(['count', 'mean'])\n    smoothed = (stats['count']*stats['mean'] + 10.0*global_mean)/(stats['count'] + 10.0)\n    cleaned_df[f'{{col}}_Target_Encoded'] = cleaned_df[col].map(smoothed).fillna(global_mean)\ncleaned_df = cleaned_df.drop(columns={categorical_cols_initial})")
    
    elif encoding_strategy == "One-Hot Matrix Expansion":
        targets_to_encode = [c for c in categorical_cols_initial if c != target_variable]
        if targets_to_encode:
            cleaned_df = pd.get_dummies(cleaned_df, columns=targets_to_encode, drop_first=True)
            executed_pipeline_steps.append(f"# One-Hot Expansion\ncleaned_df = pd.get_dummies(cleaned_df, columns={targets_to_encode}, drop_first=True)")

# --- STEP 3: ADVANCED MATRIX IMPUTATION ENGINE ---
if impute_strategy == "Drop Rows":
    cleaned_df = cleaned_df.dropna()
    executed_pipeline_steps.append("cleaned_df = cleaned_df.dropna()")
else:
    numeric_features_processing = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
    if target_variable in numeric_features_processing:
        numeric_features_processing.remove(target_variable)
        
    if numeric_features_processing and cleaned_df[numeric_features_processing].isna().sum().sum() > 0:
        if impute_strategy == "KNN Multi-Variable Network":
            knn_imp = KNNImputer(n_neighbors=5)
            cleaned_df[numeric_features_processing] = knn_imp.fit_transform(cleaned_df[numeric_features_processing])
            executed_pipeline_steps.append(f"from sklearn.impute import KNNImputer\nknn = KNNImputer(n_neighbors=5)\ncleaned_df[{numeric_features_processing}] = knn.fit_transform(cleaned_df[{numeric_features_processing}])")
        
        elif impute_strategy == "SVD Iterative Low-Rank Matrix Completion":
            matrix_completion_df = cleaned_df[numeric_features_processing].copy()
            for col in matrix_completion_df.columns:
                matrix_completion_df[col] = matrix_completion_df[col].fillna(matrix_completion_df[col].mean())
                
            for iteration in range(3):
                U, s, Vt = np.linalg.svd(matrix_completion_df.values, full_matrices=False)
                s_thresholded = np.diag([sv if i < 2 else 0.0 for i, sv in enumerate(s)])
                low_rank_reconstruction = np.dot(U, np.dot(s_thresholded, Vt))
                
                for c_idx, col in enumerate(numeric_features_processing):
                    nan_mask = raw_df[col].isna() if col in raw_df.columns else cleaned_df[col].isna()
                    matrix_completion_df.loc[nan_mask, col] = low_rank_reconstruction[nan_mask, c_idx]
                    
            cleaned_df[numeric_features_processing] = matrix_completion_df
            executed_pipeline_steps.append(f"# SVD Low-Rank Iterative Matrix Factorization Imputation\nfor col in {numeric_features_processing}:\n    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())\nU, s, Vt = np.linalg.svd(cleaned_df[{numeric_features_processing}].values, full_matrices=False)\ns_clean = np.diag([sv if i < 2 else 0 for i, sv in enumerate(s)])\ncleaned_df[{numeric_features_processing}] = np.dot(U, np.dot(s_clean, Vt))")

# --- STEP 4: COLLINEARITY REMOVAL MATRIX (MUTUAL EXCLUSION VIF) ---
collinear_features_dropped = []
numeric_features_now = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
if target_variable in numeric_features_now:
    numeric_features_now.remove(target_variable)

if len(numeric_features_now) > 1:
    correlation_calc_matrix = cleaned_df[numeric_features_now].corr().abs()
    upper_triangular_mask = correlation_calc_matrix.where(np.triu(np.ones(correlation_calc_matrix.shape), k=1).astype(bool))
    collinear_features_dropped = [column for column in upper_triangular_mask.columns if any(upper_triangular_mask[column] > 0.88)]
    cleaned_df = cleaned_df.drop(columns=collinear_features_dropped)
    if collinear_features_dropped:
        executed_pipeline_steps.append(f"# Drop High Multicollinearity features (Corr > 0.88)\ncleaned_df = cleaned_df.drop(columns={collinear_features_dropped})")

# --- STEP 5: AUTOMATED MATHEMATICAL INTERACTION GENERATION ---
features_engineered_count = 0
if generate_features:
    numeric_features_post_corr = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
    if target_variable in numeric_features_post_corr:
        numeric_features_post_corr.remove(target_variable)
        
    if len(numeric_features_post_corr) >= 2:
        variances = cleaned_df[numeric_features_post_corr].var().sort_values(ascending=False)
        top_interact_features = variances.index[:2].tolist()
        
        f1, f2 = top_interact_features[0], top_interact_features[1]
        cleaned_df[f"{f1}_X_{f2}_Interaction"] = cleaned_df[f1] * cleaned_df[f2]
        features_engineered_count += 1
        executed_pipeline_steps.append(f"# Automated Polynomial Interaction Feature Generation\ncleaned_df['{f1}_X_{f2}_Interaction'] = cleaned_df['{f1}'] * cleaned_df['{f2}']")

# --- STEP 6: SPATIAL ANOMALY ISOLATION (LOF VS ISO FOREST) ---
anomalies_removed_count = 0
numeric_features_final = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
if target_variable in numeric_features_final:
    numeric_features_final.remove(target_variable)

if len(numeric_features_final) > 0:
    if outlier_engine == "Local Outlier Factor (LOF Density Path)":
        lof = LocalOutlierFactor(n_neighbors=20, contamination=outlier_fraction)
        outlier_predictions = lof.fit_predict(cleaned_df[numeric_features_final])
        anomalies_removed_count = np.sum(outlier_predictions == -1)
        cleaned_df = cleaned_df[outlier_predictions == 1]
        executed_pipeline_steps.append(f"from sklearn.neighbors import LocalOutlierFactor\nlof = LocalOutlierFactor(n_neighbors=20, contamination={outlier_fraction})\npreds = lof.fit_predict(cleaned_df[{numeric_features_final}])\ncleaned_df = cleaned_df[preds == 1]")
        
    elif outlier_engine == "Isolation Forest (Global Splits)":
        if_forest = IsolationForest(contamination=outlier_fraction, random_state=42)
        outlier_predictions = if_forest.fit_predict(cleaned_df[numeric_features_final])
        anomalies_removed_count = np.sum(outlier_predictions == -1)
        cleaned_df = cleaned_df[outlier_predictions == 1]
        executed_pipeline_steps.append(f"from sklearn.ensemble import IsolationForest\nif_forest = IsolationForest(contamination={outlier_fraction}, random_state=42)\npreds = if_forest.fit_predict(cleaned_df[{numeric_features_final}])\ncleaned_df = cleaned_df[preds == 1]")

# --- STEP 7: STANDARDIZATION / REGULARIZATION SCALING ---
if len(numeric_features_final) > 0:
    if scaler_engine == "RobustScaler (IQR Adaptive)":
        scaler_instance = RobustScaler()
        cleaned_df[numeric_features_final] = scaler_instance.fit_transform(cleaned_df[numeric_features_final])
        executed_pipeline_steps.append(f"from sklearn.preprocessing import RobustScaler\nscaler = RobustScaler()\ncleaned_df[{numeric_features_final}] = scaler.fit_transform(cleaned_df[{numeric_features_final}])")
    elif scaler_engine == "StandardScaler (Z-Score)":
        scaler_instance = StandardScaler()
        cleaned_df[numeric_features_final] = scaler_instance.fit_transform(cleaned_df[numeric_features_final])
        executed_pipeline_steps.append(f"from sklearn.preprocessing import StandardScaler\nscaler = StandardScaler()\ncleaned_df[{numeric_features_final}] = scaler.fit_transform(cleaned_df[{numeric_features_final}])")


# ========================================================
# 4. ENTERPRISE LAYOUT RENDERING COMPONENT
# ========================================================
st.subheader("📊 Algorithmic Output & Pipe Metrics")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(f"<div class='metric-card'><div class='metric-label'>Input Shape Baseline</div><div class='metric-value'>{initial_rows} R × {initial_cols} C</div></div>", unsafe_allow_html=True)
with kpi2:
    st.markdown(f"<div class='metric-card'><div class='metric-label'>Resolved Matrix Gaps</div><div class='metric-value'>{missing_cells_count} Points</div></div>", unsafe_allow_html=True)
with kpi3:
    st.markdown(f"<div class='metric-card' style='border-color: #C2410C;'><div class='metric-label'>Algorithmic Prunings</div><div class='metric-value' style='color: #C2410C;'>-{len(collinear_features_dropped)} Over-Corr / -{anomalies_removed_count} Anomalies</div></div>", unsafe_allow_html=True)
with kpi4:
    st.markdown(f"<div class='metric-card' style='border-color: #15803D;'><div class='metric-label' style='color: #15803D;'>Model-Ready Outputs</div><div class='metric-value' style='color: #15803D;'>{cleaned_df.shape[0]} R × {cleaned_df.shape[1]} C</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# MULTI-DIMENSIONAL PIPELINE VISUAL DIAGNOSTICS
st.subheader("🔬 Strategic Variance Analytics Dashboard")
tab_data, tab_viz = st.tabs(["📋 Structural Matrix Inspection", "📈 Inter-Covariance Density Plots"])

with tab_data:
    v_col1, v_col2 = st.columns(2)
    with v_col1:
        st.markdown("**Raw Input Profile Snapshot**")
        st.dataframe(raw_df.head(10), use_container_width=True)
    with v_col2:
        st.markdown("**Cleaned Engineered Target Output Snapshot**")
        st.dataframe(cleaned_df.head(10), use_container_width=True)

with tab_viz:
    st.markdown("### 📊 Distribution Variance Comparison")
    st.markdown("Select a variable to inspect how the pipeline resolved skewness, handled missing states, and scaled the distribution bounds.")
    
    # Smart structural context linking dictionary
    viz_mapping = {}
    for col in numerical_cols_initial:
        if col in cleaned_df.columns:
            viz_mapping[f"Numerical: {col}"] = (raw_df[col], cleaned_df[col], col)
        elif f"{col}_X_" in "".join(cleaned_df.columns): 
            int_col = [c for c in cleaned_df.columns if f"{col}_X_" in c][0]
            viz_mapping[f"Engineered Interaction (from {col})"] = (raw_df[col], cleaned_df[int_col], int_col)
            
    for col in categorical_cols_initial:
        encoded_name = f"{col}_Target_Encoded"
        if encoded_name in cleaned_df.columns:
            viz_mapping[f"Categorical: {col} ➔ Smoothed Target Encoded"] = (raw_df[col], cleaned_df[encoded_name], encoded_name)
        elif any(c.startswith(f"{col}_") for c in cleaned_df.columns):
            oh_col = [c for c in cleaned_df.columns if c.startswith(f"{col}_")][0]
            viz_mapping[f"Categorical: {col} ➔ One-Hot Matrix Flag"] = (raw_df[col], cleaned_df[oh_col], oh_col)

    if viz_mapping:
        selected_viz_label = st.selectbox("Select Target Variable Vector to Inspect:", list(viz_mapping.keys()))
        raw_series, clean_series, clean_title = viz_mapping[selected_viz_label]
        
        fig_dist = go.Figure()
        
        # Dual-Axis categorical alignment mapping fix
        if isinstance(raw_series.iloc[0], str):
            fig_dist.add_trace(go.Histogram(x=raw_series, name='Original Raw Categories', marker_color='#EF4444', opacity=0.5, xaxis='x2'))
            fig_dist.add_trace(go.Histogram(x=clean_series, name='Processed Encoded Continuum', marker_color='#10B981', opacity=0.7, xaxis='x1'))
            
            fig_dist.update_layout(
                barmode='overlay',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(title=f"Cleaned Values ({clean_title})", side='bottom'),
                xaxis2=dict(title="Original Category Casing", side='top', overlaying='x'),
                yaxis_title="Sample Frequency Density Count",
                height=450
            )
        else:
            # Symmetrical numerical distribution layouts
            fig_dist.add_trace(go.Histogram(x=raw_series, name='Original Raw Distribution', marker_color='#EF4444', opacity=0.5))
            fig_dist.add_trace(go.Histogram(x=clean_series, name='Processed Normalized Vector', marker_color='#10B981', opacity=0.7))
            
            fig_dist.update_layout(
                barmode='overlay', 
                plot_bgcolor='rgba(0,0,0,0)', 
                paper_bgcolor='rgba(0,0,0,0)', 
                xaxis_title=f"Value Scale Comparison (Original vs. Wrapped Output)", 
                yaxis_title="Sample Frequency Density Count", 
                height=400
            )
            
        st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.info("Insufficient parallel dimensions available to construct distribution arrays.")

# REAL-TIME SKLEARN CODE RECONSTRUCTION EXPORTER
st.markdown("---")
st.subheader("⚙️ Programmatic Deployment Code & Artifact Pipeline Export")
exp1, exp2 = st.columns(2)

with exp1:
    st.markdown("### 🎛️ Dynamic Machine Learning Pipeline Script")
    st.markdown("Copy-paste this production-grade execution string straight into script repositories to programmatically replicate this architecture:")
    
    python_pipeline_code = "import pandas as pd\nimport numpy as np\n\n# Load targeted messy matrix file\ncleaned_df = pd.read_csv('production_ingest.csv')\n\n"
    if executed_pipeline_steps:
        python_pipeline_code += "\n".join(executed_pipeline_steps)
    else:
        python_pipeline_code += "# Standard matrix features retained. No adjustment pipelines executed."
        
    st.code(python_pipeline_code, language="python")

with exp2:
    st.markdown("### 💾 Export Modeling Data Asset")
    st.markdown("Deploy this curated data file into production pipelines or remote cloud storage structures.")
    
    csv_mem_io = io.StringIO()
    cleaned_df.to_csv(csv_mem_io, index=False)
    csv_payload = csv_mem_io.getvalue()
    
    st.download_button(
        label="📥 Download Machine Learning-Ready CSV Archive",
        data=csv_payload,
        file_name="algorithmic_production_ready_dataset.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    with st.expander("🔬 Structural Pipeline Deep System Audit"):
        st.markdown(f"""
        *   **Target Vector Labeling Strategy:** Tracking optimizations against target parameter: `{target_variable}`
        *   **High-Cardinality Strategy:** Executed categorical alignment path: `{encoding_strategy}` (Tracked features: `{encoded_categories_logged}`)
        *   **Algorithmic Imputation Strategy:** Executed matrix auto-fill structural layer: `{impute_strategy}`
        *   **Feature Generation Metrics:** Synthesized `{features_engineered_count}` completely new non-linear interaction terms.
        *   **Anomalous Pruning Metrics:** Dropped `{collinear_features_dropped}` multi-collinear columns; pruned `{anomalies_removed_count}` spatial outlier rows via `{outlier_engine}` methodology.
        """)