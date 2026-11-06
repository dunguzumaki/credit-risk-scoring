"""
Streamlit app thẩm định rủi ro tín dụng.

Dùng model THẬT từ models/credit_risk_artifacts.pkl (xuất từ notebook Colab
hoặc từ `python train.py`). Không có chế độ giả lập: thiếu file model thì app
báo lỗi và hướng dẫn cách tạo.

Chạy:  streamlit run app.py
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.features import RAW_COLUMNS, apply_imputation_and_feature_engineering

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
METRICS_PATH = os.path.join(BASE_DIR, "metrics_summary.json")
ARTIFACT_CANDIDATES = [
    os.path.join(BASE_DIR, "models", "credit_risk_artifacts.pkl"),
    os.path.join(BASE_DIR, "credit_risk_artifacts.pkl"),  # vị trí mặc định khi tải từ Colab về
]

st.set_page_config(page_title="Credit Risk Scoring", page_icon="🏦", layout="wide")


# ----------------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    for path in ARTIFACT_CANDIDATES:
        if os.path.exists(path):
            return joblib.load(path), path
    return None, None


@st.cache_data
def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_resource
def load_explainer(_artifacts):
    """SHAP TreeExplainer cho XGBoost; nếu lỗi tương thích thì dùng LightGBM (giống notebook)."""
    import shap

    try:
        return shap.TreeExplainer(_artifacts["xgb_model"]), "XGBoost (Tuned)"
    except Exception:  # noqa: BLE001
        return shap.TreeExplainer(_artifacts["lgb_model"]), "LightGBM (Tuned)"


artifacts, artifact_path = load_artifacts()
metrics = load_metrics()

st.title("🏦 Credit Risk Scoring: Dự báo khả năng vỡ nợ")
st.caption("XGBoost + LightGBM + Random Forest · Tối ưu ngưỡng theo chi phí · Giải thích bằng SHAP")

if artifacts is None:
    st.error(
        "Chưa tìm thấy file model `credit_risk_artifacts.pkl`.\n\n"
        "Cách 1: chạy `python train.py` (tự tải dữ liệu và train khoảng 1 phút trên CPU).\n\n"
        "Cách 2: tải file `credit_risk_artifacts.pkl` từ Colab về và đặt vào thư mục `models/`."
    )
    st.stop()

feature_names = artifacts["feature_names"]
preprocessor = artifacts["preprocessor"]
w_xgb, w_lgb, w_rf = artifacts["voting_weights"]
cost_threshold = float(artifacts["best_cost_threshold"])
f1_threshold = float(artifacts["best_f1_threshold"])

# ----------------------------------------------------------------------------
# Sidebar: hồ sơ
# ----------------------------------------------------------------------------
st.sidebar.header("Hồ sơ vay")
person_age = st.sidebar.number_input("Tuổi", 18, 100, 25)
person_income = st.sidebar.number_input("Thu nhập năm (USD)", 1000, 2_000_000, 45_000, step=1000)
emp_unknown = st.sidebar.checkbox("Không rõ số năm đi làm")
person_emp_length = None if emp_unknown else st.sidebar.number_input("Số năm đi làm", 0.0, 60.0, 3.0, step=0.5)
person_home_ownership = st.sidebar.selectbox("Nhà ở", ["RENT", "MORTGAGE", "OWN", "OTHER"])
loan_intent = st.sidebar.selectbox(
    "Mục đích vay",
    ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
)
loan_grade = st.sidebar.selectbox("Hạng khoản vay (A tốt nhất → G xấu nhất)", list("ABCDEFG"), index=1)
loan_amnt = st.sidebar.number_input("Số tiền vay (USD)", 500, 100_000, 8_000, step=500)
rate_unknown = st.sidebar.checkbox("Không rõ lãi suất (điền theo median của hạng vay)")
loan_int_rate = None if rate_unknown else st.sidebar.number_input("Lãi suất (%/năm)", 4.0, 30.0, 11.0, step=0.1)
cb_default = st.sidebar.selectbox("Từng có nợ xấu?", ["N", "Y"])
cb_hist = st.sidebar.number_input("Số năm lịch sử tín dụng", 1, 40, 4)

st.sidebar.header("Ngưỡng quyết định")
th_mode = st.sidebar.radio(
    "Chọn ngưỡng",
    [f"Tối ưu chi phí ({cost_threshold:.2f})", f"Tối ưu F1 ({f1_threshold:.2f})", "Mặc định (0.50)", "Tự chọn"],
)
if th_mode.startswith("Tối ưu chi phí"):
    threshold = cost_threshold
elif th_mode.startswith("Tối ưu F1"):
    threshold = f1_threshold
elif th_mode.startswith("Mặc định"):
    threshold = 0.50
else:
    threshold = st.sidebar.slider("Ngưỡng", 0.05, 0.95, 0.30, 0.01)
st.sidebar.caption(
    "Ngưỡng được tối ưu trên tập Validation cho xác suất của **Soft-Voting**, "
    "nên quyết định cuối cùng dùng mô hình Soft-Voting."
)

# ----------------------------------------------------------------------------
# Suy luận
# ----------------------------------------------------------------------------
raw = pd.DataFrame([{
    "person_age": person_age,
    "person_income": person_income,
    "person_home_ownership": person_home_ownership,
    "person_emp_length": np.nan if person_emp_length is None else person_emp_length,
    "loan_intent": loan_intent,
    "loan_grade": loan_grade,
    "loan_amnt": loan_amnt,
    "loan_int_rate": np.nan if loan_int_rate is None else loan_int_rate,
    # Trong dataset gốc cột này đã làm tròn 2 chữ số
    "loan_percent_income": round(loan_amnt / person_income, 2),
    "cb_person_default_on_file": cb_default,
    "cb_person_cred_hist_length": cb_hist,
}])[RAW_COLUMNS]

fe = apply_imputation_and_feature_engineering(
    raw,
    artifacts["grade_int_rate_median"],
    artifacts["global_int_rate_median"],
    artifacts["emp_length_median"],
)
fe = fe[list(preprocessor.feature_names_in_)]
X = pd.DataFrame(preprocessor.transform(fe), columns=feature_names)

p_xgb = float(artifacts["xgb_model"].predict_proba(X)[0, 1])
p_lgb = float(artifacts["lgb_model"].predict_proba(X)[0, 1])
p_rf = float(artifacts["rf_model"].predict_proba(X)[0, 1])
p_vote = w_xgb * p_xgb + w_lgb * p_lgb + w_rf * p_rf
p_stack = float(artifacts["meta_learner"].predict_proba(np.array([[p_xgb, p_lgb, p_rf]]))[0, 1])

tab1, tab2, tab3, tab4 = st.tabs(["🎯 Thẩm định", "📊 So sánh mô hình", "💰 Ngưỡng & chi phí", "🔍 SHAP & EDA"])

# ----------------------------------------------------------------------------
with tab1:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Kết quả")
        st.metric("Xác suất vỡ nợ (Soft-Voting)", f"{p_vote:.1%}",
                  delta=f"{(p_vote - threshold) * 100:+.1f} điểm % so với ngưỡng {threshold:.2f}",
                  delta_color="inverse")
        if p_vote >= threshold:
            st.error(f"⛔ TỪ CHỐI / CẦN THẨM ĐỊNH THÊM (xác suất ≥ {threshold:.2f})")
        else:
            st.success(f"✅ CHẤP THUẬN (xác suất < {threshold:.2f})")

        st.markdown("**Xác suất theo từng mô hình**")
        st.dataframe(pd.DataFrame({
            "Mô hình": ["XGBoost (Tuned)", "LightGBM (Tuned)", "Random Forest",
                        f"Soft-Voting ({w_xgb}/{w_lgb}/{w_rf})", "Stacking (Meta LR)"],
            "Xác suất vỡ nợ": [p_xgb, p_lgb, p_rf, p_vote, p_stack],
        }).style.format({"Xác suất vỡ nợ": "{:.1%}"}), hide_index=True, use_container_width=True)

        st.markdown("**Một vài đặc trưng đã tính**")
        row = fe.iloc[0]
        st.write(f"- Nợ / thu nhập (`true_loan_to_income`): **{row['true_loan_to_income']:.2f}**")
        st.write(f"- Lãi / thu nhập (`interest_to_income_ratio`): **{row['interest_to_income_ratio']:.3f}**")
        st.write(f"- Lãi suất dùng cho model: **{row['loan_int_rate']:.2f}%**"
                 + (" (điền theo median của hạng)" if rate_unknown else ""))

    with c2:
        st.subheader("Vì sao model ra kết quả này? (SHAP)")
        try:
            import shap

            explainer, explained_model = load_explainer(artifacts)
            sv = explainer(X)
            if len(sv.shape) == 3:  # LightGBM có thể trả về 2 lớp
                sv = sv[:, :, 1]
            fig = plt.figure(figsize=(7, 5))
            shap.plots.waterfall(sv[0], max_display=10, show=False)
            st.pyplot(plt.gcf(), clear_figure=True)
            plt.close(fig)
            st.caption(
                f"SHAP giải thích riêng mô hình **{explained_model}** (đơn vị log-odds), "
                "không phải toàn bộ ensemble. Giá trị bên trái là giá trị đã chuẩn hóa. "
                "Đỏ = đẩy rủi ro lên, xanh = kéo rủi ro xuống."
            )
        except Exception as e:  # noqa: BLE001
            st.warning(f"Không tính được SHAP: {e}")

# ----------------------------------------------------------------------------
with tab2:
    st.subheader("Kết quả trên tập Test độc lập (4,862 hồ sơ, ngưỡng 0.50)")
    table = artifacts.get("results_table") or (metrics or {}).get("models_benchmark")
    if table:
        df = pd.DataFrame(table)
        st.dataframe(df.style.format({c: "{:.4f}" for c in df.columns if c != "Model"}),
                     hide_index=True, use_container_width=True)
    st.markdown(
        "- XGBoost Tuned có ROC-AUC/PR-AUC cao nhất. **Ensemble không vượt được XGBoost đơn lẻ** "
        "vì 3 mô hình cây học trên cùng đặc trưng nên sai giống nhau.\n"
        "- Soft-Voting có Precision cao nhất.\n"
        "- Val_ROC_AUC của Stacking bị lạc quan vì meta-learner được fit trên chính tập Val."
    )

# ----------------------------------------------------------------------------
with tab3:
    st.subheader("Chọn ngưỡng theo chi phí: FN = \\$5,000, FP = \\$1,000")
    if metrics and "test_cost_at_default_threshold" in metrics:
        d, o = metrics["test_cost_at_default_threshold"], metrics["test_cost_at_best_threshold"]
        st.dataframe(pd.DataFrame([
            {"Ngưỡng": d["threshold"], "Bắt được vỡ nợ": f"{d['tp']}/{d['tp'] + d['fn']}",
             "Recall": f"{d['recall']:.1%}", "FN": d["fn"], "FP": d["fp"], "Tổng chi phí (USD)": f"{d['total_cost_usd']:,}"},
            {"Ngưỡng": o["threshold"], "Bắt được vỡ nợ": f"{o['tp']}/{o['tp'] + o['fn']}",
             "Recall": f"{o['recall']:.1%}", "FN": o["fn"], "FP": o["fp"], "Tổng chi phí (USD)": f"{o['total_cost_usd']:,}"},
        ]), hide_index=True, use_container_width=True)
        st.success(f"Tiết kiệm trên tập Test: **\\${metrics['cost_saved_on_test_usd']:,}**, "
                   f"đổi lại phải từ chối nhầm thêm {o['fp'] - d['fp']} khách tốt.")
    img = os.path.join(ASSETS_DIR, "threshold_optimization.png")
    if os.path.exists(img):
        st.image(img, caption="Tổng chi phí trên tập Validation theo từng ngưỡng", use_container_width=True)

# ----------------------------------------------------------------------------
with tab4:
    c1, c2 = st.columns(2)
    for col, name, cap in [
        (c1, "shap_summary.png", "SHAP toàn cục (XGBoost, 400 hồ sơ Test)"),
        (c2, "shap_waterfall_single.png", "SHAP waterfall: hồ sơ #11 trong tập Test"),
    ]:
        p = os.path.join(ASSETS_DIR, name)
        if os.path.exists(p):
            col.image(p, caption=cap, use_container_width=True)
    p = os.path.join(ASSETS_DIR, "eda_overview.png")
    if os.path.exists(p):
        st.image(p, caption="EDA: tỷ lệ vỡ nợ theo hạng, lãi suất theo hạng, outlier tuổi/số năm đi làm",
                 use_container_width=True)

st.caption(f"Model: `{os.path.relpath(artifact_path, BASE_DIR)}` · Tác giả: Lê Công Anh Dũng")
