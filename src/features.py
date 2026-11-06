"""
Feature engineering dùng chung cho train.py và app.py.

Logic giữ NGUYÊN như hàm `apply_imputation_and_feature_engineering` trong
notebook Credit_Risk_XGBoost_Colab.ipynb (Bước 1.3), để dữ liệu lúc suy luận
được xử lý giống hệt lúc huấn luyện.
"""

import numpy as np
import pandas as pd

GRADE_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}

# 11 cột đầu vào thô (bỏ nhãn loan_status)
RAW_COLUMNS = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
]

CAT_COLS = ["person_home_ownership", "loan_intent"]
DROP_RAW_COLS = ["loan_grade", "cb_person_default_on_file"] + CAT_COLS


def apply_imputation_and_feature_engineering(
    df_part: pd.DataFrame,
    grade_int_rate_median: dict,
    global_int_rate_median: float,
    emp_length_median: float,
) -> pd.DataFrame:
    """Điền khuyết (dùng thống kê học từ tập Train) + tạo đặc trưng tài chính."""
    d = df_part.copy()

    # 1. Cờ đánh dấu hồ sơ bị thiếu thông tin
    d["int_rate_missing"] = d["loan_int_rate"].isna().astype(int)
    d["emp_length_missing"] = d["person_emp_length"].isna().astype(int)

    # 2. Điền khuyết loan_int_rate theo nhóm loan_grade (median học từ Train)
    d["loan_int_rate"] = (
        d["loan_int_rate"]
        .fillna(d["loan_grade"].map(grade_int_rate_median))
        .fillna(global_int_rate_median)
    )
    d["person_emp_length"] = d["person_emp_length"].fillna(emp_length_median)

    # 3. Đặc trưng tài chính
    d["true_loan_to_income"] = d["loan_amnt"] / (d["person_income"] + 1.0)
    d["loan_percent_discrepancy"] = (d["loan_percent_income"] - d["true_loan_to_income"]).abs()

    d["annual_interest_amount"] = d["loan_amnt"] * (d["loan_int_rate"] / 100.0)
    d["interest_to_income_ratio"] = d["annual_interest_amount"] / (d["person_income"] + 1.0)

    d["income_per_emp_year"] = d["person_income"] / (d["person_emp_length"] + 1.0)
    d["cred_hist_to_adult_age"] = d["cb_person_cred_hist_length"] / np.maximum(d["person_age"] - 18.0, 1.0)

    d["loan_grade_num"] = d["loan_grade"].map(GRADE_MAP).fillna(4).astype(int)
    d["prior_default_flag"] = (d["cb_person_default_on_file"] == "Y").astype(int)
    return d
