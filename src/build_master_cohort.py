import os
import pandas as pd
import numpy as np


# ====== 0) 项目根目录 ======
BASE = r"F:\DLBCL_pathway_fault"

# 自动寻找 clinical 文件（你现在是 tsv）
CLINICAL_CANDIDATES = [
    os.path.join(BASE, r"data\raw\clinical\clinical_patient.tsv"),
    os.path.join(BASE, r"data\raw\clinical\clinical_patient.csv"),
    os.path.join(BASE, r"data\raw\clinical\data_clinical_patient.txt"),
]

EXPR = os.path.join(BASE, r"data\raw\expr\expr_rsem.tsv")
CNV  = os.path.join(BASE, r"data\raw\cnv\cnv_gistic.tsv")
MUT  = os.path.join(BASE, r"data\raw\mut\mutations_maf.tsv")  # 可选，没有就自动跳过

OUT_DIR  = os.path.join(BASE, r"data\processed")
OUT_FILE = os.path.join(OUT_DIR, "master_cohort.csv")


# ====== 1) 工具函数 ======
def to_patient_id(x: str) -> str:
    """TCGA-XX-YYYY-01 -> TCGA-XX-YYYY"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    if s.startswith("TCGA-") and len(s) >= 12:
        return s[:12]
    return s

def smart_read_table(path: str) -> pd.DataFrame:
    """自动判断 csv/tsv"""
    ext = os.path.splitext(path)[1].lower()
    if ext in [".tsv", ".txt"]:
        return pd.read_csv(path, sep="\t", dtype=str)
    try:
        return pd.read_csv(path, dtype=str)
    except Exception:
        return pd.read_csv(path, sep=None, engine="python", dtype=str)

def find_col(df: pd.DataFrame, candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None

def find_col_contains(df: pd.DataFrame, keywords):
    """列名包含所有关键词（不区分大小写）"""
    for c in df.columns:
        low = c.lower()
        ok = True
        for k in keywords:
            if k.lower() not in low:
                ok = False
                break
        if ok:
            return c
    return None

def guess_id_col(df: pd.DataFrame) -> str:
    # 你的文件里就是 '#Patient Identifier'
    for cand in ["#Patient Identifier", "Patient Identifier", "PATIENT_ID", "patient_id", "CASE_ID", "case_id",
                 "SAMPLE_ID", "sample_id", "submitter_id"]:
        if cand in df.columns:
            return cand
    # 兜底：找包含 TCGA- 的列
    for c in df.columns:
        if df[c].astype(str).str.contains("TCGA-").any():
            return c
    raise ValueError("Cannot find a patient/sample id column in clinical file.")


# ====== 2) 主程序 ======
def main():
    print("=== Checking files ===")
    clinical_path = None
    for p in CLINICAL_CANDIDATES:
        if os.path.exists(p):
            clinical_path = p
            break

    print("clinical =>", clinical_path if clinical_path else "(NOT FOUND)")
    print(EXPR, "=>", os.path.exists(EXPR))
    print(CNV,  "=>", os.path.exists(CNV))
    print()

    if clinical_path is None:
        raise FileNotFoundError("Clinical file not found. Put clinical_patient.tsv under data/raw/clinical/")

    # --- 2.1 读 clinical ---
    cli = smart_read_table(clinical_path)
    print("clinical shape:", cli.shape)
    print("clinical columns (first 30):", cli.columns.tolist()[:30])

    id_col = guess_id_col(cli)
    cli["patient_id"] = cli[id_col].map(to_patient_id)

    # --- 2.2 构造 OS_time / OS_event（适配你这份 PanCanAtlas 风格） ---
    overall_status_col = find_col(cli, ["Overall Survival Status"]) or find_col_contains(cli, ["overall", "survival", "status"])
    # 可能存在更标准的 OS 月/日字段（不一定在前30列）
    overall_months_col = find_col_contains(cli, ["overall", "survival", "month"])
    overall_days_col   = find_col_contains(cli, ["overall", "survival", "day"])

    # 你这份表明确有这列（天数）
    last_alive_days_col = find_col(cli, ["Last Alive Less Initial Pathologic Diagnosis Date Calculated Day Value"]) \
                          or find_col_contains(cli, ["last alive", "initial pathologic", "day value"])

    if not overall_status_col:
        surv_cols = [c for c in cli.columns if "survival" in c.lower()]
        print("Available survival-like columns:", surv_cols)
        raise ValueError("Cannot find Overall Survival Status column.")

    # event: DECEASED/DEAD => 1, else 0
    cli["OS_event"] = cli[overall_status_col].astype(str).str.upper().str.contains("DECEASED|DEAD").astype(int)

    if overall_days_col:
        cli["OS_time"] = pd.to_numeric(cli[overall_days_col], errors="coerce")
        print(f"Using OS_time from: {overall_days_col}")
    elif overall_months_col:
        cli["OS_time"] = pd.to_numeric(cli[overall_months_col], errors="coerce") * 30.4375
        print(f"Using OS_time from: {overall_months_col} (months->days)")
    elif last_alive_days_col:
        cli["OS_time"] = pd.to_numeric(cli[last_alive_days_col], errors="coerce")
        print(f"Using OS_time from: {last_alive_days_col}")
    else:
        surv_cols = [c for c in cli.columns if "alive" in c.lower() or "death" in c.lower() or "survival" in c.lower()]
        print("Available time-like columns:", surv_cols)
        raise ValueError("Cannot find an OS time column (days or months).")

    cli_surv = cli[["patient_id", "OS_time", "OS_event"]].dropna()
    cli_surv = cli_surv.drop_duplicates("patient_id")
    print("survival rows:", cli_surv.shape)

    # --- 2.3 读 expr/cnv，标记 has_expr/has_cnv ---
    expr = smart_read_table(EXPR)
    cnv  = smart_read_table(CNV)

    expr_patients = [to_patient_id(c) for c in expr.columns[1:]]
    cnv_patients  = [to_patient_id(c) for c in cnv.columns[1:]]

    has_expr = pd.Series(1, index=pd.Index(expr_patients, name="patient_id"), dtype=int).groupby(level=0).max()
    has_cnv  = pd.Series(1, index=pd.Index(cnv_patients,  name="patient_id"), dtype=int).groupby(level=0).max()

    # mutation 可选
    has_mut = pd.Series(dtype=int)
    if os.path.exists(MUT):
        mut = smart_read_table(MUT)
        mcol = find_col(mut, ["Tumor_Sample_Barcode", "tumor_sample_barcode", "SAMPLE_ID", "sample_id"])
        if mcol:
            mut_patients = mut[mcol].map(to_patient_id).dropna().unique().tolist()
            has_mut = pd.Series(1, index=pd.Index(mut_patients, name="patient_id"), dtype=int).groupby(level=0).max()
        else:
            print("[WARN] mutation file exists but cannot find barcode column; skip has_mut.")
    else:
        print("(mutations file not found, skip has_mut)")

    # --- 2.4 合并 master ---
    master = cli_surv.merge(has_expr.rename("has_expr"), on="patient_id", how="left") \
                     .merge(has_cnv.rename("has_cnv"),  on="patient_id", how="left") \
                     .merge(has_mut.rename("has_mut"),  on="patient_id", how="left")

    for c in ["has_expr", "has_cnv", "has_mut"]:
        master[c] = master[c].fillna(0).astype(int)

    os.makedirs(OUT_DIR, exist_ok=True)
    master.to_csv(OUT_FILE, index=False)

    print("\n=== Saved ===")
    print(OUT_FILE)

    print("\n=== Counts ===")
    print("N_total:", len(master))
    print("N_expr :", master["has_expr"].sum())
    print("N_cnv  :", master["has_cnv"].sum())
    print("N_mut  :", master["has_mut"].sum())
    print("N_expr_cnv:", ((master["has_expr"] == 1) & (master["has_cnv"] == 1)).sum())
    print("N_all (expr+cnv+mut):", ((master["has_expr"] == 1) & (master["has_cnv"] == 1) & (master["has_mut"] == 1)).sum())


if __name__ == "__main__":
    main()