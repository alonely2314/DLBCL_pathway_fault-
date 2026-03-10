import os
import json
import numpy as np
import pandas as pd

# =====================
# 0) 路径
# =====================
BASE = r"F:\DLBCL_pathway_fault"
MASTER = os.path.join(BASE, r"data\processed\master_cohort.csv")
EXPR   = os.path.join(BASE, r"data\raw\expr\expr_rsem.tsv")
CNV    = os.path.join(BASE, r"data\raw\cnv\cnv_gistic.tsv")

OUT_NODE = os.path.join(BASE, r"data\processed\node_fault_scores.tsv")
OUT_PATH = os.path.join(BASE, r"data\processed\pathway_dys_scores.tsv")
FIG_DIR  = os.path.join(BASE, r"results\figures")
os.makedirs(os.path.dirname(OUT_NODE), exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# =====================
# 1) 通路节点清单（先做两条，够一学期闭环）
# =====================
PATHWAYS = {
    "p53_axis": [
        "TP53","MDM2","MDM4","CDKN2A","ATM","ATR","CHEK1","CHEK2","CDKN1A","BAX","BBC3","PMAIP1"
    ],
    "RB_cycle": [
        "RB1","CDK4","CDK6","CCND1","CCND2","CCND3","CDKN2A","CDKN1B","E2F1"
    ]
}
GENES = sorted(set(sum(PATHWAYS.values(), [])))

# =====================
# 2) 基因角色（决定“高表达更坏”还是“低表达更坏”，以及 CNV 是缺失坏还是扩增坏）
# =====================
TUMOR_SUPPRESSOR = {"TP53","CDKN2A","RB1","ATM","ATR","CHEK1","CHEK2","CDKN1A","CDKN1B","BAX","BBC3","PMAIP1"}
NEG_REGULATOR_P53 = {"MDM2","MDM4"}               # p53 负调控：高表达/扩增更坏
PROLIFERATION     = {"CDK4","CDK6","CCND1","CCND2","CCND3","E2F1"}  # 增殖驱动：高表达/扩增更坏

def expr_direction(g):
    # 返回 +1 表示 high_is_bad（高表达更坏）；-1 表示 low_is_bad（低表达更坏）
    if g in NEG_REGULATOR_P53 or g in PROLIFERATION:
        return +1
    return -1  # 默认按抑癌节点：低表达更坏

def cnv_direction(g):
    # 返回 "loss_is_bad" 或 "gain_is_bad"
    if g in NEG_REGULATOR_P53 or g in PROLIFERATION:
        return "gain_is_bad"
    return "loss_is_bad"

# =====================
# 3) 读矩阵（cBioPortal 常见：前两列为 Hugo_Symbol / Entrez_Gene_Id）
# =====================
def read_matrix(path):
    df = pd.read_csv(path, sep="\t", dtype=str)
    # 找基因列
    gene_col = None
    for cand in ["Hugo_Symbol", "gene", "Gene", "GENE"]:
        if cand in df.columns:
            gene_col = cand
            break
    if gene_col is None:
        gene_col = df.columns[0]

    # 去掉 Entrez 列（如有）
    drop_cols = [c for c in df.columns if c.lower() in ["entrez_gene_id", "entrez"]]
    keep_cols = [c for c in df.columns if c not in drop_cols]
    df = df[keep_cols]

    df = df.rename(columns={gene_col: "gene"})
    # 转成数值（样本列）
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def to_patient_id(sample_id: str) -> str:
    s = str(sample_id)
    return s[:12] if s.startswith("TCGA-") and len(s) >= 12 else s

# =====================
# 4) 评分函数
# =====================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def robust_z(x, med, iqr, eps=1e-6):
    return (x - med) / (iqr + eps)

def main():
    # ---- 4.1 读 master cohort，取 expr+cnv 的患者（48个）
    master = pd.read_csv(MASTER)
    work = master[(master["has_expr"] == 1) & (master["has_cnv"] == 1)].copy()
    patients = work["patient_id"].tolist()
    print("work patients (expr+cnv):", len(patients))

    # ---- 4.2 读 expr/cnv
    expr = read_matrix(EXPR)
    cnv  = read_matrix(CNV)

    # 映射样本列 -> patient_id（可能多个样本映射到同一 patient，取均值）
    expr_cols = expr.columns[1:]
    cnv_cols  = cnv.columns[1:]
    expr_pid = [to_patient_id(c) for c in expr_cols]
    cnv_pid  = [to_patient_id(c) for c in cnv_cols]

    # 只保留工作集患者列
    expr_keep = [col for col, pid in zip(expr_cols, expr_pid) if pid in patients]
    cnv_keep  = [col for col, pid in zip(cnv_cols,  cnv_pid)  if pid in patients]

    expr = expr[["gene"] + expr_keep].copy()
    cnv  = cnv[["gene"] + cnv_keep].copy()

    # 将列名改成 patient_id（如果同一patient多个列，后面会聚合）
    expr.rename(columns={c: to_patient_id(c) for c in expr_keep}, inplace=True)
    cnv.rename(columns={c: to_patient_id(c) for c in cnv_keep}, inplace=True)

    # 若同一 patient 有重复列，按 patient 聚合取均值
    expr = expr.groupby("gene", as_index=False).first()
    cnv  = cnv.groupby("gene",  as_index=False).first()

    expr_mat = expr.set_index("gene")
    cnv_mat  = cnv.set_index("gene")

    # ---- 4.3 取节点基因子集（GENES）
    # 有些基因可能不在矩阵里：先对齐交集
    genes_present = [g for g in GENES if g in expr_mat.index and g in cnv_mat.index]
    missing = [g for g in GENES if g not in genes_present]
    print("genes present:", len(genes_present), "missing:", missing)

    X_expr = expr_mat.loc[genes_present, patients].astype(float)
    X_cnv  = cnv_mat.loc[genes_present, patients].astype(float)

    # ---- 4.4 计算 expr 的 robust z-score（每个基因一套 median/IQR）
    med = X_expr.median(axis=1)
    q75 = X_expr.quantile(0.75, axis=1)
    q25 = X_expr.quantile(0.25, axis=1)
    iqr = (q75 - q25).replace(0, np.nan).fillna(1e-6)

    Z = (X_expr.sub(med, axis=0)).div(iqr, axis=0)

    # ---- 4.5 expr_fault：根据方向映射到 0-1
    expr_fault = pd.DataFrame(index=genes_present, columns=patients, dtype=float)
    for g in genes_present:
        sign = expr_direction(g)  # +1 high_is_bad; -1 low_is_bad
        expr_fault.loc[g] = sigmoid(sign * Z.loc[g].values)

    # ---- 4.6 cnv_fault：GISTIC -2..2 => loss/gain flag
    cnv_fault = pd.DataFrame(index=genes_present, columns=patients, dtype=float)
    for g in genes_present:
        rule = cnv_direction(g)
        v = X_cnv.loc[g].values
        if rule == "loss_is_bad":
            cnv_fault.loc[g] = (v <= -1).astype(float)
        else:
            cnv_fault.loc[g] = (v >= 1).astype(float)

    # ---- 4.7 NodeFault：加权融合（先用固定权重跑通）
    w_e, w_c = 0.6, 0.4
    node_fault = (w_e * expr_fault + w_c * cnv_fault).clip(0, 1)

    # 转成：行=patient，列=gene
    node_fault_T = node_fault.T  # patients x genes
    node_fault_T.index.name = "patient_id"
    node_fault_T.to_csv(OUT_NODE, sep="\t")
    print("saved:", OUT_NODE, node_fault_T.shape)

    # ---- 4.8 PathDys：通路受损分数 = 通路内节点故障均值
    path_scores = pd.DataFrame(index=patients)
    for k, glist in PATHWAYS.items():
        gl = [g for g in glist if g in genes_present]
        path_scores[k] = node_fault_T[gl].mean(axis=1)
    path_scores.index.name = "patient_id"
    path_scores.to_csv(OUT_PATH, sep="\t")
    print("saved:", OUT_PATH, path_scores.shape)

    # ---- 4.9 画热图（保存到 results/figures）
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # 按通路顺序排列列
        col_order = []
        for k in ["p53_axis", "RB_cycle"]:
            for g in PATHWAYS[k]:
                if g in node_fault_T.columns and g not in col_order:
                    col_order.append(g)

        plot_df = node_fault_T[col_order].copy()

        plt.figure(figsize=(1 + 0.5*len(col_order), 0.3*len(patients) + 4))
        sns.heatmap(plot_df, cmap="viridis", vmin=0, vmax=1)
        plt.title("DLBCL NodeFault Heatmap (expr+cnv)")
        plt.tight_layout()
        fig_path = os.path.join(FIG_DIR, "fault_heatmap.png")
        plt.savefig(fig_path, dpi=200)
        plt.close()
        print("saved figure:", fig_path)

        # 通路分数分布图
        plt.figure(figsize=(6,4))
        sns.kdeplot(path_scores["p53_axis"], label="p53_axis")
        sns.kdeplot(path_scores["RB_cycle"], label="RB_cycle")
        plt.legend()
        plt.title("Pathway Dysfunction Score Distribution")
        plt.tight_layout()
        fig2 = os.path.join(FIG_DIR, "pathway_scores.png")
        plt.savefig(fig2, dpi=200)
        plt.close()
        print("saved figure:", fig2)

    except Exception as e:
        print("[WARN] plotting failed (you can ignore). Reason:", repr(e))
        print("NodeFault & Pathway scores are already saved.")

if __name__ == "__main__":
    main()