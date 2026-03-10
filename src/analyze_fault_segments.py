import os
import numpy as np
import pandas as pd

BASE = r"F:\DLBCL_pathway_fault"

MASTER  = os.path.join(BASE, r"data\processed\master_cohort.csv")
CLIN    = os.path.join(BASE, r"data\raw\clinical\clinical_patient.tsv")
NODE    = os.path.join(BASE, r"data\processed\node_fault_scores.tsv")
PATH    = os.path.join(BASE, r"data\processed\pathway_dys_scores.tsv")

OUT_TAB = os.path.join(BASE, r"results\tables")
OUT_FIG = os.path.join(BASE, r"results\figures")
os.makedirs(OUT_TAB, exist_ok=True)
os.makedirs(OUT_FIG, exist_ok=True)

# ====== 1) 你要的“故障部位”分段（segments） ======
SEGMENTS = {
    # p53 轴：上游损伤感知
    "p53_upstream_damage": ["ATM", "ATR", "CHEK1", "CHEK2"],
    # p53 轴：核心调控（抑癌核心 & 负调控 & ARF）
    "p53_core_regulation": ["TP53", "MDM2", "MDM4", "CDKN2A"],
    # p53 输出：细胞周期阻滞
    "p53_cell_cycle_stop": ["CDKN1A"],
    # p53 输出：凋亡执行（线粒体促凋亡）
    "p53_apoptosis_output": ["BAX", "BBC3", "PMAIP1"],

    # RB/细胞周期检查点：上游驱动（增殖推动）
    "RB_upstream_drive": ["CCND1", "CCND2", "CCND3", "CDK4", "CDK6"],
    # RB/检查点：核心门控与抑制
    "RB_checkpoint_core": ["RB1", "CDKN2A", "CDKN1B"],
    # RB 下游增殖程序（可选读出）
    "RB_E2F_output": ["E2F1"],
}

def to_patient_id(x: str) -> str:
    s = str(x).strip()
    return s[:12] if s.startswith("TCGA-") and len(s) >= 12 else s

def main():
    master = pd.read_csv(MASTER)
    node = pd.read_csv(NODE, sep="\t")  # patient_id + genes
    path = pd.read_csv(PATH,  sep="\t")
    node["patient_id"] = node["patient_id"].map(to_patient_id)
    path["patient_id"] = path["patient_id"].map(to_patient_id)

    # 临床：取 patient_id 与 subtype（有就用）
    clin = pd.read_csv(CLIN, sep="\t", dtype=str)
    id_col = "#Patient Identifier" if "#Patient Identifier" in clin.columns else clin.columns[0]
    clin["patient_id"] = clin[id_col].map(to_patient_id)
    subtype_col = "Subtype" if "Subtype" in clin.columns else None
    clin_small = clin[["patient_id"] + ([subtype_col] if subtype_col else [])].drop_duplicates("patient_id")

    # 合并成分析表
    df = master.merge(node, on="patient_id", how="inner") \
               .merge(path, on="patient_id", how="left") \
               .merge(clin_small, on="patient_id", how="left")

    # 只保留 expr+cnv 的工作集（你现在是48）
    df = df[(df["has_expr"] == 1) & (df["has_cnv"] == 1)].copy()
    print("analysis rows:", df.shape)

    # ====== 2) 计算 segment 分数（每段 = 该段节点故障均值） ======
    genes_available = set(df.columns)
    seg_scores = pd.DataFrame({"patient_id": df["patient_id"]})
    for seg, glist in SEGMENTS.items():
        gl = [g for g in glist if g in genes_available]
        if len(gl) == 0:
            seg_scores[seg] = np.nan
        else:
            seg_scores[seg] = df[gl].mean(axis=1)

    seg_scores.to_csv(os.path.join(OUT_TAB, "segment_scores.tsv"), sep="\t", index=False)
    print("saved:", os.path.join(OUT_TAB, "segment_scores.tsv"))

    # ====== 3) 每个患者输出“Top 故障节点/故障段”报告 ======
    gene_cols = [c for c in df.columns if c.isupper() and len(c) <= 10]  # 粗略抓基因列
    # 更稳：用 NODE 文件里的列名
    node_cols = [c for c in node.columns if c != "patient_id"]

    report_rows = []
    for _, row in df.iterrows():
        pid = row["patient_id"]
        # Top genes
        gvals = row[node_cols].astype(float).sort_values(ascending=False)
        top_genes = ",".join([f"{g}:{gvals[g]:.2f}" for g in gvals.index[:5]])
        # Top segments
        srow = seg_scores[seg_scores["patient_id"] == pid].iloc[0].drop("patient_id").astype(float).sort_values(ascending=False)
        top_segs = ",".join([f"{s}:{srow[s]:.2f}" for s in srow.index[:3]])
        report_rows.append({
            "patient_id": pid,
            "OS_time": row["OS_time"],
            "OS_event": row["OS_event"],
            "Subtype": row.get("Subtype", ""),
            "Top5_genes": top_genes,
            "Top3_segments": top_segs
        })

    report = pd.DataFrame(report_rows)
    report.to_csv(os.path.join(OUT_TAB, "patient_fault_report.tsv"), sep="\t", index=False)
    print("saved:", os.path.join(OUT_TAB, "patient_fault_report.tsv"))

    # ====== 4) 简单可视化：segment 热图 & KM 曲线（如果 lifelines 有装） ======
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plot_seg = seg_scores.set_index("patient_id")
        plt.figure(figsize=(10, 0.28 * len(plot_seg) + 3))
        sns.heatmap(plot_seg, vmin=0, vmax=1, cmap="viridis")
        plt.title("DLBCL SegmentFault Heatmap (what part is broken?)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_FIG, "segment_heatmap.png"), dpi=200)
        plt.close()
        print("saved:", os.path.join(OUT_FIG, "segment_heatmap.png"))
    except Exception as e:
        print("[WARN] segment heatmap failed:", repr(e))

    # ====== 5) 生存验证：p53_axis / RB_cycle 高低组 KM（可选但强烈建议） ======
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test
        import matplotlib.pyplot as plt

        def km_plot(score_col, out_name):
            tmp = df[["OS_time", "OS_event", score_col]].dropna().copy()
            med = tmp[score_col].median()
            tmp["group"] = (tmp[score_col] >= med).astype(int)  # 1=high dysfunction
            g0 = tmp[tmp["group"] == 0]
            g1 = tmp[tmp["group"] == 1]

            kmf0 = KaplanMeierFitter()
            kmf1 = KaplanMeierFitter()

            plt.figure(figsize=(6,4))
            kmf0.fit(g0["OS_time"], event_observed=g0["OS_event"], label=f"{score_col} low")
            kmf1.fit(g1["OS_time"], event_observed=g1["OS_event"], label=f"{score_col} high")
            ax = kmf0.plot()
            kmf1.plot(ax=ax)
            res = logrank_test(g0["OS_time"], g1["OS_time"], event_observed_A=g0["OS_event"], event_observed_B=g1["OS_event"])
            plt.title(f"KM: {score_col} (median split), p={res.p_value:.3g}")
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_FIG, out_name), dpi=200)
            plt.close()

        km_plot("p53_axis", "km_p53_axis.png")
        km_plot("RB_cycle", "km_RB_cycle.png")
        print("saved KM plots:", os.path.join(OUT_FIG, "km_p53_axis.png"), os.path.join(OUT_FIG, "km_RB_cycle.png"))

    except Exception as e:
        print("[WARN] survival plots skipped (lifelines maybe not installed):", repr(e))

    # ====== 6) 统计：哪些节点最“拉开差异”（按 p53_axis 高低组举例） ======
    tmp = df[["p53_axis"] + node_cols].dropna().copy()
    med = tmp["p53_axis"].median()
    high = tmp[tmp["p53_axis"] >= med]
    low  = tmp[tmp["p53_axis"] <  med]
    deltas = []
    for g in node_cols:
        deltas.append({"gene": g, "delta_high_minus_low": float(high[g].mean() - low[g].mean()),
                       "mean_high": float(high[g].mean()), "mean_low": float(low[g].mean())})
    deltas = pd.DataFrame(deltas).sort_values("delta_high_minus_low", ascending=False)
    deltas.to_csv(os.path.join(OUT_TAB, "gene_delta_p53_high_low.tsv"), sep="\t", index=False)
    print("saved:", os.path.join(OUT_TAB, "gene_delta_p53_high_low.tsv"))

if __name__ == "__main__":
    main()