import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score
)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Teen Mental Health · Analisis Depresi",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Dark sidebar */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%);
    color: #e2e8f0;
  }
  [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
  [data-testid="stSidebar"] .stMarkdown h2 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    letter-spacing: .06em;
    text-transform: uppercase;
    color: #818cf8 !important;
    border-bottom: 1px solid #334155;
    padding-bottom: .4rem;
    margin-top: 1.5rem;
  }

  /* Hero banner */
  .hero {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4c1d95 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: white;
  }
  .hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0 0 .4rem;
  }
  .hero p { margin: 0; opacity: .85; font-size: 1rem; }

  /* Stage header pills */
  .stage-pill {
    display: inline-block;
    background: #312e81;
    color: #a5b4fc;
    font-size: .72rem;
    font-weight: 600;
    letter-spacing: .1em;
    text-transform: uppercase;
    border-radius: 999px;
    padding: .2rem .75rem;
    margin-bottom: .5rem;
  }

  /* Metric cards */
  .metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }
  .metric-card {
    flex: 1; min-width: 140px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
  }
  .metric-card .val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #312e81;
  }
  .metric-card .lbl { font-size: .78rem; color: #64748b; margin-top: .15rem; }

  /* Section divider */
  .stage-divider { border: none; border-top: 2px solid #e0e7ff; margin: 2rem 0; }

  /* Info boxes */
  .info-box {
    background: #eef2ff;
    border-left: 4px solid #6366f1;
    border-radius: 0 8px 8px 0;
    padding: .8rem 1rem;
    font-size: .9rem;
    color: #3730a3;
    margin-bottom: 1rem;
  }

  /* Table styling override */
  [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
LABEL_MAP = {0: "Normal", 1: "Depresi"}
FEATURE_COLS = [
    "age", "daily_social_media_hours", "sleep_hours",
    "screen_time_before_sleep", "academic_performance",
    "physical_activity", "stress_level", "anxiety_level", "addiction_level",
]
FEATURE_LABELS = {
    "age": "Usia",
    "daily_social_media_hours": "Jam Medsos / Hari",
    "sleep_hours": "Jam Tidur",
    "screen_time_before_sleep": "Screen Time Sebelum Tidur",
    "academic_performance": "Performa Akademik",
    "physical_activity": "Aktivitas Fisik",
    "stress_level": "Tingkat Stres",
    "anxiety_level": "Tingkat Kecemasan",
    "addiction_level": "Tingkat Kecanduan",
}

@st.cache_data
def load_data():
    df = pd.read_excel("Teen_Mental_Health_Dataset.xlsx")
    # encode categoricals
    le_gender  = LabelEncoder()
    le_platf   = LabelEncoder()
    le_social  = LabelEncoder()
    df["gender_enc"]  = le_gender.fit_transform(df["gender"])
    df["platform_enc"] = le_platf.fit_transform(df["platform_usage"])
    df["social_enc"]  = le_social.fit_transform(df["social_interaction_level"])
    return df

@st.cache_resource
def train_model(n_trees, max_depth, random_state, df):
    X = df[FEATURE_COLS]
    y = df["depression_label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    rf = RandomForestClassifier(
        n_estimators=n_trees,
        max_depth=max_depth if max_depth > 0 else None,
        oob_score=True,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # OOB curve (staged estimators)
    oob_errors = []
    for i, pred in enumerate(rf.estimators_):
        # approximate staged OOB using estimators list
        pass
    # proper staged OOB via warm_start
    oob_staged = []
    rf_staged = RandomForestClassifier(
        warm_start=True, oob_score=True,
        max_depth=max_depth if max_depth > 0 else None,
        random_state=random_state
    )
    for n in range(1, n_trees + 1, max(1, n_trees // 50)):
        rf_staged.n_estimators = n
        rf_staged.fit(X_train, y_train)
        oob_staged.append((n, 1 - rf_staged.oob_score_))

    y_pred  = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)
    return rf, X_train, X_test, y_train, y_test, y_pred, y_proba, oob_staged

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Teen Mental Health")
    st.markdown("*Dashboard Analisis Depresi Remaja berbasis Random Forest*")
    st.markdown("---")
    st.markdown("## ⚙️ Parameter Model")
    n_trees    = st.slider("Jumlah Pohon", 50, 500, 200, 50)
    max_depth  = st.slider("Kedalaman Pohon (0 = bebas)", 0, 20, 0)
    rand_seed  = st.number_input("Random Seed", value=42, step=1)
    st.markdown("## 📋 Tampilkan Tahap")
    show_all   = st.checkbox("Tampilkan semua tahap", value=True)
    stages = {}
    if not show_all:
        for i, label in enumerate([
            "Ringkasan & OOB", "Pohon Keputusan", "Confusion Matrix",
            "Error per Kategori", "Probabilitas Prediksi", "Feature Importance"
        ], 1):
            stages[i] = st.checkbox(f"Tahap {i}: {label}", value=True)
    else:
        stages = {i: True for i in range(1, 7)}
    st.markdown("---")
    st.caption("Data: Teen Mental Health Dataset · 1,200 sampel · 12 fitur")

# ── Load & train ──────────────────────────────────────────────────────────────
df = load_data()
with st.spinner("🔄 Melatih model Random Forest..."):
    rf, X_train, X_test, y_train, y_test, y_pred, y_proba, oob_staged = train_model(
        n_trees, max_depth, rand_seed, df
    )

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🧠 Analisis Kesehatan Mental Remaja</h1>
  <p>Model Random Forest · Deteksi Risiko Depresi pada Remaja berdasarkan Pola Aktivitas Digital & Gaya Hidup</p>
</div>
""", unsafe_allow_html=True)

# Quick stats row
total     = len(df)
n_depresi = int(df["depression_label"].sum())
acc       = accuracy_score(y_test, y_pred)
oob_err   = round(1 - rf.oob_score_, 4)

st.markdown(f"""
<div class="metric-row">
  <div class="metric-card"><div class="val">{total:,}</div><div class="lbl">Total Sampel</div></div>
  <div class="metric-card"><div class="val">{n_depresi}</div><div class="lbl">Terindikasi Depresi</div></div>
  <div class="metric-card"><div class="val">{acc:.1%}</div><div class="lbl">Akurasi Model</div></div>
  <div class="metric-card"><div class="val">{oob_err:.4f}</div><div class="lbl">OOB Error Rate</div></div>
  <div class="metric-card"><div class="val">{n_trees}</div><div class="lbl">Jumlah Pohon</div></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAHAP 1: Ringkasan & OOB
# ─────────────────────────────────────────────────────────────────────────────
if stages[1]:
    st.markdown('<hr class="stage-divider">', unsafe_allow_html=True)
    st.markdown('<span class="stage-pill">Tahap 1</span>', unsafe_allow_html=True)
    st.subheader("📊 Ringkasan Model & Optimasi Parameter")

    col1, col2 = st.columns([1, 1.6])

    with col1:
        st.markdown('<div class="info-box">ℹ️ <b>Non-Visual:</b> Ringkasan statistik model yang telah dilatih.</div>',
                    unsafe_allow_html=True)
        summary_data = {
            "Parameter": ["Total Pohon", "Kedalaman Maks", "OOB Error Rate", "OOB Score",
                          "Fitur per Split", "Sampel Training", "Sampel Testing"],
            "Nilai": [
                str(rf.n_estimators),
                str(rf.max_depth) if rf.max_depth else "Tidak dibatasi",
                f"{oob_err:.4f} ({oob_err*100:.2f}%)",
                f"{rf.oob_score_:.4f} ({rf.oob_score_*100:.2f}%)",
                str(rf.max_features),
                f"{len(X_train):,}",
                f"{len(X_test):,}",
            ],
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    with col2:
        st.markdown('<div class="info-box">📈 <b>Visual:</b> Kurva OOB Error — lihat bagaimana error turun seiring bertambahnya pohon.</div>',
                    unsafe_allow_html=True)
        if oob_staged:
            xs, ys = zip(*oob_staged)
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.plot(xs, ys, color="#6366f1", linewidth=2.2, label="OOB Error")
            ax.fill_between(xs, ys, alpha=0.12, color="#6366f1")
            ax.axhline(oob_err, color="#ef4444", linestyle="--", linewidth=1.4,
                       label=f"Final OOB: {oob_err:.4f}")
            ax.set_xlabel("Jumlah Pohon", fontsize=10)
            ax.set_ylabel("OOB Error Rate", fontsize=10)
            ax.set_title("Konvergensi OOB Error terhadap Jumlah Pohon", fontsize=11, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(axis="y", alpha=0.3, linestyle="--")
            ax.set_facecolor("#f8fafc")
            fig.patch.set_facecolor("white")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# TAHAP 2: Pembongkaran Logika Model
# ─────────────────────────────────────────────────────────────────────────────
if stages[2]:
    st.markdown('<hr class="stage-divider">', unsafe_allow_html=True)
    st.markdown('<span class="stage-pill">Tahap 2</span>', unsafe_allow_html=True)
    st.subheader("🌳 Pembongkaran Logika Model (Transparansi)")

    col1, col2 = st.columns([1, 1.6])

    with col1:
        st.markdown('<div class="info-box">ℹ️ <b>Non-Visual:</b> Struktur pohon pertama — variabel pemotong & split point.</div>',
                    unsafe_allow_html=True)
        first_tree = rf.estimators_[0]
        tree_struct = first_tree.tree_
        feat_names_mapped = [FEATURE_LABELS.get(FEATURE_COLS[i], FEATURE_COLS[i])
                              if i >= 0 else "Leaf"
                              for i in tree_struct.feature]
        n_show = min(15, tree_struct.node_count)
        tree_df = pd.DataFrame({
            "Node ID": range(n_show),
            "Fitur Split": [feat_names_mapped[i] for i in range(n_show)],
            "Split Point": [f"{tree_struct.threshold[i]:.3f}" if tree_struct.threshold[i] != -2
                            else "—" for i in range(n_show)],
            "Child Kiri": [tree_struct.children_left[i]  for i in range(n_show)],
            "Child Kanan": [tree_struct.children_right[i] for i in range(n_show)],
        })
        st.dataframe(tree_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown('<div class="info-box">🌿 <b>Visual:</b> Diagram alur pohon keputusan (depth 3 untuk keterbacaan).</div>',
                    unsafe_allow_html=True)
        feat_labels_list = [FEATURE_LABELS.get(c, c) for c in FEATURE_COLS]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        plot_tree(
            first_tree,
            max_depth=3,
            feature_names=feat_labels_list,
            class_names=["Normal", "Depresi"],
            filled=True,
            rounded=True,
            fontsize=7,
            ax=ax,
            impurity=False,
            precision=2,
        )
        ax.set_title("Pohon Keputusan Pertama (kedalaman 3)", fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# TAHAP 3 & 4: Validasi Performa
# ─────────────────────────────────────────────────────────────────────────────
if stages[3]:
    st.markdown('<hr class="stage-divider">', unsafe_allow_html=True)
    st.markdown('<span class="stage-pill">Tahap 3 & 4</span>', unsafe_allow_html=True)
    st.subheader("✅ Validasi Performa Global dan Per Kategori")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown('<div class="info-box">ℹ️ <b>Non-Visual:</b> Laporan evaluasi & confusion matrix.</div>',
                    unsafe_allow_html=True)
        cm   = confusion_matrix(y_test, y_pred)
        cr   = classification_report(y_test, y_pred,
                                     target_names=["Normal", "Depresi"],
                                     output_dict=True)
        st.markdown(f"**Akurasi Global: `{acc:.4f}` ({acc*100:.2f}%)**")

        # Heatmap confusion matrix
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Normal", "Depresi"],
                    yticklabels=["Normal", "Depresi"],
                    linewidths=.5, ax=ax, cbar=False,
                    annot_kws={"size": 14, "weight": "bold"})
        ax.set_xlabel("Prediksi", fontsize=10)
        ax.set_ylabel("Aktual", fontsize=10)
        ax.set_title("Confusion Matrix", fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Classification report table
        report_rows = []
        for cls in ["Normal", "Depresi"]:
            r = cr[cls]
            report_rows.append({
                "Kelas": cls,
                "Precision": f"{r['precision']:.3f}",
                "Recall": f"{r['recall']:.3f}",
                "F1-Score": f"{r['f1-score']:.3f}",
                "Support": int(r["support"]),
            })
        st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)

    with col2:
        st.markdown('<div class="info-box">📊 <b>Visual:</b> Error rate per kategori — kategori mana yang paling sulit diprediksi?</div>',
                    unsafe_allow_html=True)
        class_errors = []
        for i, cls_name in enumerate(["Normal", "Depresi"]):
            total_cls = cm[i].sum()
            wrong     = total_cls - cm[i, i]
            class_errors.append(wrong / total_cls if total_cls > 0 else 0)

        fig, ax = plt.subplots(figsize=(5, 3.5))
        colors = ["#6366f1", "#ef4444"]
        bars = ax.bar(["Normal", "Depresi"], class_errors, color=colors,
                      width=0.5, edgecolor="white", linewidth=1.5)
        ax.set_ylabel("Error Rate", fontsize=10)
        ax.set_title("Error Rate per Kategori", fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(class_errors) * 1.4 + 0.02)
        for bar, val in zip(bars, class_errors):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.1%}", ha="center", fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_facecolor("#f8fafc")
        fig.patch.set_facecolor("white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.info(f"🔴 Kategori **Depresi** memiliki error rate **{class_errors[1]:.1%}** — "
                f"ini wajar karena data sangat tidak seimbang ({n_depresi} dari {total} sampel).")

# ─────────────────────────────────────────────────────────────────────────────
# TAHAP 5: Analisis Tingkat Keyakinan
# ─────────────────────────────────────────────────────────────────────────────
if stages[4]:
    st.markdown('<hr class="stage-divider">', unsafe_allow_html=True)
    st.markdown('<span class="stage-pill">Tahap 5</span>', unsafe_allow_html=True)
    st.subheader("🎯 Analisis Tingkat Keyakinan Prediksi")

    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.markdown('<div class="info-box">ℹ️ <b>Non-Visual:</b> Tabel probabilitas prediksi untuk 20 sampel pertama.</div>',
                    unsafe_allow_html=True)
        proba_df = pd.DataFrame({
            "Sampel": range(1, 21),
            "P(Normal) %": (y_proba[:20, 0] * 100).round(1),
            "P(Depresi) %": (y_proba[:20, 1] * 100).round(1),
            "Prediksi": [LABEL_MAP[p] for p in y_pred[:20]],
            "Aktual": [LABEL_MAP[a] for a in y_test.values[:20]],
        })
        proba_df["✓"] = proba_df["Prediksi"] == proba_df["Aktual"]
        proba_df["✓"] = proba_df["✓"].map({True: "✅", False: "❌"})
        st.dataframe(proba_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown('<div class="info-box">🔵 <b>Visual:</b> Sebaran keyakinan model — titik di atas garis merah = model percaya diri.</div>',
                    unsafe_allow_html=True)
        max_proba = y_proba.max(axis=1)
        threshold = 1 / rf.n_classes_

        fig, ax = plt.subplots(figsize=(7, 4))
        colors_pt = ["#6366f1" if p == 0 else "#ef4444" for p in y_pred]
        ax.scatter(range(len(max_proba)), max_proba, c=colors_pt,
                   alpha=0.55, s=22, edgecolors="none")
        ax.axhline(threshold, color="#ef4444", linestyle="--",
                   linewidth=1.8, label=f"Threshold keyakinan ({threshold:.2f})")
        ax.set_xlabel("Indeks Sampel (Test Set)", fontsize=10)
        ax.set_ylabel("Probabilitas Tertinggi", fontsize=10)
        ax.set_title("Tingkat Keyakinan Model per Sampel", fontsize=11, fontweight="bold")
        legend_handles = [
            mpatches.Patch(color="#6366f1", label="Prediksi Normal"),
            mpatches.Patch(color="#ef4444", label="Prediksi Depresi"),
        ]
        ax.legend(handles=legend_handles + [
            plt.Line2D([0], [0], color="#ef4444", linestyle="--", linewidth=1.8,
                       label=f"Threshold ({threshold:.2f})")
        ], fontsize=8)
        ax.set_facecolor("#f8fafc")
        fig.patch.set_facecolor("white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# TAHAP 6: Feature Importance
# ─────────────────────────────────────────────────────────────────────────────
if stages[5]:
    st.markdown('<hr class="stage-divider">', unsafe_allow_html=True)
    st.markdown('<span class="stage-pill">Tahap 6</span>', unsafe_allow_html=True)
    st.subheader("🔑 Kesimpulan & Faktor Kunci (Feature Importance)")

    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.markdown('<div class="info-box">ℹ️ <b>Non-Visual:</b> Skor Mean Decrease Gini — seberapa penting tiap fitur.</div>',
                    unsafe_allow_html=True)
        importances = rf.feature_importances_
        feat_imp_df = pd.DataFrame({
            "Fitur": [FEATURE_LABELS.get(c, c) for c in FEATURE_COLS],
            "Kode Fitur": FEATURE_COLS,
            "Mean Decrease Gini": importances.round(4),
            "Persentase": (importances / importances.sum() * 100).round(2),
        }).sort_values("Mean Decrease Gini", ascending=False).reset_index(drop=True)
        feat_imp_df.index += 1
        st.dataframe(feat_imp_df[["Fitur", "Mean Decrease Gini", "Persentase"]],
                     use_container_width=True)
        top3 = feat_imp_df.head(3)["Fitur"].tolist()
        st.success(f"🏆 **Top 3 Faktor:** {', '.join(top3)}")

    with col2:
        st.markdown('<div class="info-box">📊 <b>Visual:</b> Grafik Feature Importance — faktor paling berpengaruh di atas.</div>',
                    unsafe_allow_html=True)
        sorted_idx = np.argsort(importances)
        feat_names_sorted = [FEATURE_LABELS.get(FEATURE_COLS[i], FEATURE_COLS[i])
                              for i in sorted_idx]

        palette = plt.cm.Blues(np.linspace(0.35, 0.85, len(sorted_idx)))
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.barh(feat_names_sorted, importances[sorted_idx],
                       color=palette, edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, importances[sorted_idx]):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=8.5)
        ax.set_xlabel("Mean Decrease Gini (Importansi Fitur)", fontsize=10)
        ax.set_title("Peringkat Faktor Penyebab Depresi", fontsize=11, fontweight="bold")
        ax.axvline(importances.mean(), color="#ef4444", linestyle="--",
                   linewidth=1.2, label="Rata-rata importansi")
        ax.legend(fontsize=8)
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        ax.set_facecolor("#f8fafc")
        fig.patch.set_facecolor("white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# BONUS: Eksplorasi Data Interaktif
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="stage-divider">', unsafe_allow_html=True)
st.markdown('<span class="stage-pill">Bonus</span>', unsafe_allow_html=True)
st.subheader("🔍 Eksplorasi Data Interaktif")

col1, col2, col3 = st.columns(3)
with col1:
    x_axis = st.selectbox("Sumbu X", FEATURE_COLS, index=6,
                           format_func=lambda c: FEATURE_LABELS.get(c, c))
with col2:
    y_axis = st.selectbox("Sumbu Y", FEATURE_COLS, index=7,
                           format_func=lambda c: FEATURE_LABELS.get(c, c))
with col3:
    gender_filter = st.multiselect("Filter Gender", ["male", "female"],
                                   default=["male", "female"])

df_filtered = df[df["gender"].isin(gender_filter)]
fig, ax = plt.subplots(figsize=(8, 4.5))
for label, color, marker in [(0, "#6366f1", "o"), (1, "#ef4444", "^")]:
    mask = df_filtered["depression_label"] == label
    ax.scatter(df_filtered.loc[mask, x_axis],
               df_filtered.loc[mask, y_axis],
               c=color, label=LABEL_MAP[label],
               alpha=0.45, s=25, marker=marker, edgecolors="none")
ax.set_xlabel(FEATURE_LABELS.get(x_axis, x_axis), fontsize=10)
ax.set_ylabel(FEATURE_LABELS.get(y_axis, y_axis), fontsize=10)
ax.set_title(f"{FEATURE_LABELS.get(x_axis,x_axis)} vs {FEATURE_LABELS.get(y_axis,y_axis)}",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.set_facecolor("#f8fafc")
fig.patch.set_facecolor("white")
ax.grid(alpha=0.25, linestyle="--")
plt.tight_layout()
st.pyplot(fig)
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# BONUS: Prediksi Sampel Baru
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="stage-divider">', unsafe_allow_html=True)
st.markdown('<span class="stage-pill">Bonus</span>', unsafe_allow_html=True)
st.subheader("🧪 Coba Prediksi Sampel Baru")
st.markdown("Masukkan data individu untuk melihat prediksi model secara langsung.")

with st.form("predict_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        p_age     = st.slider("Usia", 13, 19, 16)
        p_medsos  = st.slider("Jam Medsos/Hari", 0.0, 10.0, 4.0, 0.1)
        p_sleep   = st.slider("Jam Tidur", 3.0, 10.0, 7.0, 0.1)
    with c2:
        p_screen  = st.slider("Screen Time Sebelum Tidur", 0.0, 5.0, 1.5, 0.1)
        p_acad    = st.slider("Performa Akademik (GPA)", 0.0, 4.0, 3.0, 0.01)
        p_phys    = st.slider("Aktivitas Fisik (jam/hari)", 0.0, 3.0, 1.0, 0.1)
    with c3:
        p_stress  = st.slider("Tingkat Stres (1–10)", 1, 10, 5)
        p_anxiety = st.slider("Tingkat Kecemasan (1–10)", 1, 10, 5)
        p_addict  = st.slider("Tingkat Kecanduan (1–10)", 1, 10, 5)
    submitted = st.form_submit_button("🔮 Prediksi Sekarang", use_container_width=True)

if submitted:
    sample = np.array([[p_age, p_medsos, p_sleep, p_screen,
                        p_acad, p_phys, p_stress, p_anxiety, p_addict]])
    pred  = rf.predict(sample)[0]
    proba = rf.predict_proba(sample)[0]
    label = LABEL_MAP[pred]

    if pred == 0:
        st.success(f"✅ **Prediksi: {label}**  — Keyakinan Model: `{proba[0]*100:.1f}%` Normal · `{proba[1]*100:.1f}%` Depresi")
    else:
        st.error(f"⚠️ **Prediksi: {label}**  — Keyakinan Model: `{proba[0]*100:.1f}%` Normal · `{proba[1]*100:.1f}%` Depresi")

    fig, ax = plt.subplots(figsize=(5, 2))
    colors_b = ["#6366f1", "#ef4444"]
    ax.barh(["Normal", "Depresi"], proba * 100, color=colors_b,
            height=0.45, edgecolor="white")
    for i, v in enumerate(proba * 100):
        ax.text(v + 1, i, f"{v:.1f}%", va="center", fontweight="bold", fontsize=11)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Probabilitas (%)")
    ax.set_title("Distribusi Probabilitas Prediksi", fontsize=10, fontweight="bold")
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Footer
st.markdown("""
<div style="text-align:center; color:#94a3b8; font-size:.8rem; margin-top:3rem; padding-top:1.5rem;
     border-top: 1px solid #e2e8f0;">
  Teen Mental Health Dashboard · Random Forest Classifier · Built with Streamlit & scikit-learn
</div>
""", unsafe_allow_html=True)