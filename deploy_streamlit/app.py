"""Aplikasi Streamlit: Prediksi Skor Kesehatan Mental berbasis Machine Learning dan Explainable AI (SHAP).

Jalankan:  streamlit run app.py
Prasyarat: folder artifacts/ (hasil `python train_export.py ...`)
"""
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from utils import (FEATURES, NUM, CAT, load_artifacts, make_codec, build_explainer, explain_rows)

st.set_page_config(page_title="Prediksi Skor Kesehatan Mental", page_icon="🧠", layout="wide")

LABELS = {
    "Daily_AI_Tool_Usage_Hours": "Penggunaan alat AI (jam/hari)",
    "Daily_Social_Media_Hours": "Penggunaan media sosial (jam/hari)",
    "Sleep_Hours": "Durasi tidur (jam/hari)",
    "Physical_Activity_Hours": "Aktivitas fisik (jam/hari)",
    "Age": "Usia (tahun)",
    "Gender": "Gender",
    "Education_Level": "Jenjang pendidikan",
}


# ---------------------------------------------------------------- memuat artefak
@st.cache_resource(show_spinner="Memuat model...")
def load_all():
    model, meta, background = load_artifacts("artifacts")
    gmap, emap = make_codec(meta)
    explainer = build_explainer(model, background, gmap, emap)
    return model, meta, background, gmap, emap, explainer


try:
    model, meta, background, gmap, emap, explainer = load_all()
except Exception as e:  # artefak belum dibuat
    st.error("Artefak model belum ditemukan. Jalankan terlebih dahulu: "
             "`python train_export.py --data AI_SocialMedia_Student_Dataset.csv`")
    st.caption(f"Detail teknis: {e}")
    st.stop()

rng = meta["numeric_ranges"]
tg = meta["target"]


def derive_education(age):
    for level, (lo, hi) in meta["edu_age_ranges"].items():
        if lo <= age <= hi:
            return level
    return meta["categories"]["Education_Level"][0]


def clip_score(v):
    return float(np.clip(v, 0, 100))


@st.cache_data(show_spinner=False)
def explain_one(values: tuple):
    row = pd.DataFrame([dict(zip(FEATURES, values))])
    sv, base = explain_rows(explainer, row, gmap, emap, max_evals=300)
    return sv.iloc[0].to_dict(), base


# ---------------------------------------------------------------- header
st.title("🧠 Prediksi Skor Kesehatan Mental")
st.caption("Model regresi (%s) dengan penjelasan SHAP. Dilatih pada %s observasi, %s data latih dan %s data uji."
           % (meta["model_name"], f"{meta['n_total']:,}".replace(",", "."), f"{meta['n_train']:,}".replace(",", "."), f"{meta['n_test']:,}".replace(",", ".")))
st.warning(
    "**Alat bantu riset, bukan alat diagnosis.** Model hanya menjelaskan sekitar %.0f%% variasi skor (R² = %.2f) "
    "dan menunjukkan asosiasi, bukan hubungan sebab-akibat. Hasilnya tidak boleh dipakai untuk menilai atau melabeli individu. "
    "Jika Anda merasa tertekan, hubungi tenaga profesional atau layanan konseling kampus."
    % (meta["metrics"]["R2"] * 100, meta["metrics"]["R2"]))

# ---------------------------------------------------------------- input (sidebar)
st.sidebar.header("Masukan variabel")
st.sidebar.caption("Rentang slider mengikuti rentang data latih agar prediksi tidak keluar dari wilayah data.")


def num_slider(feat, step=0.1):
    lo, hi = float(np.floor(rng[feat]["min"])), float(np.ceil(rng[feat]["max"]))
    default = float(round(rng[feat]["mean"], 1))
    return st.sidebar.slider(LABELS[feat], lo, hi, default, step)


social = num_slider("Daily_Social_Media_Hours")
ai = num_slider("Daily_AI_Tool_Usage_Hours")
sleep = num_slider("Sleep_Hours")
activity = num_slider("Physical_Activity_Hours")
age = st.sidebar.slider(LABELS["Age"], int(rng["Age"]["min"]), int(rng["Age"]["max"]), int(round(rng["Age"]["mean"])), 1)
gender = st.sidebar.selectbox(LABELS["Gender"], meta["categories"]["Gender"])

derived_edu = derive_education(age)
manual = st.sidebar.checkbox("Atur jenjang pendidikan manual", value=False)
if manual:
    edu = st.sidebar.selectbox(LABELS["Education_Level"], meta["categories"]["Education_Level"],
                               index=meta["categories"]["Education_Level"].index(derived_edu))
    lo, hi = meta["edu_age_ranges"][edu]
    if not (lo <= age <= hi):
        st.sidebar.warning(f"Pada data latih, {edu} hanya muncul pada usia {lo} sampai {hi}. Kombinasi ini tidak pernah dilihat model.")
else:
    edu = derived_edu
    st.sidebar.info(f"Jenjang pendidikan: **{edu}** (diturunkan dari usia, karena pada data latih jenjang ditentukan oleh usia).")

values = (ai, social, sleep, activity, float(age), gender, edu)
row = pd.DataFrame([dict(zip(FEATURES, values))])
pred = clip_score(model.predict(row)[0])

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Prediksi", "Penjelasan SHAP", "Simulasi", "Prediksi massal (CSV)", "Tentang model"])

# ---------------------------------------------------------------- tab 1: prediksi
with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Skor kesehatan mental (prediksi)", f"{pred:.1f}", f"{pred - tg['mean']:+.1f} dari rata-rata data latih ({tg['mean']:.1f})")
    cut1, cut2 = tg["tertile_cuts"]
    kategori = "Rendah" if pred <= cut1 else "Sedang" if pred <= cut2 else "Tinggi"
    c2.metric("Posisi relatif (tertil data latih)", kategori, f"≤ {cut1:.1f} rendah, > {cut2:.1f} tinggi", delta_color="off")
    pct = float(np.interp(pred, tg["quantiles"], np.linspace(0, 100, 101)))
    c3.metric("Persentil pada data latih", f"{pct:.0f}", "dari 100", delta_color="off")

    rq = meta["residual_quantiles"]
    lo90, hi90 = clip_score(pred + rq["q05"]), clip_score(pred + rq["q95"])
    st.progress(int(round(pred)), text=f"Skor prediksi {pred:.1f} dari 100")
    st.info(f"**Kisaran galat 90%:** {lo90:.1f} sampai {hi90:.1f}. Kisaran ini lebar karena galat tipikal model (RMSE) sekitar "
            f"{meta['metrics']['RMSE']:.1f} poin. Gunakan angka di atas sebagai gambaran pola kelompok, bukan nilai pasti seseorang.")
    st.dataframe(pd.DataFrame({"Variabel": [LABELS[f] for f in FEATURES], "Nilai": [str(round(v, 2)) if isinstance(v, float) else v for v in values]}),
                 hide_index=True)

# ---------------------------------------------------------------- tab 2: SHAP
with tab2:
    st.subheader("Mengapa model memprediksi skor ini?")
    st.caption("Nilai SHAP menunjukkan seberapa jauh setiap variabel menggeser prediksi dari nilai dasar (rata-rata prediksi model).")
    if st.toggle("Hitung penjelasan SHAP", value=True):
        with st.spinner("Menghitung SHAP..."):
            contrib, base = explain_one(values)
        s = pd.Series(contrib).reindex(FEATURES)
        order = s.abs().sort_values().index
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.barh([LABELS[f] for f in order], s[order].values, color=["#d62728" if v < 0 else "#1f77b4" for v in s[order].values], edgecolor="black")
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Kontribusi terhadap prediksi (poin skor)")
        ax.set_title(f"Nilai dasar {base:.1f}  →  prediksi {base + s.sum():.1f}")
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("Biru menaikkan prediksi, merah menurunkannya. Total kontribusi ditambah nilai dasar sama dengan prediksi model.")
        tbl = pd.DataFrame({"Variabel": [LABELS[f] for f in s.index], "Nilai": [values[FEATURES.index(f)] for f in s.index],
                            "Kontribusi SHAP (poin)": s.round(2).values}).sort_values("Kontribusi SHAP (poin)", key=lambda x: x.abs(), ascending=False)
        tbl["Nilai"] = tbl["Nilai"].astype(str)
        st.dataframe(tbl, hide_index=True)
        top = s.abs().idxmax()
        st.markdown(f"**Faktor paling berpengaruh pada kasus ini:** {LABELS[top]} ({s[top]:+.2f} poin).")

# ---------------------------------------------------------------- tab 3: simulasi
with tab3:
    st.subheader("Simulasi: bagaimana prediksi berubah jika satu variabel diubah?")
    st.caption("Variabel lain dipertahankan sesuai masukan di sidebar. Kurva menggambarkan asosiasi yang dipelajari model, bukan efek intervensi.")
    var = st.selectbox("Variabel yang diubah", [f for f in NUM if f != "Age"], format_func=lambda f: LABELS[f])
    grid = np.linspace(np.floor(rng[var]["min"]), np.ceil(rng[var]["max"]), 60)
    sim = pd.concat([row] * len(grid), ignore_index=True)
    sim[var] = grid
    curve = np.clip(model.predict(sim), 0, 100)
    cur_val = row[var].iloc[0]
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(grid, curve, color="#1f77b4", lw=2)
    ax2.scatter([cur_val], [pred], color="red", zorder=5, label="Masukan saat ini")
    ax2.set_xlabel(LABELS[var]); ax2.set_ylabel("Prediksi skor"); ax2.legend(); ax2.grid(alpha=.3)
    plt.tight_layout(); st.pyplot(fig2)
    delta = curve[-1] - curve[0]
    st.write(f"Dari nilai terendah hingga tertinggi pada data latih, prediksi berubah **{delta:+.1f} poin**.")

# ---------------------------------------------------------------- tab 4: batch
with tab4:
    st.subheader("Prediksi massal dari berkas CSV")
    st.caption("Kolom wajib: " + ", ".join(FEATURES))
    template = pd.DataFrame([dict(zip(FEATURES, values))])
    st.download_button("Unduh templat CSV", template.to_csv(index=False).encode("utf-8"), "templat_prediksi.csv", "text/csv")
    up = st.file_uploader("Unggah CSV", type=["csv"])
    if up is not None:
        try:
            df_in = pd.read_csv(up)
            missing = [c for c in FEATURES if c not in df_in.columns]
            if missing:
                st.error("Kolom tidak ditemukan: " + ", ".join(missing))
            else:
                bad_g = ~df_in["Gender"].isin(meta["categories"]["Gender"])
                bad_e = ~df_in["Education_Level"].isin(meta["categories"]["Education_Level"])
                if bad_g.any() or bad_e.any():
                    st.error(f"Nilai kategori tidak dikenal pada {int((bad_g | bad_e).sum())} baris. "
                             f"Gender: {meta['categories']['Gender']}; Education_Level: {meta['categories']['Education_Level']}.")
                else:
                    out = df_in.copy()
                    out["Prediksi_Mental_Health_Score"] = np.clip(model.predict(df_in[FEATURES]), 0, 100).round(2)
                    st.success(f"{len(out)} baris diproses.")
                    st.dataframe(out)
                    st.download_button("Unduh hasil", out.to_csv(index=False).encode("utf-8"), "hasil_prediksi.csv", "text/csv")
        except Exception as e:
            st.error(f"Berkas tidak dapat diproses: {e}")

# ---------------------------------------------------------------- tab 5: tentang
with tab5:
    st.subheader("Kinerja model pada data uji")
    m = meta["metrics"]
    a, b, c, d = st.columns(4)
    a.metric("MAE", f"{m['MAE']:.3f}"); b.metric("RMSE", f"{m['RMSE']:.3f}"); c.metric("R²", f"{m['R2']:.3f}")
    d.metric("RMSE baseline (rata-rata)", f"{m['baseline_RMSE']:.3f}")
    st.subheader("Kepentingan variabel global (rata-rata |SHAP|)")
    g = pd.Series(meta["shap_global"]).sort_values(ascending=False)
    st.bar_chart(pd.DataFrame({"Rata-rata |SHAP|": g.values}, index=[LABELS[i] for i in g.index]))
    st.caption(f"Dihitung pada {meta['shap_n']} sampel data uji.")
    st.subheader("Catatan penting")
    st.markdown(
        "- Model menjelaskan sebagian kecil variasi skor, sehingga prediksi individual memiliki galat yang besar.\n"
        "- SHAP menjelaskan cara model memakai variabel, bukan hubungan sebab-akibat.\n"
        "- Dataset bersifat sekunder dan tidak dikumpulkan oleh peneliti; generalisasi ke populasi lain harus hati-hati.\n"
        "- Jenjang pendidikan pada data latih ditentukan sepenuhnya oleh usia, sehingga kontribusi keduanya perlu dibaca bersama.")
    st.caption(f"Model: {meta['model_name']} | dilatih {meta['trained_at']} | scikit-learn {meta['versions']['scikit_learn']} | Python {meta['versions']['python']}")
