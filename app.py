import streamlit as st
import numpy as np
from PIL import Image
import os
from datetime import datetime
import pandas as pd

# ================================================================
# KONFIGURASI HALAMAN
# ================================================================

st.set_page_config(
    page_title="SmartEgg Detect — Klasifikasi Telur Ayam",
    page_icon="🥚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# CSS CUSTOM — TEMA HANGAT & NATURAL
# ================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Lato:wght@300;400;700&display=swap');

:root {
    --cream      : #FDF6EC;
    --warm-white : #FFF8F0;
    --amber      : #D4820A;
    --amber-light: #F5A623;
    --brown      : #6B3F1F;
    --brown-light: #A0613A;
    --green      : #4A7C59;
    --green-light: #6BAD7E;
    --red        : #C0392B;
    --text-dark  : #2C1A0E;
    --text-mid   : #5A3E28;
    --border     : #E8D5B7;
    --shadow     : rgba(107, 63, 31, 0.15);
}

.stApp { background-color: var(--cream); font-family: 'Lato', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }

section[data-testid="stSidebar"] {
    display:     flex        !important;
    visibility:  visible     !important;
    width:       280px       !important;
    min-width:   280px       !important;
    max-width:   280px       !important;
    transform:   none        !important;
    left:        0           !important;
    background: linear-gradient(180deg, #3D1F0A 0%, #6B3F1F 60%, #A0613A 100%) !important;
    border-right: 3px solid var(--amber) !important;
}

button[data-testid="collapsedControl"],
[data-testid="collapsedControl"] {
    display: none !important;
    visibility: hidden !important;
}

section[data-testid="stSidebar"] * { color: #FDF6EC !important; }
section[data-testid="stSidebar"] .stRadio label { cursor: pointer; }

.app-header {
    background: linear-gradient(135deg, #3D1F0A 0%, #6B3F1F 50%, #A0613A 100%);
    border-radius: 20px; padding: 2rem 3rem; margin-bottom: 2rem;
    text-align: center; position: relative; overflow: hidden;
    box-shadow: 0 8px 32px var(--shadow);
}
.app-header::before {
    content: ''; position: absolute; top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(245,166,35,0.15) 0%, transparent 60%);
    animation: pulse 4s ease-in-out infinite;
}
@keyframes pulse {
    0%,100% { transform: scale(1); opacity: 0.5; }
    50%     { transform: scale(1.1); opacity: 1; }
}
.app-title {
    font-family: 'Playfair Display', serif; font-size: 3rem;
    font-weight: 700; color: #FDF6EC; margin: 0;
    text-shadow: 2px 2px 8px rgba(0,0,0,0.3); position: relative;
}
.app-subtitle {
    font-size: 1rem; color: var(--amber-light); letter-spacing: 3px;
    text-transform: uppercase; margin-top: 0.5rem; position: relative;
}

.card {
    background: var(--warm-white); border-radius: 16px; padding: 1.5rem;
    border: 1px solid var(--border); box-shadow: 0 4px 20px var(--shadow);
    margin-bottom: 1rem;
}
.card-title {
    font-family: 'Playfair Display', serif; font-size: 1.3rem;
    color: var(--brown); border-bottom: 2px solid var(--amber-light);
    padding-bottom: 0.5rem; margin-bottom: 1rem;
}

.upload-area {
    background: linear-gradient(135deg, #FFF8F0, #FDF6EC);
    border: 3px dashed var(--amber-light); border-radius: 20px;
    padding: 2.5rem; text-align: center;
}

.badge-fertil {
    background: linear-gradient(135deg, #2D6A4F, #4A7C59);
    color: white; padding: 1rem 2rem; border-radius: 50px;
    font-size: 1.5rem; font-weight: 700; display: inline-block;
    box-shadow: 0 4px 15px rgba(74,124,89,0.4);
    font-family: 'Playfair Display', serif;
}
.badge-abnormal {
    background: linear-gradient(135deg, #922B21, #C0392B);
    color: white; padding: 1rem 2rem; border-radius: 50px;
    font-size: 1.5rem; font-weight: 700; display: inline-block;
    box-shadow: 0 4px 15px rgba(192,57,43,0.4);
    font-family: 'Playfair Display', serif;
}
.badge-infertil {
    background: linear-gradient(135deg, #B7770D, #D4820A);
    color: white; padding: 1rem 2rem; border-radius: 50px;
    font-size: 1.5rem; font-weight: 700; display: inline-block;
    box-shadow: 0 4px 15px rgba(212,130,10,0.4);
    font-family: 'Playfair Display', serif;
}

.stat-box {
    background: var(--warm-white); border-radius: 14px; padding: 1.2rem;
    text-align: center; border: 1px solid var(--border);
    box-shadow: 0 2px 12px var(--shadow);
}
.stat-number {
    font-family: 'Playfair Display', serif; font-size: 2.5rem;
    font-weight: 700; line-height: 1;
}
.stat-label {
    font-size: 0.8rem; letter-spacing: 1px; text-transform: uppercase;
    color: var(--text-mid); margin-top: 0.3rem;
}

.conf-bar-bg {
    background: #E8D5B7; border-radius: 10px; height: 12px;
    margin: 4px 0 10px 0; overflow: hidden;
}
.conf-bar-fill { height: 100%; border-radius: 10px; }

.info-fertil   { background:#D5F5E3; border-left:5px solid #4A7C59; border-radius:8px; padding:1rem; margin-bottom:0.5rem; }
.info-abnormal { background:#FADBD8; border-left:5px solid #C0392B; border-radius:8px; padding:1rem; margin-bottom:0.5rem; }
.info-infertil { background:#FDEBD0; border-left:5px solid #D4820A; border-radius:8px; padding:1rem; margin-bottom:0.5rem; }

.divider {
    border: none; height: 2px;
    background: linear-gradient(90deg, transparent, var(--amber-light), transparent);
    margin: 1.5rem 0;
}

.stButton > button {
    background: linear-gradient(135deg, #6B3F1F, #A0613A);
    color: white; border: none; border-radius: 10px;
    padding: 0.6rem 1.5rem; font-weight: 700; transition: all 0.3s;
    box-shadow: 0 4px 12px var(--shadow);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #A0613A, #D4820A);
    transform: translateY(-2px);
}
</style>
""", unsafe_allow_html=True)

# ================================================================
# SESSION STATE
# ================================================================

for key, val in [
    ('riwayat', []), ('total_scan', 0),
    ('total_fertil', 0), ('total_abnormal', 0),
    ('total_infertil', 0), ('last_saved_id', '')
]:
    if key not in st.session_state:
        st.session_state[key] = val

# ================================================================
# KONFIGURASI MODEL
# ================================================================

CLASS_NAMES = ['abnormal', 'fertil', 'infertil']
IMG_SIZE    = 224
CONFIDENCE_THRESHOLD = 60.0

AKURASI_MODEL = {
    "ResNet50"      : 0.9630,
    "EfficientNetB0": 0.9568,
    "MobileNetV2"   : 0.8395,
}

GDRIVE_IDS = {
    "ResNet50"      : "1OKfavO7OhVVJAhS5H4AghUr8oTT2JWTc",
    "EfficientNetB0": "1yPUgImX_FXnWOWFY9OqbR5K2rzEjxpfG",
    "MobileNetV2"   : "1YRpEnykZjv9PMEn2UIE2E3KEEeZhz8J_",
}

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

WARNA = {"fertil":"#4A7C59", "abnormal":"#C0392B", "infertil":"#D4820A"}
EMOJI = {"fertil":"🟢",      "abnormal":"🔴",       "infertil":"🟡"}
DESKRIPSI = {
    "fertil"  : "Telur telah dibuahi dan embrio berkembang dengan baik. Ditandai dengan jaringan pembuluh darah yang terlihat jelas saat dicandling.",
    "abnormal": "Telur mengalami kematian embrio atau tidak berkembang. Biasanya terlihat gelap atau terdapat bercak darah tanpa jaringan.",
    "infertil": "Telur tidak dibuahi sehingga tidak terdapat embrio. Tampak tembus cahaya dan jernih saat proses candling dilakukan.",
}
SARAN = {
    "fertil"  : "✅ Lanjutkan inkubasi. Pantau suhu (37.5–38°C) dan kelembaban (55–60%) secara rutin.",
    "abnormal": "⚠️ Segera keluarkan dari mesin tetas untuk mencegah kontaminasi telur lain.",
    "infertil": "ℹ️ Keluarkan dari mesin tetas. Dapat digunakan untuk konsumsi jika masih segar.",
}

# ================================================================
# LOAD MODEL
# ================================================================

@st.cache_resource(show_spinner="⏳ Memuat model AI... (download pertama mungkin butuh beberapa menit)")
def load_best_model():
    import gdown
    os.environ["KERAS_BACKEND"] = "tensorflow"
    from keras.models import load_model
    from keras.applications.resnet50     import preprocess_input as pre_resnet
    from keras.applications.efficientnet import preprocess_input as pre_efficient
    from keras.applications.mobilenet_v2 import preprocess_input as pre_mobile

    preprocess_map = {
        "ResNet50"      : pre_resnet,
        "EfficientNetB0": pre_efficient,
        "MobileNetV2"   : pre_mobile,
    }

    for nama in sorted(AKURASI_MODEL, key=AKURASI_MODEL.get, reverse=True):
        path = os.path.join(MODEL_DIR, f"best_{nama}.h5")

        if not os.path.exists(path):
            file_id = GDRIVE_IDS.get(nama, "")
            url = f"https://drive.google.com/uc?id={file_id}"
            try:
                gdown.download(url, path, quiet=False)
            except Exception as e:
                st.warning(f"⚠️ Gagal download {nama}: {e}")
                continue

        if os.path.exists(path):
            return load_model(path), nama, preprocess_map[nama]

    raise FileNotFoundError(
        "Tidak ada model yang berhasil dimuat dari Google Drive. "
        "Pastikan FILE_ID sudah benar dan file dibagikan secara publik."
    )

try:
    model, nama_model, preprocess_fn = load_best_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    error_msg    = str(e)

# ================================================================
# FUNGSI PREDIKSI
# ================================================================

def prediksi(img_pil: Image.Image):
    img_arr  = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)
    img_proc = preprocess_fn(np.expand_dims(img_arr, axis=0))
    probs    = model.predict(img_proc, verbose=0)[0]
    idx      = np.argmax(probs)
    conf     = float(probs[idx]) * 100
    is_valid = conf >= CONFIDENCE_THRESHOLD
    return CLASS_NAMES[idx], conf, probs, is_valid

# ================================================================
# SIDEBAR
# ================================================================

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0;'>
        <div style='font-size:4rem;'>🥚</div>
        <div style='font-family:Playfair Display,serif; font-size:1.4rem;
                    font-weight:700; color:#F5A623;'>SmartEgg Detect</div>
        <div style='font-size:0.75rem; letter-spacing:2px; color:#E8D5B7;
                    text-transform:uppercase;'>AI Candling System</div>
    </div>
    <hr style='border-color:#A0613A; margin:1rem 0;'>
    """, unsafe_allow_html=True)

    menu = st.radio("📋 Menu", [
        "🔬 Deteksi Telur",
        "📊 Statistik Sesi",
        "📜 Riwayat Scan",
        "📖 Panduan Candling",
        "ℹ️ Tentang Aplikasi"
    ], label_visibility="collapsed")

    st.markdown("<hr style='border-color:#A0613A;'>", unsafe_allow_html=True)

    if model_loaded:
        st.markdown(f"""
        <div style='background:rgba(74,124,89,0.2); border-radius:10px;
                    padding:0.8rem; border:1px solid #4A7C59;'>
            <div style='color:#6BAD7E; font-size:0.8rem; font-weight:700;'>
                ✅ MODEL AKTIF</div>
            <div style='color:#FDF6EC; font-size:0.85rem; margin-top:4px;'>
                {nama_model}</div>
            <div style='color:#E8D5B7; font-size:0.75rem;'>
                Akurasi: {AKURASI_MODEL[nama_model]*100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("❌ Model tidak ditemukan!")

    st.markdown(f"""
    <div style='margin-top:1rem; font-size:0.75rem; color:#E8D5B7; text-align:center;'>
        Total scan sesi ini<br>
        <span style='font-size:2rem; font-weight:700; color:#F5A623;'>
            {st.session_state.total_scan}</span>
    </div>
    """, unsafe_allow_html=True)

# ================================================================
# HEADER
# ================================================================

st.markdown("""
<div class='app-header'>
    <div class='app-title'>🥚 SmartEgg Detect</div>
    <div class='app-subtitle'>Sistem Deteksi Fertilitas Telur Ayam Berbasis AI</div>
    <div style='color:#E8D5B7; font-size:0.85rem; margin-top:0.5rem; position:relative;'>
        Teknologi CNN Fine-Tuning · ResNet50 · EfficientNetB0 · MobileNetV2
    </div>
</div>
""", unsafe_allow_html=True)

# ================================================================
# HALAMAN: DETEKSI TELUR
# ================================================================

if menu == "🔬 Deteksi Telur":

    if not model_loaded:
        st.error(f"❌ **Model tidak dapat dimuat.**\n\n```\n{error_msg}\n```")
        st.stop()

    col_upload, col_hasil = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>📸 Upload Gambar Telur</div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size:0.85rem; color:#5A3E28; margin-bottom:1rem;'>
            Upload foto telur hasil candling untuk dianalisis oleh sistem AI.
            Pastikan gambar jelas dan telur berada di tengah frame.
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Pilih gambar (JPG, PNG)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            max_h = 350
            w, h  = img.size
            if h > max_h:
                ratio       = max_h / h
                display_img = img.resize((int(w * ratio), max_h), Image.LANCZOS)
            else:
                display_img = img
            st.image(display_img,
                     caption=f"📁 {uploaded.name}  |  📐 {w}×{h} px",
                     use_container_width=False)
        else:
            st.markdown("""
            <div class='upload-area'>
                <div style='font-size:3rem;'>🥚</div>
                <div style='color:#A0613A; font-weight:700; margin:0.5rem 0;'>
                    Drag & Drop atau Klik Upload
                </div>
                <div style='font-size:0.8rem; color:#8B6914;'>Format: JPG, JPEG, PNG</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='card' style='background:#FFF3E0;'>
            <div class='card-title' style='font-size:1rem;'>💡 Tips Foto yang Baik</div>
            <ul style='font-size:0.82rem; color:#5A3E28; margin:0; padding-left:1.2rem;'>
                <li>Gunakan ruangan gelap saat candling</li>
                <li>Arahkan cahaya dari bawah/belakang telur</li>
                <li>Pastikan telur berada di tengah frame</li>
                <li>Hindari gambar blur atau terlalu gelap</li>
                <li>Idealnya foto di hari ke-7 inkubasi</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_hasil:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>🔬 Hasil Analisis</div>", unsafe_allow_html=True)

        if uploaded:
            with st.spinner("🔍 Menganalisis telur..."):
                kelas, conf, probs, is_valid = prediksi(img)

            if not is_valid:
                st.markdown("""
                <div style='text-align:center; padding:2rem; background:#FFF3CD;
                            border:2px solid #FFC107; border-radius:16px; margin:1rem 0;'>
                    <div style='font-size:3rem;'>⚠️</div>
                    <div style='font-family:Playfair Display,serif; font-size:1.3rem;
                                font-weight:700; color:#856404; margin:0.5rem 0;'>
                        Bukan Foto Telur
                    </div>
                    <div style='font-size:0.9rem; color:#856404;'>
                        Gambar yang diupload tidak terdeteksi sebagai telur ayam.<br>
                        Pastikan foto menampilkan telur dengan jelas saat proses candling.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.markdown(f"""
                <div style='text-align:center; margin:1rem 0;'>
                    <div class='badge-{kelas}'>{EMOJI[kelas]} {kelas.upper()}</div>
                    <div style='font-size:0.9rem; color:#5A3E28;
                                margin-top:0.8rem; font-style:italic;'>
                        {DESKRIPSI[kelas]}
                    </div>
                </div>
                <hr class='divider'>
                <div style='margin:1rem 0;'>
                    <div style='display:flex; justify-content:space-between;
                                font-size:0.85rem; color:#5A3E28; margin-bottom:4px;'>
                        <span>🎯 Tingkat Keyakinan Model</span>
                        <span style='font-weight:700; color:{WARNA[kelas]};'>{conf:.1f}%</span>
                    </div>
                    <div class='conf-bar-bg'>
                        <div class='conf-bar-fill'
                             style='width:{conf}%; background:{WARNA[kelas]};'></div>
                    </div>
                </div>
                <div style='font-size:0.85rem; font-weight:700; color:#5A3E28;
                            margin:1rem 0 0.5rem;'>📊 Probabilitas per Kelas:</div>
                """, unsafe_allow_html=True)

                for c, p in zip(CLASS_NAMES, probs):
                    pct  = float(p) * 100
                    bold = "font-weight:700;" if c == kelas else ""
                    st.markdown(f"""
                    <div style='margin-bottom:8px;'>
                        <div style='display:flex; justify-content:space-between;
                                    font-size:0.82rem; {bold} color:#2C1A0E;'>
                            <span>{EMOJI[c]} {c.capitalize()}</span>
                            <span style='color:{WARNA[c]};'>{pct:.2f}%</span>
                        </div>
                        <div class='conf-bar-bg'>
                            <div class='conf-bar-fill'
                                 style='width:{pct}%; background:{WARNA[c]}; opacity:0.85;'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(f"""
                <hr class='divider'>
                <div class='info-{kelas}'>
                    <div style='font-weight:700; font-size:0.9rem; margin-bottom:4px;'>
                        📋 Rekomendasi Tindakan:
                    </div>
                    <div style='font-size:0.85rem;'>{SARAN[kelas]}</div>
                </div>
                """, unsafe_allow_html=True)

                file_id = f"{uploaded.name}_{uploaded.size}"
                if st.session_state.last_saved_id != file_id:
                    st.session_state.riwayat.append({
                        "No"        : len(st.session_state.riwayat) + 1,
                        "Waktu"     : datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                        "File"      : uploaded.name,
                        "Hasil"     : kelas.upper(),
                        "Confidence": f"{conf:.1f}%",
                        "Model"     : nama_model,
                    })
                    st.session_state.total_scan        += 1
                    st.session_state[f"total_{kelas}"] += 1
                    st.session_state.last_saved_id      = file_id

        else:
            st.markdown("""
            <div style='text-align:center; padding:3rem 1rem; color:#A0613A;'>
                <div style='font-size:4rem; margin-bottom:1rem;'>🔍</div>
                <div style='font-family:Playfair Display,serif;
                            font-size:1.2rem; font-weight:600;'>Menunggu Gambar...</div>
                <div style='font-size:0.85rem; margin-top:0.5rem; color:#8B6914;'>
                    Upload gambar telur di sebelah kiri untuk memulai analisis
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ================================================================
# HALAMAN: STATISTIK SESI
# ================================================================

elif menu == "📊 Statistik Sesi":
    st.markdown("<div class='card-title' style='font-family:Playfair Display,serif; "
                "font-size:1.8rem; color:#6B3F1F;'>📊 Statistik Sesi</div>",
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label, color in [
        (c1, st.session_state.total_scan,     "Total Scan",  "#6B3F1F"),
        (c2, st.session_state.total_fertil,   "Fertil",      "#4A7C59"),
        (c3, st.session_state.total_abnormal, "Abnormal",    "#C0392B"),
        (c4, st.session_state.total_infertil, "Infertil",    "#D4820A"),
    ]:
        with col:
            st.markdown(f"""
            <div class='stat-box'>
                <div class='stat-number' style='color:{color};'>{val}</div>
                <div class='stat-label'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.total_scan > 0:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        fig.patch.set_facecolor('#FDF6EC')

        labels     = ['Fertil', 'Abnormal', 'Infertil']
        values     = [st.session_state.total_fertil,
                      st.session_state.total_abnormal,
                      st.session_state.total_infertil]
        colors_pie = ['#4A7C59', '#C0392B', '#D4820A']
        non_zero   = [(l,v,c) for l,v,c in zip(labels,values,colors_pie) if v > 0]

        if non_zero:
            l, v, c = zip(*non_zero)
            _, _, autotexts = axes[0].pie(
                v, labels=l, colors=c, autopct='%1.1f%%', startangle=90,
                textprops={'fontsize':11},
                wedgeprops={'edgecolor':'white','linewidth':2}
            )
            for at in autotexts:
                at.set_color('white'); at.set_fontweight('bold')
            axes[0].set_title('Distribusi Hasil Scan', fontsize=13,
                              fontweight='bold', color='#3D1F0A', pad=15)

        axes[1].bar(labels, values, color=colors_pie, alpha=0.85,
                    edgecolor='white', linewidth=2, width=0.5)
        for bar, v in zip(axes[1].patches, values):
            axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                         str(v), ha='center', va='bottom',
                         fontsize=12, fontweight='bold', color='#3D1F0A')
        axes[1].set_facecolor('#FFF8F0')
        axes[1].set_title('Jumlah per Kelas', fontsize=13,
                          fontweight='bold', color='#3D1F0A')
        axes[1].set_ylabel('Jumlah', color='#5A3E28')
        axes[1].tick_params(colors='#5A3E28')
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
        axes[1].grid(axis='y', alpha=0.3, color='#D4820A')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        total = st.session_state.total_scan
        st.markdown(f"""
        <div class='card' style='margin-top:1rem;'>
            <div class='card-title'>📈 Ringkasan Persentase</div>
            <div style='display:flex; gap:1rem; flex-wrap:wrap;'>
                <div class='info-fertil' style='flex:1; min-width:150px;'>
                    🟢 <b>Fertil:</b> {st.session_state.total_fertil/total*100:.1f}%
                </div>
                <div class='info-abnormal' style='flex:1; min-width:150px;'>
                    🔴 <b>Abnormal:</b> {st.session_state.total_abnormal/total*100:.1f}%
                </div>
                <div class='info-infertil' style='flex:1; min-width:150px;'>
                    🟡 <b>Infertil:</b> {st.session_state.total_infertil/total*100:.1f}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("📭 Belum ada data scan. Mulai deteksi telur untuk melihat statistik.")

# ================================================================
# HALAMAN: RIWAYAT SCAN
# ================================================================

elif menu == "📜 Riwayat Scan":
    st.markdown("<div class='card-title' style='font-family:Playfair Display,serif; "
                "font-size:1.8rem; color:#6B3F1F;'>📜 Riwayat Scan</div>",
                unsafe_allow_html=True)

    if st.session_state.riwayat:
        df = pd.DataFrame(st.session_state.riwayat)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Export ke CSV",
                data=csv,
                file_name=f"riwayat_ovoscan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_exp2:
            if st.button("🗑️ Hapus Riwayat", use_container_width=True):
                for key in ['riwayat','total_scan','total_fertil',
                            'total_abnormal','total_infertil']:
                    st.session_state[key] = [] if key == 'riwayat' else 0
                st.session_state.last_saved_id = ''
                st.rerun()
    else:
        st.info("📭 Belum ada riwayat scan. Mulai deteksi telur terlebih dahulu.")

# ================================================================
# HALAMAN: PANDUAN CANDLING
# ================================================================

elif menu == "📖 Panduan Candling":
    st.markdown("<div class='card-title' style='font-family:Playfair Display,serif; "
                "font-size:1.8rem; color:#6B3F1F;'>📖 Panduan Candling Telur Ayam</div>",
                unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🕯️ Apa itu Candling?", "📅 Waktu yang Tepat",
        "🔍 Cara Membaca Hasil", "⚙️ Alat & Tips"
    ])

    with tab1:
        st.markdown("""
        <div class='card'>
            <div class='card-title'>🕯️ Pengertian Candling</div>
            <p style='color:#5A3E28; line-height:1.8;'>
                <b>Candling</b> adalah teknik pemeriksaan telur tetas dengan menggunakan
                sumber cahaya kuat yang diarahkan dari belakang telur di ruangan gelap.
                Teknik ini memungkinkan peternak melihat kondisi bagian dalam telur
                tanpa harus memecahnya.
            </p>
            <hr class='divider'>
            <div style='display:flex; gap:1rem; flex-wrap:wrap;'>
                <div class='info-fertil' style='flex:1; min-width:200px;'>
                    <b>🟢 Telur Fertil</b><br>
                    Terlihat jaringan pembuluh darah seperti sarang laba-laba,
                    dan titik gelap (embrio) di bagian tengah.
                </div>
                <div class='info-abnormal' style='flex:1; min-width:200px;'>
                    <b>🔴 Telur Abnormal/Dead</b><br>
                    Embrio mati, terlihat bercak darah
                    atau massa gelap tanpa jaringan hidup.
                </div>
                <div class='info-infertil' style='flex:1; min-width:200px;'>
                    <b>🟡 Telur Infertil</b><br>
                    Tidak ada embrio, isi telur terlihat jernih dan
                    tembus cahaya sepenuhnya.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        waktu_data = {
            "Hari ke-": ["3–4", "7", "14", "18"],
            "Tujuan": ["Deteksi awal fertilitas", "Pemeriksaan utama — paling akurat",
                       "Cek perkembangan embrio", "Pemeriksaan akhir sebelum menetas"],
            "Yang Terlihat": [
                "Titik merah kecil (blastodisc) pada telur fertil",
                "Jaringan pembuluh darah jelas, embrio bergerak",
                "Embrio besar, ruang udara membesar",
                "Telur hampir penuh embrio, siap menetas"
            ]
        }
        st.table(pd.DataFrame(waktu_data))
        st.markdown("""
        <div class='info-fertil'>
            💡 <b>Rekomendasi:</b> Candling terbaik dilakukan pada hari ke-7
            karena pembuluh darah sudah cukup jelas terlihat.
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown("""
        <div class='card'>
            <div style='color:#5A3E28; line-height:1.8;'>
                <h4 style='color:#4A7C59;'>🟢 Telur FERTIL — Tanda-tanda:</h4>
                <ul>
                    <li>Jaringan pembuluh darah merah seperti bintang/laba-laba</li>
                    <li>Ada titik gelap (embrio) yang bisa bergerak jika digoyang perlahan</li>
                    <li>Ruang udara di ujung tumpul terlihat jelas</li>
                </ul>
                <h4 style='color:#C0392B; margin-top:1rem;'>🔴 Telur ABNORMAL — Tanda-tanda:</h4>
                <ul>
                    <li>Terlihat cincin darah (blood ring) tanpa jaringan hidup</li>
                    <li>Ada massa gelap tidak beraturan</li>
                    <li>Bau tidak sedap saat didekatkan ke hidung</li>
                </ul>
                <h4 style='color:#D4820A; margin-top:1rem;'>🟡 Telur INFERTIL — Tanda-tanda:</h4>
                <ul>
                    <li>Isi telur jernih dan tembus cahaya sepenuhnya</li>
                    <li>Tidak ada pembuluh darah sama sekali</li>
                    <li>Terlihat kuning telur bergerak bebas</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with tab4:
        st.markdown("""
        <div class='card'>
            <div style='color:#5A3E28;'>
                <h4>Alat yang Dibutuhkan:</h4>
                <ul style='line-height:2;'>
                    <li>🔦 <b>Candler / Ovoscope</b> — alat khusus candling telur</li>
                    <li>💡 <b>Senter LED terang</b> — alternatif candler</li>
                    <li>🏠 <b>Ruangan gelap</b> — kamar mandi atau lemari gelap</li>
                    <li>🧤 <b>Sarung tangan bersih</b> — jaga kebersihan telur</li>
                </ul>
                <hr class='divider'>
                <h4>Tips Penting:</h4>
                <div class='info-fertil'>✅ Cuci tangan sebelum memegang telur tetas</div>
                <div class='info-fertil'>✅ Batasi proses candling maksimal 5 menit per sesi</div>
                <div class='info-abnormal'>❌ Jangan mengguncang telur terlalu keras</div>
                <div class='info-abnormal'>❌ Jangan terlalu lama mengeluarkan telur dari mesin tetas</div>
                <div class='info-infertil'>⚠️ Segera kembalikan ke mesin tetas setelah pemeriksaan</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ================================================================
# HALAMAN: TENTANG APLIKASI
# ================================================================

elif menu == "ℹ️ Tentang Aplikasi":
    st.markdown("<div class='card-title' style='font-family:Playfair Display,serif; "
                "font-size:1.8rem; color:#6B3F1F;'>ℹ️ Tentang SmartEgg Detect</div>",
                unsafe_allow_html=True)

    model_info = nama_model if model_loaded else "Tidak tersedia"
    akurasi    = AKURASI_MODEL.get(nama_model, 0) * 100 if model_loaded else 0

    st.markdown(f"""
    <div class='card'>
        <div class='card-title'>🥚 SmartEgg Detect — AI Candling System</div>
        <p style='color:#5A3E28; line-height:1.8;'>
            SmartEgg Detect adalah aplikasi berbasis kecerdasan buatan (AI) yang dirancang
            untuk membantu peternak ayam dalam mengklasifikasikan tingkat fertilitas
            telur ayam melalui analisis citra candling secara otomatis.
        </p>
        <hr class='divider'>
        <div style='display:flex; gap:1rem; flex-wrap:wrap;'>
            <div class='card' style='flex:1; min-width:200px;'>
                <b>🧠 Teknologi</b><br>
                <span style='color:#5A3E28; font-size:0.85rem;'>
                    CNN Fine-Tuning dengan pretrained ImageNet weights.
                </span>
            </div>
            <div class='card' style='flex:1; min-width:200px;'>
                <b>📊 Model Aktif</b><br>
                <span style='color:#5A3E28; font-size:0.85rem;'>
                    {model_info}<br>Akurasi: {akurasi:.2f}%
                </span>
            </div>
            <div class='card' style='flex:1; min-width:200px;'>
                <b>🎯 Kelas Output</b><br>
                <span style='color:#5A3E28; font-size:0.85rem;'>
                    🔴 Abnormal<br>🟢 Fertil<br>🟡 Infertil
                </span>
            </div>
        </div>
    </div>
    <div class='card'>
        <div class='card-title'>🔬 Perbandingan Model</div>
        <table style='width:100%; border-collapse:collapse; font-size:0.85rem; color:#5A3E28;'>
            <tr style='background:#6B3F1F; color:white;'>
                <th style='padding:10px; text-align:left;'>Model</th>
                <th style='padding:10px; text-align:center;'>Akurasi</th>
                <th style='padding:10px; text-align:center;'>Kecepatan</th>
                <th style='padding:10px; text-align:left;'>Keunggulan</th>
            </tr>
            <tr style='background:#FFF8F0;'>
                <td style='padding:10px; font-weight:700;'>ResNet50</td>
                <td style='padding:10px; text-align:center; color:#4A7C59; font-weight:700;'>96.30%</td>
                <td style='padding:10px; text-align:center;'>⭐⭐⭐</td>
                <td style='padding:10px;'>Akurasi tinggi, fitur dalam</td>
            </tr>
            <tr style='background:#FDF6EC;'>
                <td style='padding:10px; font-weight:700;'>EfficientNetB0</td>
                <td style='padding:10px; text-align:center; color:#4A7C59; font-weight:700;'>95.68%</td>
                <td style='padding:10px; text-align:center;'>⭐⭐⭐⭐</td>
                <td style='padding:10px;'>Efisien & akurat</td>
            </tr>
            <tr style='background:#FFF8F0;'>
                <td style='padding:10px; font-weight:700;'>MobileNetV2</td>
                <td style='padding:10px; text-align:center; color:#D4820A; font-weight:700;'>83.95%</td>
                <td style='padding:10px; text-align:center;'>⭐⭐⭐⭐⭐</td>
                <td style='padding:10px;'>Ringan, cocok mobile</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

# ================================================================
# FOOTER
# ================================================================

st.markdown("""
<div style='text-align:center; padding:2rem; margin-top:2rem;
            border-top:2px solid #E8D5B7; color:#A0613A; font-size:0.8rem;'>
    <div style='font-family:Playfair Display,serif; font-size:1rem;
                color:#6B3F1F; margin-bottom:0.3rem;'>🥚 SmartEgg Detect</div>
    Sistem Klasifikasi Fertilitas Telur Ayam Berbasis CNN Fine-Tuning<br>
    Dibuat untuk keperluan penelitian skripsi · 2024
</div>
""", unsafe_allow_html=True)
