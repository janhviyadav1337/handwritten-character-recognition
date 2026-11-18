import sys
from pathlib import Path

import streamlit as st
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model_utils import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    load_class_names,
    load_model,
    predict_character,
    preprocess_uploaded_image,
)


st.set_page_config(
    page_title="ScriptAI | Handwritten Character Recognition",
    page_icon="S",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
    :html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #fbfbfe; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { display: none; }
    .block-container { max-width: 1180px; padding: 1.8rem 2rem 4rem; }
    .nav-wrap {
        display:flex; align-items:center; justify-content:space-between;
        padding: 12px 0 20px; border-bottom:1px solid #ececf5;
    }
    .brand { display:flex; align-items:center; gap:10px; font-family:'Manrope'; font-size:23px; font-weight:800; color:#17172a; }
    .brand-mark {
        width:38px; height:38px; border-radius:12px;
        background:linear-gradient(135deg,#7557d9,#4f78d7);
        display:grid; place-items:center; color:white; font-size:18px;
        box-shadow:0 8px 22px rgba(108,87,205,.24);
    }
    .nav-links { display:flex; gap:28px; color:#686879; font-size:14px; font-weight:600; }
    .nav-cta {
        background:#17172a; color:white !important; border-radius:12px;
        padding:11px 18px; text-decoration:none;
    }
    .hero { display:grid; grid-template-columns:1.08fr .92fr; align-items:center; gap:48px; padding:72px 0 58px; }
    .eyebrow {
        display:inline-flex; align-items:center; gap:8px; color:#6856bd;
        background:#f1effc; border:1px solid #e6e1fb; border-radius:999px;
        padding:8px 12px; font-size:13px; font-weight:700;
    }
    .hero h1 {
        font-family:'Manrope'; letter-spacing:-3px; line-height:1.02;
        font-size:68px; margin:22px 0 18px; color:#18182b;
    }
    .hero h1 span {
        background:linear-gradient(100deg,#7455d4,#467fd8);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    }
    .hero-copy { max-width:560px; color:#69697c; font-size:18px; line-height:1.72; }
    .hero-proof { display:flex; gap:24px; margin-top:26px; color:#77778a; font-size:13px; }
    .proof-dot { color:#6b55c8; font-weight:800; margin-right:6px; }
    .visual-card {
        min-height:410px; border-radius:28px; padding:28px;
        background:linear-gradient(145deg,#f2efff,#edf4ff);
        border:1px solid #e8e5f7; position:relative;
        box-shadow:0 24px 60px rgba(75,70,130,.12);
    }
    .visual-card:before {
        content:""; position:absolute; width:150px; height:150px; right:-28px; top:-28px;
        border-radius:999px; background:rgba(117,87,217,.08);
    }
    .ai-window {
        position:absolute; inset:70px 42px 48px; border-radius:22px;
        background:rgba(255,255,255,.92); border:1px solid white;
        box-shadow:0 18px 46px rgba(69,64,120,.14);
        display:grid; place-items:center;
    }
    .hand-char {
        width:160px; height:160px; border-radius:32px;
        display:grid; place-items:center; background:#f7f6ff;
        color:#765bd1; font-family:'Manrope'; font-size:112px; line-height:1;
        transform:rotate(-4deg); box-shadow:inset 0 0 0 1px #e8e4f8;
    }
    .scan {
        position:absolute; width:190px; height:3px; border-radius:99px;
        background:linear-gradient(90deg,transparent,#7458d2,transparent);
        box-shadow:0 0 18px rgba(116,88,210,.55); top:50%;
        animation:scan 3s ease-in-out infinite;
    }
    @keyframes scan { 0%,100% { transform:translateY(-85px); } 50% { transform:translateY(85px); } }
    .metric-chip {
        position:absolute; right:-12px; bottom:28px; background:white;
        border-radius:16px; padding:12px 15px; box-shadow:0 12px 32px rgba(69,64,120,.15);
        font-size:12px; color:#77778a;
    }
    .metric-chip strong { display:block; color:#6d55c5; font-size:20px; margin-top:2px; }
    .section { padding:54px 0; }
    .section-head { text-align:center; max-width:680px; margin:0 auto 32px; }
    .section-kicker { color:#735bd1; font-size:13px; font-weight:800; letter-spacing:.12em; text-transform:uppercase; }
    .section-title { font-family:'Manrope'; font-size:38px; letter-spacing:-1.5px; color:#18182b; margin:8px 0 10px; }
    .section-copy { color:#77778a; line-height:1.65; }
    .recognition-shell {
        background:white; border:1px solid #ededf4; border-radius:24px;
        padding:20px; box-shadow:0 16px 42px rgba(69,64,120,.08);
    }
    .upload-zone {
        border:1.5px dashed #cbc5e8; border-radius:18px; min-height:280px;
        display:grid; place-items:center; text-align:center; background:#fafaff;
    }
    .upload-icon {
        width:58px; height:58px; margin:0 auto 12px; border-radius:18px;
        background:#eeeafd; display:grid; place-items:center; color:#7055cb; font-size:26px;
    }
    .result-card {
        height:100%; min-height:280px; border-radius:18px;
        background:linear-gradient(145deg,#f7f5ff,#eef4ff);
        border:1px solid #e7e3f7; padding:24px;
    }
    .result-card h3 { color:#77778a; font-size:14px; font-weight:600; margin:0 0 16px; }
    .prediction-char {
        font-family:'Manrope'; font-size:84px; line-height:1; color:#7358c9;
        text-align:center; margin:12px 0 8px;
    }
    .confidence { text-align:center; color:#77778a; font-size:14px; margin-bottom:18px; }
    .top-row { display:flex; justify-content:space-between; align-items:center; margin:8px 0; color:#55556a; font-size:14px; }
    .top-row strong { color:#23233a; }
    .bar { height:6px; background:#e5e4ef; border-radius:99px; margin-top:-3px; overflow:hidden; }
    .bar-fill { height:100%; background:linear-gradient(90deg,#7557d1,#5680d6); border-radius:99px; }
    .steps { display:grid; grid-template-columns:repeat(4,1fr); gap:16px; }
    .step-card {
        background:white; border:1px solid #ededf4; border-radius:18px;
        padding:22px; box-shadow:0 10px 28px rgba(69,64,120,.05);
        transition:.25s ease;
    }
    .step-card:hover { transform:translateY(-5px); box-shadow:0 16px 34px rgba(69,64,120,.10); }
    .step-number {
        width:38px; height:38px; border-radius:12px; background:#f0edfc;
        display:grid; place-items:center; color:#7055c8; font-weight:800;
    }
    .step-card h3 { font-size:16px; color:#25253a; margin:16px 0 8px; }
    .step-card p { color:#7b7b8c; font-size:14px; line-height:1.55; margin:0; }
    .about-grid { display:grid; grid-template-columns:1fr 1fr; gap:20px; }
    .about-card { background:white; border:1px solid #ededf4; border-radius:20px; padding:28px; }
    .about-card h3 { font-family:'Manrope'; color:#242439; margin-top:0; }
    .about-card p { color:#77778a; line-height:1.7; }
    .model-alert {
        background:#fff8e8; border:1px solid #f3dea4; color:#72591d;
        border-radius:14px; padding:14px 16px; margin-bottom:16px; font-size:14px;
    }
    [data-testid="stFileUploader"] section {
        border:1.5px dashed #cbc5e8; border-radius:18px; background:#fafaff;
        min-height:230px;
    }
    [data-testid="stFileUploader"] button {
        border-radius:10px; border:1px solid #dad7ec; color:#5945ad;
    }
    div.stButton > button {
        width:100%; border-radius:12px; border:none; min-height:48px;
        background:linear-gradient(135deg,#7055c9,#507ed6); color:white;
        font-weight:700; box-shadow:0 10px 24px rgba(105,86,199,.20);
    }
    div.stButton > button:hover { filter:brightness(1.04); color:white; border:none; }
    @media (max-width: 800px) {
        .block-container { padding:1rem; }
        .nav-links { display:none; }
        .hero { grid-template-columns:1fr; padding:42px 0 36px; }
        .hero h1 { font-size:44px; letter-spacing:-2px; }
        .visual-card { min-height:330px; }
        .ai-window { inset:54px 24px 38px; }
        .steps, .about-grid { grid-template-columns:1fr; }
        .section-title { font-size:32px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="nav-wrap">
      <div class="brand"><div class="brand-mark">S</div>ScriptAI</div>
      <div class="nav-links">
        <span>Home</span><span>Recognize</span><span>How It Works</span>
        <span>About</span><a class="nav-cta" href="#recognize">Get Started</a>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
      <div>
        <div class="eyebrow">✦ AI-POWERED CHARACTER INTELLIGENCE</div>
        <h1>Turn Handwriting Into <span>Intelligence.</span></h1>
        <p class="hero-copy">Transform handwritten characters into digital predictions with a deep learning CNN trained on the EMNIST balanced dataset.</p>
        <div class="hero-proof">
          <span><b class="proof-dot">●</b> CNN-powered</span>
          <span><b class="proof-dot">●</b> 47 character classes</span>
        </div>
      </div>
      <div class="visual-card">
        <div class="ai-window">
          <div class="hand-char">A</div>
          <div class="scan"></div>
        </div>
        <div class="metric-chip">Model confidence<strong>Ready to analyze</strong></div>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div id="recognize"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <section class="section">
      <div class="section-head">
        <div class="section-kicker">Character Recognition</div>
        <h2 class="section-title">Recognize a handwritten character</h2>
        <p class="section-copy">Upload a clear PNG or JPG image. ScriptAI preprocesses it and sends it to the CNN.</p>
      </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload PNG or JPG",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed",
)

model_ready = DEFAULT_MODEL_PATH.exists()
if not model_ready:
    st.markdown(
        '<div class="model-alert"><b>Model not ready.</b> Train a CNN before recognition. '
        'Run <code>.venv/bin/python src/train_model.py</code> in the project folder.</div>',
        unsafe_allow_html=True,
    )

left, right = st.columns([1.05, 0.95], gap="large")
with left:
    if uploaded_file:
        try:
            uploaded_image = Image.open(uploaded_file)
            st.image(uploaded_image, caption="Uploaded image", use_container_width=True)
        except (OSError, Image.DecompressionBombError) as error:
            st.error(f"Could not open this image: {error}")

with right:
    st.markdown('<div class="result-card"><h3>Prediction</h3>', unsafe_allow_html=True)
    st.markdown('<div class="prediction-char">—</div>', unsafe_allow_html=True)
    st.markdown('<div class="confidence">Upload an image to begin</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

recognize = st.button(
    "Recognize Character",
    disabled=not uploaded_file or not model_ready,
    use_container_width=True,
)

if recognize and uploaded_file:
    try:
        uploaded_image = Image.open(uploaded_file)
        input_image, preview_image = preprocess_uploaded_image(uploaded_image)
        with st.spinner("Analyzing handwriting with the CNN..."):
            model = load_model()
            class_names = load_class_names()
            predictions = predict_character(model, input_image, class_names)

        st.subheader("Prediction results")
        result_left, result_right = st.columns([0.55, 1.45], gap="large")
        with result_left:
            st.image(preview_image, caption="CNN input (28 × 28)", width=220)
        with result_right:
            top_character, top_confidence = predictions[0]
            st.markdown(
                f"""
                <div class="result-card">
                  <h3>Predicted character</h3>
                  <div class="prediction-char">{top_character}</div>
                  <div class="confidence">{top_confidence:.2f}% confidence</div>
                """,
                unsafe_allow_html=True,
            )
            for character, confidence in predictions:
                st.markdown(
                    f'<div class="top-row"><strong>{character}</strong><span>{confidence:.2f}%</span></div>'
                    f'<div class="bar"><div class="bar-fill" style="width:{confidence:.2f}%"></div></div>',
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)
    except (OSError, ValueError, RuntimeError, FileNotFoundError) as error:
        st.error(str(error))

st.markdown("</section>", unsafe_allow_html=True)

st.markdown(
    """
    <section class="section">
      <div class="section-head">
        <div class="section-kicker">How It Works</div>
        <h2 class="section-title">From handwriting to prediction</h2>
      </div>
      <div class="steps">
        <div class="step-card"><div class="step-number">01</div><h3>Upload</h3><p>Select a clear PNG or JPG handwritten character.</p></div>
        <div class="step-card"><div class="step-number">02</div><h3>Preprocess</h3><p>The image is normalized, resized, centered, and converted to grayscale.</p></div>
        <div class="step-card"><div class="step-number">03</div><h3>CNN analysis</h3><p>The trained network studies edges and character patterns.</p></div>
        <div class="step-card"><div class="step-number">04</div><h3>Prediction</h3><p>The top character and confidence scores are returned.</p></div>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="section">
      <div class="section-head">
        <div class="section-kicker">About the Project</div>
        <h2 class="section-title">Built around a practical AI purpose</h2>
      </div>
      <div class="about-grid">
        <div class="about-card">
          <h3>EMNIST Balanced</h3>
          <p>EMNIST provides handwritten letters and digits designed for machine-learning benchmarks. This project uses the balanced split and its 47 character classes.</p>
        </div>
        <div class="about-card">
          <h3>Convolutional Neural Networks</h3>
          <p>A CNN learns useful visual patterns from images, from simple strokes to complete character shapes. It is well suited to recognizing handwriting accurately.</p>
        </div>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)
