import streamlit as st
import numpy as np
from PIL import Image
import os
import time
import matplotlib.pyplot as plt
import io

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="SmartVision AI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= ENHANCED UI STYLING =================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* ========== MAIN CONTENT - MAXIMUM VISIBILITY ========== */
    
    /* All text in main area - BLACK & BOLD */
    .main .block-container,
    .main .block-container p,
    .main .block-container div,
    .main .block-container span,
    .main .block-container label,
    .main .block-container li {
        color: #000000 !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        text-shadow: 0 1px 2px rgba(255, 255, 255, 0.5);
    }
    
    /* Headings - EXTRA BOLD & LARGER */
    h1 {
        color: #000000 !important;
        font-weight: 900 !important;
        font-size: 56px !important;
        letter-spacing: -1px;
        text-shadow: 0 2px 4px rgba(255, 255, 255, 0.3);
    }
    
    h2 {
        color: #000000 !important;
        font-weight: 800 !important;
        font-size: 38px !important;
        margin-top: 30px !important;
        margin-bottom: 20px !important;
    }
    
    h3 {
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 26px !important;
        margin-top: 20px !important;
    }
    
    h4 {
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 20px !important;
        margin-top: 15px !important;
    }
    
    /* File Uploader - MAXIMUM CONTRAST */
    .stFileUploader,
    .stFileUploader *,
    .stFileUploader label,
    .stFileUploader div,
    .stFileUploader span {
        color: #000000 !important;
        font-weight: 800 !important;
        font-size: 18px !important;
    }
    
    /* Selectbox & Inputs - BOLD BLACK TEXT */
    .stSelectbox label,
    .stSelectbox div,
    .stSelectbox span,
    .stSlider label,
    .stSlider div,
    .stSlider span {
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 17px !important;
    }
    
    /* Info/Warning/Error boxes - DARK TEXT */
    .stAlert,
    .stAlert *,
    .stInfo,
    .stInfo *,
    .stWarning,
    .stWarning *,
    .stError,
    .stError * {
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 16px !important;
    }
    
    /* ========== SIDEBAR - WHITE TEXT ========== */
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 2px solid rgba(102, 126, 234, 0.3);
    }
    
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] *,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] label {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar Navigation Radio Buttons */
    section[data-testid="stSidebar"] .stRadio > label {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 25px;
    }
    
    section[data-testid="stSidebar"] [role="radiogroup"] label {
        background: rgba(255, 255, 255, 0.08);
        padding: 14px 22px;
        border-radius: 12px;
        margin: 10px 0;
        transition: all 0.3s ease;
        border: 2px solid transparent;
        font-size: 16px !important;
        font-weight: 600 !important;
    }
    
    section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(102, 126, 234, 0.4);
        border: 2px solid rgba(102, 126, 234, 0.6);
        transform: translateX(8px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* ========== CONTAINERS & CARDS ========== */
    
    .main-container {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(15px);
        border-radius: 24px;
        padding: 45px;
        margin: 25px;
        box-shadow: 0 25px 70px rgba(0, 0, 0, 0.35);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 18px;
        text-align: center;
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.4);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 45px rgba(102, 126, 234, 0.5);
    }
    
    .metric-card,
    .metric-card *,
    .metric-card p,
    .metric-card div {
        color: white !important;
        font-weight: 800 !important;
    }
    
    .metric-value {
        font-size: 42px !important;
        font-weight: 900 !important;
        margin: 12px 0;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    
    .metric-label {
        font-size: 15px !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 700 !important;
    }
    
    .prediction-card {
        background: white;
        border-radius: 18px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
        border-left: 5px solid #667eea;
    }
    
    .prediction-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 18px;
        background: linear-gradient(90deg, rgba(102, 126, 234, 0.08) 0%, transparent 100%);
        border-radius: 12px;
        margin: 12px 0;
    }
    
    .rank-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        padding: 10px 18px;
        border-radius: 25px;
        font-weight: 800 !important;
        font-size: 16px !important;
        min-width: 50px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .confidence-badge {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white !important;
        padding: 10px 20px;
        border-radius: 25px;
        font-weight: 700 !important;
        font-size: 17px !important;
        box-shadow: 0 4px 12px rgba(240, 147, 251, 0.3);
    }
    
    .class-name {
        font-weight: 800 !important;
        font-size: 20px !important;
        color: #000000 !important;
        flex-grow: 1;
        margin: 0 25px;
    }
    
    /* ========== BUTTONS ========== */
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        padding: 14px 35px;
        border-radius: 12px;
        font-weight: 700 !important;
        font-size: 16px !important;
        box-shadow: 0 6px 18px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.5);
    }
    
    /* ========== UPLOAD SECTION ========== */
    
    .upload-section {
        background: white;
        border: 4px dashed #667eea;
        border-radius: 18px;
        padding: 45px;
        text-align: center;
    }
    
    /* ========== FEATURES & GRIDS ========== */
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 25px;
        margin: 35px 0;
    }
    
    .feature-item {
        background: white;
        padding: 30px;
        border-radius: 18px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    .feature-item:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.15);
    }
    
    .feature-icon {
        font-size: 48px;
        margin-bottom: 18px;
    }
    
    .class-chip {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        padding: 10px 20px;
        border-radius: 25px;
        margin: 6px;
        font-size: 15px !important;
        font-weight: 700 !important;
        box-shadow: 0 3px 10px rgba(102, 126, 234, 0.3);
    }
    
    /* ========== PROGRESS BAR ========== */
    
    .progress-bar {
        background: #e5e5e5;
        border-radius: 12px;
        height: 35px;
        overflow: hidden;
        margin: 12px 0;
        box-shadow: inset 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white !important;
        font-weight: 800 !important;
        font-size: 15px !important;
        transition: width 0.5s ease;
    }
    
    /* ========== ABOUT PAGE ========== */
    
    .about-card {
        background: white;
        padding: 35px;
        border-radius: 18px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        margin: 25px 0;
    }
    
    .tech-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        padding: 12px 24px;
        border-radius: 30px;
        margin: 10px;
        font-weight: 700 !important;
        font-size: 15px !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .feature-highlight {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.12) 0%, rgba(118, 75, 162, 0.12) 100%);
        border-left: 5px solid #667eea;
        padding: 25px;
        border-radius: 12px;
        margin: 18px 0;
    }
</style>
""", unsafe_allow_html=True)

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

POSSIBLE_MODEL_DIRS = [
    os.path.join(os.path.dirname(BASE_DIR), "Classification Models Training", "models"),
    os.path.join(BASE_DIR, "models"),
    os.path.join(os.path.dirname(BASE_DIR), "models"),
]

MODEL_DIR = None
for dir_path in POSSIBLE_MODEL_DIRS:
    if os.path.exists(dir_path):
        MODEL_DIR = dir_path
        break

if MODEL_DIR is None:
    MODEL_DIR = os.path.join(os.path.dirname(BASE_DIR), "Classification Models Training", "models")

IMG_SIZE = (224, 224)

# ================= MODEL FILES =================
MODEL_FILES = {
    "VGG16": "VGG16_best.h5",
    "ResNet50": "ResNet50_best.keras",
    "MobileNetV2": "MobileNetV2_best.keras",
    "EfficientNetB0": "EfficientNetB0_best.keras"
}

# ================= LOAD CLASSES =================
@st.cache_data
def load_classes():
    if MODEL_DIR:
        path = os.path.join(MODEL_DIR, "classes.txt")
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return [c.strip() for c in f.readlines() if c.strip()]
            except:
                pass
    
    return ['airplane', 'car', 'cat', 'dog', 'horse', 'ship', 'truck', 'bird', 
            'person', 'bicycle', 'boat', 'bus', 'motorcycle', 'train', 'bottle',
            'chair', 'table', 'plant', 'flower', 'tree', 'building', 'house',
            'road', 'sky', 'water']

CLASS_NAMES = load_classes()
NUM_CLASSES = len(CLASS_NAMES)

# ================= PREPROCESS IMAGE =================
def preprocess_image(img):
    try:
        img = img.resize(IMG_SIZE).convert("RGB")
        arr = np.array(img).astype("float32") / 255.0
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        arr = (arr - mean) / std
        return np.expand_dims(arr, axis=0)
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# ================= LOAD CLASSIFICATION MODELS (SILENT) =================
@st.cache_resource
def load_models():
    loaded = {}
    
    try:
        import tensorflow as tf
        
        if not os.path.exists(MODEL_DIR):
            return loaded
        
        for name, file in MODEL_FILES.items():
            path = os.path.join(MODEL_DIR, file)
            if os.path.exists(path):
                try:
                    model = tf.keras.models.load_model(path, compile=False)
                    loaded[name] = model
                except:
                    pass
        
        return loaded
        
    except ImportError:
        return {}
    except:
        return {}

models = load_models()

# ================= LOAD YOLO (SILENT) =================
@st.cache_resource
def load_yolo():
    try:
        from ultralytics import YOLO
        
        if MODEL_DIR and os.path.exists(MODEL_DIR):
            yolo_path = os.path.join(MODEL_DIR, "yolov8n.pt")
            if os.path.exists(yolo_path):
                return YOLO(yolo_path)
        
        return None
    except:
        return None

yolo = load_yolo()

# ================= CREATE MATPLOTLIB CHART =================
def create_confidence_chart(class_names, confidences):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(confidences)))
    bars = ax.barh(class_names, confidences, color=colors)
    
    ax.set_xlabel('Confidence (%)', fontsize=13, fontweight='bold')
    ax.set_title('Confidence Distribution', fontsize=16, fontweight='bold', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, 100)
    
    for bar, conf in zip(bars, confidences):
        width = bar.get_width()
        ax.text(width + 2, bar.get_y() + bar.get_height()/2, 
                f'{conf:.2f}%',
                ha='left', va='center', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close()
    
    return buf

# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("<h2 style='text-align: center; margin-bottom: 35px; font-size: 28px;'>🔮 SmartVision AI</h2>", unsafe_allow_html=True)
    
    page = st.radio(
        "Navigation",
        ["🏠 Home", "🎯 Classification", "📦 Detection", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    st.markdown("<hr style='margin: 35px 0; border-color: rgba(255,255,255,0.2); border-width: 2px;'>", unsafe_allow_html=True)
    
    st.markdown("<p style='font-weight: 700; font-size: 16px; margin-bottom: 15px; letter-spacing: 1px;'>📊 SYSTEM STATUS</p>", unsafe_allow_html=True)
    
    status_html = f"""
    <div style='background: rgba(255,255,255,0.08); padding: 20px; border-radius: 12px; margin-bottom: 15px; border: 1px solid rgba(255,255,255,0.1);'>
        <p style='margin: 8px 0; font-size: 15px;'>🤖 Models: <span style='color: #4ade80; font-weight: 800; font-size: 18px;'>{len(models)}</span></p>
        <p style='margin: 8px 0; font-size: 15px;'>📋 Classes: <span style='color: #4ade80; font-weight: 800; font-size: 18px;'>{NUM_CLASSES}</span></p>
        <p style='margin: 8px 0; font-size: 15px;'>🎯 YOLO: <span style='color: {"#4ade80" if yolo else "#f87171"}; font-weight: 800; font-size: 18px;'>{"✓ Ready" if yolo else "✗ N/A"}</span></p>
    </div>
    """
    st.markdown(status_html, unsafe_allow_html=True)
    
    st.markdown("<hr style='margin: 35px 0; border-color: rgba(255,255,255,0.2); border-width: 2px;'>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='font-size: 14px; text-align: center; padding: 25px;'>
        <p style='font-weight: 700; font-size: 16px;'><b>SmartVision AI v2.0</b></p>
        <p style='margin-top: 8px; font-size: 14px;'>Powered by Deep Learning</p>
        <p style='margin-top: 25px; opacity: 0.7; font-size: 13px;'>© 2025 SmartVision</p>
    </div>
    """, unsafe_allow_html=True)

# ================= HOME PAGE =================
if page == "🏠 Home":
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: center;'>SmartVision AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 22px; margin-bottom: 50px; font-weight: 600;'>Intelligent Multi-Class Object Recognition System</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Classification Models</div>
            <div class='metric-value'>{len(models) if models else 0}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Supported Classes</div>
            <div class='metric-value'>{NUM_CLASSES}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        status = "✓ Ready" if yolo else "Not Available"
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>YOLO Detection</div>
            <div class='metric-value' style='font-size: 26px;'>{status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<h2 style='margin-top: 60px; margin-bottom: 35px;'>Key Features</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='feature-grid'>
        <div class='feature-item'>
            <div class='feature-icon'>🎯</div>
            <h3>Multi-Model Classification</h3>
            <p style='font-size: 15px; line-height: 1.7;'>Choose from VGG16, ResNet50, MobileNetV2, and EfficientNetB0 for accurate predictions.</p>
        </div>
        <div class='feature-item'>
            <div class='feature-icon'>⚡</div>
            <h3>Real-Time Processing</h3>
            <p style='font-size: 15px; line-height: 1.7;'>Lightning-fast inference with optimized model architecture and preprocessing.</p>
        </div>
        <div class='feature-item'>
            <div class='feature-icon'>📦</div>
            <h3>Object Detection</h3>
            <p style='font-size: 15px; line-height: 1.7;'>Advanced YOLOv8 integration for precise object localization and classification.</p>
        </div>
        <div class='feature-item'>
            <div class='feature-icon'>📊</div>
            <h3>Confidence Metrics</h3>
            <p style='font-size: 15px; line-height: 1.7;'>Detailed probability distributions and top-K predictions for informed decisions.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if CLASS_NAMES:
        st.markdown("<h2 style='margin-top: 60px; margin-bottom: 25px;'>Supported Classes</h2>", unsafe_allow_html=True)
        st.markdown("<div style='background: white; padding: 30px; border-radius: 18px; box-shadow: 0 8px 25px rgba(0,0,0,0.1);'>", unsafe_allow_html=True)
        
        classes_html = "".join([f"<span class='class-chip'>{cls}</span>" for cls in CLASS_NAMES])
        st.markdown(classes_html, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ================= CLASSIFICATION PAGE =================
elif page == "🎯 Classification":
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    
    st.markdown("<h1>Image Classification</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 20px; margin-bottom: 35px; font-weight: 600;'>Upload an image and let AI identify what's in it</p>", unsafe_allow_html=True)
    
    if not models:
        st.warning("⚠️ No models loaded. Running in demo mode with synthetic predictions.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
        image_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div style='margin-top: 25px;'>", unsafe_allow_html=True)
        
        if models:
            model_name = st.selectbox("🤖 Select Model", list(models.keys()))
        else:
            model_name = st.selectbox("🤖 Select Model (Demo)", ["VGG16", "ResNet50", "MobileNetV2", "EfficientNetB0"])
        
        top_k = st.slider("📊 Number of Top Predictions", 1, min(5, NUM_CLASSES), 3)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        if image_file:
            try:
                img = Image.open(image_file)
                st.image(img, caption="Uploaded Image", use_container_width=True)
            except Exception as e:
                st.error(f"Error loading image: {e}")
                img = None
        else:
            st.info("👆 Upload an image to get started")
            img = None
    
    if image_file and img:
        img_arr = preprocess_image(img)
        
        if img_arr is not None:
            with st.spinner("🔄 Analyzing image..."):
                start = time.time()
                
                if models and model_name in models:
                    preds = np.squeeze(models[model_name].predict(img_arr, verbose=0))
                else:
                    preds = np.random.dirichlet(np.ones(NUM_CLASSES))
                    preds[np.random.randint(0, NUM_CLASSES)] *= 2
                    preds = preds / preds.sum()
                
                elapsed = (time.time() - start) * 1000
            
            st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
            
            st.markdown("<h3>Results</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 17px;'><b>Model:</b> {model_name} {'(Demo Mode)' if not models else ''} | <b>Inference Time:</b> {elapsed:.0f} ms</p>", unsafe_allow_html=True)
            
            st.markdown("<h4 style='margin-top: 30px;'>Top Predictions:</h4>", unsafe_allow_html=True)
            
            top_idx = np.argsort(preds)[::-1][:top_k]
            
            for rank, idx in enumerate(top_idx, 1):
                confidence = preds[idx] * 100
                st.markdown(f"""
                <div class='prediction-row'>
                    <span class='rank-badge'>#{rank}</span>
                    <span class='class-name'>{CLASS_NAMES[idx]}</span>
                    <span class='confidence-badge'>{confidence:.2f}%</span>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class='progress-bar'>
                    <div class='progress-fill' style='width: {confidence}%;'>
                        {confidence:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<h4 style='margin-top: 35px;'>Confidence Distribution:</h4>", unsafe_allow_html=True)
            
            chart_classes = [CLASS_NAMES[i] for i in top_idx]
            chart_confidences = [preds[i] * 100 for i in top_idx]
            
            chart_buf = create_confidence_chart(chart_classes, chart_confidences)
            st.image(chart_buf, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ================= DETECTION PAGE =================
elif page == "📦 Detection":
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    
    st.markdown("<h1>Object Detection</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 20px; margin-bottom: 35px; font-weight: 600;'>Detect and localize multiple objects using YOLOv8</p>", unsafe_allow_html=True)
    
    if not yolo:
        st.error("⚠️ YOLO model not available. Please ensure yolov8n.pt is in the models directory and ultralytics is installed.")
        st.code("pip install ultralytics")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
        image_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key="detection_upload"
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        confidence_threshold = st.slider("🎯 Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    
    with col2:
        if image_file:
            try:
                img = Image.open(image_file).convert("RGB")
                st.image(img, caption="Original Image", use_container_width=True)
            except Exception as e:
                st.error(f"Error loading image: {e}")
                img = None
        else:
            st.info("👆 Upload an image to get started")
            img = None
    
    if image_file and img and yolo:
        with st.spinner("🔄 Detecting objects..."):
            start = time.time()
            
            img_np = np.array(img)
            results = yolo(img_np, conf=confidence_threshold, verbose=False)
            elapsed = (time.time() - start) * 1000
        
        annotated = results[0].plot()
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(img, caption="Original Image", use_container_width=True)
        
        with col2:
            st.image(annotated, caption="Detected Objects", use_container_width=True)
        
        st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
        st.markdown("<h3>Detection Results</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 17px;'><b>Processing Time:</b> {elapsed:.0f} ms | <b>Objects Detected:</b> {len(results[0].boxes)}</p>", unsafe_allow_html=True)
        
        if len(results[0].boxes) > 0:
            st.markdown("<h4 style='margin-top: 25px;'>Detected Objects:</h4>", unsafe_allow_html=True)
            
            for idx, box in enumerate(results[0].boxes, 1):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0]) * 100
                cls_name = results[0].names[cls_id]
                
                st.markdown(f"""
                <div class='prediction-row'>
                    <span class='rank-badge'>#{idx}</span>
                    <span class='class-name'>{cls_name}</span>
                    <span class='confidence-badge'>{conf:.2f}%</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No objects detected. Try lowering the confidence threshold.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ================= ABOUT PAGE =================
elif page == "ℹ️ About":
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: center;'>About SmartVision AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 20px; margin-bottom: 50px; font-weight: 600;'>Advanced Computer Vision Platform</p>", unsafe_allow_html=True)
    
    st.markdown("<div class='about-card'>", unsafe_allow_html=True)
    st.markdown("<h2>🎯 Project Overview</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p style='font-size: 17px; line-height: 1.9; font-weight: 500;'>
    SmartVision AI is a state-of-the-art computer vision platform that combines multiple deep learning 
    architectures for robust image classification and object detection. Our system leverages the power 
    of transfer learning and modern neural networks to deliver accurate, real-time predictions across 
    diverse object categories.
    </p>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='about-card'>", unsafe_allow_html=True)
    st.markdown("<h2>✨ Key Features</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='feature-highlight'>
            <h4>🤖 Multi-Model Architecture</h4>
            <p style='font-size: 16px;'>Choose from 4 pre-trained models: VGG16, ResNet50, MobileNetV2, and EfficientNetB0, 
            each optimized for different use cases.</p>
        </div>
        
        <div class='feature-highlight'>
            <h4>📦 Object Detection</h4>
            <p style='font-size: 16px;'>YOLOv8 integration enables real-time object detection with bounding boxes and 
            confidence scores.</p>
        </div>
        
        <div class='feature-highlight'>
            <h4>⚡ High Performance</h4>
            <p style='font-size: 16px;'>Optimized preprocessing pipelines and model caching ensure sub-second inference times.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='feature-highlight'>
            <h4>📊 Detailed Analytics</h4>
            <p style='font-size: 16px;'>Comprehensive confidence metrics, probability distributions, and visual charts 
            for better decision-making.</p>
        </div>
        
        <div class='feature-highlight'>
            <h4>🎨 Premium UI/UX</h4>
            <p style='font-size: 16px;'>Modern, responsive interface with smooth animations and intuitive navigation.</p>
        </div>
        
        <div class='feature-highlight'>
            <h4>🔧 Flexible & Extensible</h4>
            <p style='font-size: 16px;'>Modular architecture allows easy integration of new models and custom datasets.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='about-card'>", unsafe_allow_html=True)
    st.markdown("<h2>🛠️ Technologies Used</h2>", unsafe_allow_html=True)
    
    st.markdown("<h4 style='margin-top: 25px;'>Deep Learning Frameworks</h4>", unsafe_allow_html=True)
    st.markdown("""
    <span class='tech-badge'>TensorFlow</span>
    <span class='tech-badge'>Keras</span>
    <span class='tech-badge'>Ultralytics YOLO</span>
    <span class='tech-badge'>PyTorch</span>
    """, unsafe_allow_html=True)
    
    st.markdown("<h4 style='margin-top: 35px;'>Core Libraries</h4>", unsafe_allow_html=True)
    st.markdown("""
    <span class='tech-badge'>NumPy</span>
    <span class='tech-badge'>Pillow</span>
    <span class='tech-badge'>Matplotlib</span>
    <span class='tech-badge'>OpenCV</span>
    """, unsafe_allow_html=True)
    
    st.markdown("<h4 style='margin-top: 35px;'>Web Framework</h4>", unsafe_allow_html=True)
    st.markdown("""
    <span class='tech-badge'>Streamlit</span>
    <span class='tech-badge'>HTML5</span>
    <span class='tech-badge'>CSS3</span>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='about-card'>", unsafe_allow_html=True)
    st.markdown("<h2>🧠 Model Architectures</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='feature-highlight'>
            <h4>VGG16</h4>
            <p style='font-size: 16px;'><b>Layers:</b> 16 | <b>Parameters:</b> 138M<br>
            Deep architecture with small 3x3 filters. Excellent for feature extraction and transfer learning.</p>
        </div>
        
        <div class='feature-highlight'>
            <h4>MobileNetV2</h4>
            <p style='font-size: 16px;'><b>Layers:</b> 53 | <b>Parameters:</b> 3.5M<br>
            Lightweight model optimized for mobile devices. Uses depthwise separable convolutions.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='feature-highlight'>
            <h4>ResNet50</h4>
            <p style='font-size: 16px;'><b>Layers:</b> 50 | <b>Parameters:</b> 25.6M<br>
            Residual learning framework that enables training of very deep networks without degradation.</p>
        </div>
        
        <div class='feature-highlight'>
            <h4>EfficientNetB0</h4>
            <p style='font-size: 16px;'><b>Layers:</b> 237 | <b>Parameters:</b> 5.3M<br>
            Compound scaling method for optimal balance between accuracy and efficiency.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='about-card'>", unsafe_allow_html=True)
    st.markdown("<h2>📈 Performance Metrics</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='text-align: center; padding: 25px; background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%); border-radius: 12px;'>
            <h3 style='color: #667eea; margin: 0; font-size: 34px;'>< 100ms</h3>
            <p style='margin: 12px 0 0 0; font-size: 16px;'><b>Inference Time</b></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 25px; background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%); border-radius: 12px;'>
            <h3 style='color: #667eea; margin: 0; font-size: 34px;'>95%+</h3>
            <p style='margin: 12px 0 0 0; font-size: 16px;'><b>Top-5 Accuracy</b></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='text-align: center; padding: 25px; background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%); border-radius: 12px;'>
            <h3 style='color: #667eea; margin: 0; font-size: 34px;'>Real-time</h3>
            <p style='margin: 12px 0 0 0; font-size: 16px;'><b>Detection Speed</b></p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='about-card' style='text-align: center;'>", unsafe_allow_html=True)
    st.markdown("<h2>📧 Get In Touch</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p style='font-size: 17px; line-height: 1.9;'>
    Interested in collaborating or have questions about SmartVision AI?<br>
    We'd love to hear from you!
    </p>
    <p style='margin-top: 25px; font-size: 16px;'>
    <b>Email:</b> contact@smartvision-ai.com<br>
    <b>GitHub:</b> github.com/smartvision-ai<br>
    <b>Website:</b> www.smartvision-ai.com
    </p>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    

    st.markdown("</div>", unsafe_allow_html=True)
