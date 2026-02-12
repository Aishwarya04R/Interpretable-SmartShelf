import streamlit as st
import tensorflow as tf
import torch
import torch.nn as nn
import timm
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
import cv2
import os
import sys
import base64
from datetime import datetime, timedelta
from tensorflow.keras import layers, models

# --- SETUP PATHS ---
sys.path.append('/content/drive/MyDrive/Shelf_Life_Project/source')
MODEL_DIR = '/content/drive/MyDrive/Shelf_Life_Project/final_src'

try:
    from xai_utils import ExplainableAI
    import novelty_features as nf
except ImportError as e:
    st.error(f"Setup Error: {e}. Ensure xai_utils.py and novelty_features.py are in 'source' folder.")
    st.stop()

# --- PAGE CONFIG ---
st.set_page_config(page_title="SmartShelf AI", page_icon="🧬", layout="wide", initial_sidebar_state="collapsed")

# ==========================================
# 1. DEFINE PYTORCH MODEL ARCHITECTURE (VEG)
# ==========================================
class MultiHeadEfficientNet(nn.Module):
    def __init__(self, num_veg_classes):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
        feat = self.backbone.num_features
        self.drop = nn.Dropout(0.4)
        self.veg_head = nn.Linear(feat, num_veg_classes)
        self.fresh_head = nn.Linear(feat, 3)
        self.reg_head = nn.Linear(feat, 1)

    def forward(self, x):
        f = self.drop(self.backbone(x))
        return self.veg_head(f), self.fresh_head(f), self.reg_head(f).squeeze(1)

# ==========================================
# 2. DEFINE KERAS MODEL ARCHITECTURE (BAKERY)
# ==========================================
def build_msff_regressor(input_shape=(224,224,3)):
    inp = layers.Input(shape=input_shape)
    b1 = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    b1 = layers.MaxPooling2D(2)(b1)
    b2 = layers.Conv2D(32, 5, padding="same", activation="relu")(inp)
    b2 = layers.MaxPooling2D(2)(b2)
    b3 = layers.Conv2D(32, 7, padding="same", activation="relu")(inp)
    b3 = layers.MaxPooling2D(2)(b3)
    x = layers.Concatenate()([b1, b2, b3])
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu", name="msff_last_conv")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu", name="feature_dense")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation="linear", name="days_remaining")(x)
    model = models.Model(inp, out, name="MSFF_Regression")
    return model

# --- CUSTOM CSS FOR DASHBOARD ---
st.markdown("""
<style>
    /* Global Font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;700&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }

    /* Header Styles */
    .dashboard-header {
        text-align: center;
        padding: 60px 20px;
        background: linear-gradient(120deg, #1E3A5F 0%, #2b5876 100%);
        border-radius: 25px;
        color: white;
        margin-bottom: 50px;
        box-shadow: 0 10px 30px rgba(30, 58, 95, 0.2);
    }
    .main-title { font-size: 4rem; font-weight: 800; margin: 0; color: white; letter-spacing: -1px; }
    .sub-title { font-size: 1.4rem; color: #aab8c2; margin-top: 10px; font-weight: 300; }

    /* Modern Image Cards */
    .category-card {
        background-color: white;
        border-radius: 20px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.08);
        text-align: center;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        border: 1px solid #f0f2f5;
        height: 100%;
        overflow: hidden;
        position: relative;
    }
    .category-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 20px 50px rgba(0,0,0,0.12);
        border-color: #667eea;
    }

    /* Image Styling */
    .card-img-container {
        height: 180px;
        overflow: hidden;
        position: relative;
    }
    .card-img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        transition: transform 0.5s ease;
    }
    .category-card:hover .card-img {
        transform: scale(1.1);
    }

    /* Text Styling */
    .card-content { padding: 20px; }
    .card-title { font-size: 1.5rem; font-weight: 700; color: #1E3A5F; margin-bottom: 5px; }
    .card-desc { font-size: 0.9rem; color: #8898aa; margin-bottom: 20px; line-height: 1.5; }

    /* Button Styling */
    div.stButton > button {
        width: 100%;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        font-weight: 600;
        border: none;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        transition: opacity 0.3s;
    }
    div.stButton > button:hover {
        opacity: 0.9;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }

    /* Analysis Page Specifics */
    .metric-container { background: white; padding: 25px; border-radius: 15px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); text-align: center; border: 1px solid #eee; }
    .xai-box { background-color: #f8f9fa; padding: 20px; border-radius: 12px; border-left: 5px solid #667eea; font-size: 1rem; color: #495057; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'

def navigate_to(page):
    st.session_state['page'] = page
    st.rerun()

# --- LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    resources = {}

    # 1. MEAT MODEL (TensorFlow)
    tf_path = os.path.join(MODEL_DIR, "final_smartshelf_model.keras")
    if os.path.exists(tf_path):
        resources['meat_model'] = tf.keras.models.load_model(tf_path)
        resources['meat_cat'] = np.load(os.path.join(MODEL_DIR, "classes_category.npy"), allow_pickle=True)
        resources['meat_stat'] = np.load(os.path.join(MODEL_DIR, "classes_status.npy"), allow_pickle=True)
        resources['meat_xai'] = ExplainableAI(resources['meat_model'], resources['meat_cat'], backend='tensorflow')

    # 2. VEG MODEL (PyTorch)
    pth_path = os.path.join(MODEL_DIR, "best_multitask_model3 (1).pth")
    if os.path.exists(pth_path):
        try:
            checkpoint = torch.load(pth_path, map_location=torch.device('cpu'))
            veg_map = checkpoint.get("veg_map")
            if not veg_map:
                veg_classes = ['Carrot', 'Cucumber', 'Tomato', 'Potato', 'Brinjal', 'Capsicum']
            else:
                veg_classes = [k for k, v in sorted(veg_map.items(), key=lambda item: item[1])]

            veg_model = MultiHeadEfficientNet(num_veg_classes=len(veg_classes))
            veg_model.load_state_dict(checkpoint["model"])
            veg_model.eval()

            resources['veg_model'] = veg_model
            resources['veg_classes'] = veg_classes
            resources['veg_fresh_classes'] = ['Fresh', 'Mid', 'Spoiled']
            resources['veg_xai'] = ExplainableAI(veg_model, veg_classes, backend='pytorch')
        except Exception as e:
            print(f"Failed to load PyTorch model: {e}")

    # 3. BAKERY MODEL (TensorFlow/Keras)
    bakery_path = os.path.join(MODEL_DIR, "msff_bread_model.keras")
    if os.path.exists(bakery_path):
        try:
            try:
                resources['bakery_model'] = tf.keras.models.load_model(bakery_path)
            except:
                model = build_msff_regressor()
                model.load_weights(bakery_path)
                resources['bakery_model'] = model

            resources['bakery_classes'] = ['Bread']
            resources['bakery_xai'] = ExplainableAI(resources['bakery_model'], ['Bread'], backend='tensorflow')
        except Exception as e:
             print(f"Failed to load Bakery model: {e}")

    # 4. FRUIT MODEL (TensorFlow/Keras)
    fruit_path = "/content/drive/MyDrive/Shelf_Life_Project/final_src/fruit_shelf_life_model.h5"
    if os.path.exists(fruit_path):
        try:
             # compile=False helps avoid issues with custom metrics/losses from older Keras versions
             resources['fruit_model'] = tf.keras.models.load_model(fruit_path, compile=False)
             resources['fruit_classes'] = ["Apple", "Banana", "Guava", "Orange", "Pomegranate", "Strawberry"]
             resources['fruit_xai'] = ExplainableAI(resources['fruit_model'], resources['fruit_classes'], backend='tensorflow')
             print("SUCCESS: Fruit model loaded!")
        except Exception as e:
             print(f"Failed to load Fruit model: {e}")
    else:
         print(f"CRITICAL ERROR: Fruit model file missing at {fruit_path}")

    return resources

res = load_resources()

# --- HELPER FUNCTIONS ---
def get_img_as_base64(file_path):
    """Converts local image to base64 string for HTML embedding"""
    if not os.path.exists(file_path):
        return ""
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def preprocess_image(image):
    img = image.resize((380, 380))
    img_array = np.array(img)
    if img_array.ndim == 2: img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4: img_array = img_array[..., :3]
    img_pre = tf.keras.applications.efficientnet.preprocess_input(img_array.astype(np.float32))
    return img_pre, img_array

def preprocess_torch(image):
    if image.mode != 'RGB': image = image.convert('RGB')

    # 1. Create a LARGE copy for Display/XAI (this fixes the blur)
    img_display = image.resize((380, 380))

    # 2. Create a SMALL copy for the Model (must match training size)
    img_model = image.resize((96, 96))

    # 3. Process only the small one for the AI
    arr = np.array(img_model).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std

    # 4. Return the PROCESSED small image for the model,
    #    but the HIGH-RES display image for XAI
    return arr, np.array(img_display)

def preprocess_bakery(image):
    if image.mode != 'RGB': image = image.convert('RGB')
    img = image.resize((224, 224))
    arr = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0), np.array(img)

def preprocess_fruit(image):
    # FIXED: Convert RGB to BGR for Fruit Model
    if image.mode != 'RGB': image = image.convert('RGB')
    img_arr = np.array(image)
    if img_arr.ndim == 2: img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
    if img_arr.shape[-1] == 4: img_arr = img_arr[..., :3]

    img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_bgr, (160, 160))
    arr = img_resized.astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0), img_arr

def get_grade(score, status):
    if status == 'spoiled' or 'Rotten' in status or status == 'spoiled': return "F", "#ef476f"
    if score > 0.8: return "A+", "#06d6a0"
    if score > 0.6: return "A", "#26de81"
    if score > 0.4: return "B", "#ffd166"
    return "C", "#ff9f43"

def get_bakery_freshness(days_remaining):
    if days_remaining >= 5: return "Fresh", "#06d6a0"
    elif days_remaining >= 3: return "Mild", "#ffd166"
    elif days_remaining >= 1: return "Okay", "#ff9f43"
    else: return "Spoiled", "#ef476f"

# --- PAGE: HOME DASHBOARD ---
def render_home():
    st.markdown("""
        <div class="dashboard-header">
            <h1 class="main-title">SmartShelf AI</h1>
            <p class="sub-title">The Next Generation of Food Quality Intelligence</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    path_meat = "/content/drive/MyDrive/Shelf_Life_Project/final_src/meat.png"
    path_fruit = "/content/drive/MyDrive/Shelf_Life_Project/final_src/fruits.png"
    path_veg = "/content/drive/MyDrive/Shelf_Life_Project/final_src/vegetables.png"
    path_bakery = "/content/drive/MyDrive/Shelf_Life_Project/final_src/bakery.png"

    b64_meat = get_img_as_base64(path_meat)
    b64_fruit = get_img_as_base64(path_fruit)
    b64_veg = get_img_as_base64(path_veg)
    b64_bakery = get_img_as_base64(path_bakery)

    placeholder = "https://via.placeholder.com/400x300?text=Image+Not+Found"

    src_meat = f"data:image/png;base64,{b64_meat}" if b64_meat else placeholder
    src_fruit = f"data:image/png;base64,{b64_fruit}" if b64_fruit else placeholder
    src_veg = f"data:image/png;base64,{b64_veg}" if b64_veg else placeholder
    src_bakery = f"data:image/png;base64,{b64_bakery}" if b64_bakery else placeholder

    with col1:
        st.markdown(f"""
            <div class="category-card">
                <div class="card-img-container"><img src="{src_meat}" class="card-img"></div>
                <div class="card-content">
                    <div class="card-title">Meat & Seafood</div>
                    <div class="card-desc">Multi-Task Protein Quality Analysis.</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Analysis", key="btn_meat"): navigate_to("meat")

    with col2:
        st.markdown(f"""
            <div class="category-card">
                <div class="card-img-container"><img src="{src_fruit}" class="card-img"></div>
                <div class="card-content">
                    <div class="card-title">Fruits</div>
                    <div class="card-desc">Ripeness and freshness detection.</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Analysis", key="btn_fruit"): navigate_to("fruit")

    with col3:
        st.markdown(f"""
            <div class="category-card">
                <div class="card-img-container"><img src="{src_veg}" class="card-img"></div>
                <div class="card-content">
                    <div class="card-title">Vegetables</div>
                    <div class="card-desc">Quality grading for greens and roots.</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Analysis", key="btn_veg"): navigate_to("veg")

    with col4:
        st.markdown(f"""
            <div class="category-card">
                <div class="card-img-container"><img src="{src_bakery}" class="card-img"></div>
                <div class="card-content">
                    <div class="card-title">Bakery</div>
                    <div class="card-desc">Mold detection for baked goods.</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Analysis", key="btn_bakery"): navigate_to("bakery")

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("System Status", "Online ✅")
    c2.metric("Models Active", "4") # Meat + Veg + Bakery + Fruit
    c3.metric("Last Update", datetime.now().strftime("%B %d, %Y"))

# --- PAGE: MEAT MODULE ---
def render_meat_page():
    # Header with Back Navigation
    c_back, c_title = st.columns([1, 8])
    with c_back:
        if st.button("⬅ Home"): navigate_to("home")
    with c_title:
        st.markdown("## 🥩 Meat & Seafood Analysis")

    if 'meat_model' not in res:
        st.error("Model not found in `final_src`. Please verify files.")
        return

    model = res['meat_model']
    cat_classes = res['meat_cat']
    stat_classes = res['meat_stat']
    xai = res['meat_xai']

    c_side, c_main = st.columns([1, 2])

    with c_side:
        st.markdown("### Input Source")
        mode = st.radio("Select:", ["Upload Image", "Live Camera"], label_visibility="collapsed")

        img_file = None
        if mode == "Upload Image":
            img_file = st.file_uploader("Upload", type=['jpg','png','jpeg'])
        else:
            img_file = st.camera_input("Capture")

        if img_file:
            image_pil = Image.open(img_file)
            st.image(image_pil, caption="Sample", use_container_width=True)
            analyze = st.button("🔬 Run Diagnostics", type="primary", use_container_width=True)
        else:
            analyze = False

    with c_main:
        if img_file and analyze:
            with st.spinner("🧠 AI is analyzing texture patterns..."):
                img_pre, img_orig = preprocess_image(image_pil)
                img_batch = np.expand_dims(img_pre, axis=0)
                preds = model.predict(img_batch, verbose=0)

                cat_idx = np.argmax(preds[0])
                food_name = cat_classes[cat_idx]
                conf = np.max(preds[0])
                stat_idx = np.argmax(preds[1])
                status = stat_classes[stat_idx]
                score = float(preds[2][0][0])

                max_life = 5.0
                if status == 'spoiled': days_left = 0.0; score = 0.1
                else: days_left = score * max_life

                expiry_str = (datetime.now() + timedelta(days=days_left)).strftime("%b %d")
                grade, color = get_grade(score, status)

                st.session_state['meat_res'] = {
                    'food': food_name, 'status': status, 'days': days_left,
                    'score': score, 'grade': grade, 'color': color, 'expiry': expiry_str,
                    'img_pre': img_pre, 'cat_idx': cat_idx, 'stat_idx': stat_idx,
                    'img_orig': img_orig # Save original for XAI visualization
                }
                if 'xai_data' in st.session_state: del st.session_state['xai_data']

        if 'meat_res' in st.session_state:
            result = st.session_state['meat_res']

            m1, m2, m3 = st.columns(3)
            m1.markdown(f"<div class='metric-container'><h4>Detected</h4><h2 style='color:#1E3A5F'>{result['food'].upper()}</h2></div>", unsafe_allow_html=True)
            m2.markdown(f"<div class='metric-container' style='border-bottom: 5px solid {result['color']}'><h4>Condition</h4><h2 style='color:{result['color']}'>{result['status'].upper()}</h2></div>", unsafe_allow_html=True)
            m3.markdown(f"<div class='metric-container'><h4>Shelf Life</h4><h2>{result['days']:.1f} Days</h2><small>Exp: {result['expiry']}</small></div>", unsafe_allow_html=True)

            st.markdown("---")

            t1, t2, t3 = st.tabs(["📉 Projections", "🧠 Explainable AI", "🌍 Sustainability"])

            with t1:
                col_a, col_b = st.columns([2,1])
                with col_a: st.plotly_chart(nf.QualityDecayPredictor.predict_decay_curve(result['score'], result['days']), use_container_width=True)
                with col_b:
                    st.write("**AI Storage Advice**")
                    for i, title, desc in nf.SmartStorageAdvisor.generate_recommendations(result['food'], result['score']*100, result['days']):
                        st.info(f"{i} **{title}**: {desc}")

            with t2:
                st.info("Visualizing decision logic with Grad-CAM and LIME.")

                if 'xai_data' not in st.session_state:
                    if st.button("Generate Deep Explanation (~10s)", use_container_width=True):
                        with st.spinner("Calculating Gradients & Superpixels..."):
                            hm_food, hm_fresh = xai.grad_cam(result['img_pre'])
                            focus = xai.analyze_heatmap_focus(hm_fresh)
                            lime = xai.lime_explanation(result['img_orig'], target_head=1, num_samples=50)
                            st.session_state['xai_data'] = {'hm': hm_fresh, 'focus': focus, 'lime': lime}
                            st.rerun()
                else:
                    xd = st.session_state['xai_data']
                    v_grad = xai.visualize_grad_cam(result['img_pre'], xd['hm'], alpha=0.4)
                    v_lime = xai.visualize_lime(xd['lime'], result['stat_idx'], num_features=5)

                    r1, r2 = st.columns(2)
                    with r1:
                        # FIX: Added clamp=True to prevent range error
                        st.image(v_grad, caption="Grad-CAM (Attention)", use_container_width=True, clamp=True)
                        grad_text = nf.XAITextGenerator.explain_gradcam(result['food'], result['status'], xd['focus'])
                        st.markdown(f"<div class='xai-box'>{grad_text}</div>", unsafe_allow_html=True)
                    with r2:
                        # FIX: Added clamp=True to prevent range error
                        st.image(v_lime, caption="LIME Features", use_container_width=True, clamp=True)
                        lime_text = nf.XAITextGenerator.explain_lime(result['food'], result['status'])
                        st.markdown(f"<div class='xai-box'>{lime_text}</div>", unsafe_allow_html=True)

                    if st.button("Reset Explanation"): del st.session_state['xai_data']; st.rerun()

            with t3:
                co2, miles = nf.CarbonFootprintCalculator.calculate_impact(result['food'], 0.5)
                st.metric("Estimated Footprint", f"{co2:.2f} kg CO2e", delta=f"{miles:.1f} car miles")
                st.markdown(f"<div class='xai-box' style='border-left-color: #28a745'>{nf.CarbonFootprintCalculator.generate_impact_text(result['food'], co2)}</div>", unsafe_allow_html=True)

# --- PAGE: VEGETABLE MODULE ---
def render_veg_page():
    if st.button("← Back to Dashboard"):
        navigate_to("home")

    st.markdown("## 🥦 Vegetable Quality Analysis")

    if 'veg_model' not in res:
        st.error("Veg model missing or failed to load. Ensure 'best_multitask_model3 (1).pth' is in `final_src`.")
        return

    model = res['veg_model']
    veg_classes = res['veg_classes']
    fresh_classes = res['veg_fresh_classes']
    xai = res['veg_xai']

    c_side, c_main = st.columns([1, 2])

    with c_side:
        st.markdown("### Input Source")
        mode = st.radio("Select:", ["Upload Image", "Live Camera"], label_visibility="collapsed", key="veg_mode")

        img_file = None
        if mode == "Upload Image":
            img_file = st.file_uploader("Upload", type=['jpg','png','jpeg'], key="veg_upload")
        else:
            img_file = st.camera_input("Capture", key="veg_cam")

        if img_file:
            image_pil = Image.open(img_file)
            st.image(image_pil, caption="Sample", use_container_width=True)
            analyze = st.button("🔬 Analyze Veg", type="primary", use_container_width=True)
        else:
            analyze = False

    with c_main:
        if img_file and analyze:
            with st.spinner("🥦 Analyzing Vegetable Freshness..."):
                # Use CORRECT Preprocessing for Veg (96x96)
                img_pre, img_orig = preprocess_torch(image_pil)
                tensor = torch.from_numpy(img_pre).permute(2, 0, 1).unsqueeze(0).float()

                with torch.no_grad():
                    veg_out, fresh_out, reg_out = model(tensor)
                    veg_probs = torch.softmax(veg_out, dim=1)
                    veg_idx = torch.argmax(veg_probs).item()
                    food = veg_classes[veg_idx]

                    fresh_probs = torch.softmax(fresh_out, dim=1)
                    fresh_idx = torch.argmax(fresh_probs).item()
                    status = fresh_classes[fresh_idx]

                    shelf_life = max(0.0, reg_out.item())

                # Logic Fix: Map "Fresh" to High Score
                score = min(1.0, shelf_life / 14.0)
                if status == 'Fresh' or status == 'Mid':
                    score = max(score, 0.45) # Ensure it passes 0.40 threshold for "Do Not Consume"
                if status == 'Rotten' or status == 'spoiled':
                    score = 0.1

                grade, color = get_grade(score, status)
                expiry_str = (datetime.now() + timedelta(days=float(shelf_life))).strftime("%b %d")

                st.session_state['veg_res'] = {
                    'food': food, 'status': status, 'days': shelf_life,
                    'score': score, 'grade': grade, 'color': color, 'expiry': expiry_str,
                    'img_pre': img_pre, 'stat_idx': fresh_idx, # Store index for XAI
                    'img_orig': img_orig # Store original for LIME
                }
                if 'veg_xai_data' in st.session_state: del st.session_state['veg_xai_data']

        if 'veg_res' in st.session_state:
            result = st.session_state['veg_res']

            m1, m2, m3 = st.columns(3)
            m1.markdown(f"<div class='metric-container'><h4>Detected</h4><h2 style='color:#1E3A5F'>{result['food']}</h2></div>", unsafe_allow_html=True)
            m2.markdown(f"<div class='metric-container' style='border-bottom: 5px solid {result['color']}'><h4>Condition</h4><h2 style='color:{result['color']}'>{result['status']}</h2></div>", unsafe_allow_html=True)
            m3.markdown(f"<div class='metric-container'><h4>Shelf Life</h4><h2>{result['days']:.1f} Days</h2><small>Exp: {result['expiry']}</small></div>", unsafe_allow_html=True)

            st.markdown("---")

            t1, t2, t3 = st.tabs(["📉 Projections", "🧠 Explainable AI", "🌍 Sustainability"])

            with t1:
                col_a, col_b = st.columns([2,1])
                with col_a: st.plotly_chart(nf.QualityDecayPredictor.predict_decay_curve(result['score'], result['days']), use_container_width=True)
                with col_b:
                    st.write("**AI Storage Advice**")
                    for i, title, desc in nf.SmartStorageAdvisor.generate_recommendations(result['food'], result['score']*100, result['days']):
                        st.info(f"{i} **{title}**: {desc}")

            with t2:
                st.info("Visualizing decision logic with Grad-CAM and LIME.")
                if 'veg_xai_data' not in st.session_state:
                    if st.button("Generate Explanation (~10s)"):
                        with st.spinner("Computing..."):
                            hm, _ = xai.grad_cam(result['img_pre'])
                            focus = xai.analyze_heatmap_focus(hm)
                            # Target Freshness Head (Index 1) for LIME
                            lime = xai.lime_explanation(result['img_orig'], target_head=1, num_samples=50) # Use img_orig
                            st.session_state['veg_xai_data'] = {'hm': hm, 'focus': focus, 'lime': lime}
                            st.rerun()
                else:
                    xd = st.session_state['veg_xai_data']
                    v_grad = xai.visualize_grad_cam(result['img_pre'], xd['hm'], alpha=0.4)
                    v_lime = xai.visualize_lime(xd['lime'], result['stat_idx'], num_features=5)
                    r1, r2 = st.columns(2)
                    with r1:
                        st.image(v_grad, caption="Grad-CAM Attention", use_container_width=True, clamp=True)
                        grad_text = nf.XAITextGenerator.explain_gradcam(result['food'], result['status'], xd['focus'])
                        st.markdown(f"<div class='xai-box'>{grad_text}</div>", unsafe_allow_html=True)
                    with r2:
                        st.image(v_lime, caption="LIME Features", use_container_width=True, clamp=True)
                        lime_text = nf.XAITextGenerator.explain_lime(result['food'], result['status'])
                        st.markdown(f"<div class='xai-box'>{lime_text}</div>", unsafe_allow_html=True)
                    if st.button("Reset Explanation"): del st.session_state['veg_xai_data']; st.rerun()

            with t3:
                 co2, miles = nf.CarbonFootprintCalculator.calculate_impact(result['food'], 0.5)
                 st.metric("Estimated Footprint", f"~0.5 kg CO2e", delta="Low Impact")
                 st.markdown(f"<div class='xai-box' style='border-left-color: #28a745'>{nf.CarbonFootprintCalculator.generate_impact_text(result['food'], co2)}</div>", unsafe_allow_html=True)

# --- PAGE: BAKERY MODULE ---
def render_bakery_page():
    if st.button("← Back to Dashboard"):
        navigate_to("home")

    st.markdown("## 🍞 Bakery Quality Analysis")

    if 'bakery_model' not in res:
        st.error("Bakery model missing or failed to load. Ensure 'msff_bread_model.keras' is in `final_src`.")
        return

    model = res['bakery_model']
    xai = res['bakery_xai']

    c_side, c_main = st.columns([1, 2])

    with c_side:
        st.markdown("### Input Source")
        mode = st.radio("Select:", ["Upload Image", "Live Camera"], label_visibility="collapsed", key="bake_mode")

        img_file = None
        if mode == "Upload Image":
            img_file = st.file_uploader("Upload", type=['jpg','png','jpeg'], key="bake_upload")
        else:
            img_file = st.camera_input("Capture", key="bake_cam")

        if img_file:
            image_pil = Image.open(img_file)
            st.image(image_pil, caption="Sample", use_container_width=True)
            analyze = st.button("🔬 Analyze Bakery", type="primary", use_container_width=True)
        else:
            analyze = False

    with c_main:
        if img_file and analyze:
            with st.spinner("🍞 Analyzing Bakery Freshness..."):
                img_batch, img_orig = preprocess_bakery(image_pil)

                # Inference
                days_reg = model.predict(img_batch, verbose=0)[0][0]

                # Logic
                shelf_life = max(0.0, float(days_reg))
                status, color = get_bakery_freshness(shelf_life)
                food = "Bread"

                # Logic Fix
                score = min(1.0, shelf_life / 14.0)
                if status == 'Fresh' or status == 'Medium': score = max(score, 0.45)

                grade, _ = get_grade(score, status)
                expiry_str = (datetime.now() + timedelta(days=float(shelf_life))).strftime("%b %d")

                st.session_state['bake_res'] = {
                    'food': food, 'status': status, 'days': shelf_life,
                    'score': score, 'grade': grade, 'color': color, 'expiry': expiry_str,
                    'img_pre': img_batch[0], # Pass 3D array for XAI
                    'img_orig': img_orig,
                    'stat_idx': 0 # Dummy index for regression XAI
                }
                if 'bake_xai_data' in st.session_state: del st.session_state['bake_xai_data']

        if 'bake_res' in st.session_state:
            result = st.session_state['bake_res']

            m1, m2, m3 = st.columns(3)
            m1.markdown(f"<div class='metric-container'><h4>Detected</h4><h2 style='color:#1E3A5F'>{result['food']}</h2></div>", unsafe_allow_html=True)
            m2.markdown(f"<div class='metric-container' style='border-bottom: 5px solid {result['color']}'><h4>Condition</h4><h2 style='color:{result['color']}'>{result['status']}</h2></div>", unsafe_allow_html=True)
            m3.markdown(f"<div class='metric-container'><h4>Shelf Life</h4><h2>{result['days']:.1f} Days</h2><small>Exp: {result['expiry']}</small></div>", unsafe_allow_html=True)

            st.markdown("---")

            # UNIFIED TABS for Bakery
            t1, t2, t3 = st.tabs(["📉 Projections", "🧠 Explainable AI", "🌍 Sustainability"])

            with t1:
                col_a, col_b = st.columns([2,1])
                with col_a: st.plotly_chart(nf.QualityDecayPredictor.predict_decay_curve(result['score'], result['days']), use_container_width=True)
                with col_b:
                    st.write("**AI Storage Advice**")
                    for i, title, desc in nf.SmartStorageAdvisor.generate_recommendations(result['food'], result['score']*100, result['days']):
                        st.info(f"{i} **{title}**: {desc}")

            with t2:
                st.info("Visualizing decision logic with Grad-CAM and LIME.")
                if 'bake_xai_data' not in st.session_state:
                    if st.button("Generate Explanation (~10s)"):
                        with st.spinner("Computing..."):
                            hm, _ = xai.grad_cam(result['img_pre'])
                            focus = xai.analyze_heatmap_focus(hm)
                            lime = xai.lime_explanation(result['img_orig'], num_samples=50) # Use img_orig
                            st.session_state['bake_xai_data'] = {'hm': hm, 'focus': focus, 'lime': lime}
                            st.rerun()
                else:
                    xd = st.session_state['bake_xai_data']
                    v_grad = xai.visualize_grad_cam(result['img_pre'], xd['hm'], alpha=0.4)
                    v_lime = xai.visualize_lime(xd['lime'], result['stat_idx'], num_features=5)
                    r1, r2 = st.columns(2)
                    with r1:
                        st.image(v_grad, caption="Grad-CAM (Attention)", use_container_width=True, clamp=True)
                        grad_text = nf.XAITextGenerator.explain_gradcam(result['food'], result['status'], xd['focus'])
                        st.markdown(f"<div class='xai-box'>{grad_text}</div>", unsafe_allow_html=True)
                    with r2:
                        st.image(v_lime, caption="LIME Features", use_container_width=True, clamp=True)
                        lime_text = nf.XAITextGenerator.explain_lime(result['food'], result['status'])
                        st.markdown(f"<div class='xai-box'>{lime_text}</div>", unsafe_allow_html=True)
                    if st.button("Reset Explanation"): del st.session_state['bake_xai_data']; st.rerun()

            with t3:
                 co2, miles = nf.CarbonFootprintCalculator.calculate_impact(result['food'], 0.5)
                 st.metric("Estimated Footprint", f"~0.8 kg CO2e", delta="Medium Impact")
                 st.markdown(f"<div class='xai-box' style='border-left-color: #28a745'>{nf.CarbonFootprintCalculator.generate_impact_text(result['food'], co2)}</div>", unsafe_allow_html=True)

# --- PAGE: FRUIT MODULE ---
def render_fruit_page():
    if st.button("← Back to Dashboard"):
        navigate_to("home")

    st.markdown("## 🍎 Fruit Quality Analysis")

    if 'fruit_model' not in res:
        st.error("Fruit model missing or failed to load. Ensure 'fruit_shelf_life_model.h5' is in `final_src`.")
        return

    model = res['fruit_model']
    fruit_classes = res['fruit_classes']
    xai = res['fruit_xai']

    c_side, c_main = st.columns([1, 2])

    with c_side:
        st.markdown("### Input Source")
        mode = st.radio("Select:", ["Upload Image", "Live Camera"], label_visibility="collapsed", key="fruit_mode")

        img_file = None
        if mode == "Upload Image":
            img_file = st.file_uploader("Upload", type=['jpg','png','jpeg'], key="fruit_upload")
        else:
            img_file = st.camera_input("Capture", key="fruit_cam")

        if img_file:
            image_pil = Image.open(img_file)
            st.image(image_pil, caption="Sample", use_container_width=True)
            analyze = st.button("🔬 Analyze Fruit", type="primary", use_container_width=True)
        else:
            analyze = False

    with c_main:
        if img_file and analyze:
            with st.spinner("🍎 Analyzing Fruit Freshness..."):
                # Preprocess for Fruit Model (160x160)
                img_batch, img_orig = preprocess_fruit(image_pil)

                # Inference
                fruit_pred, fresh_pred, days_pred = model.predict(img_batch, verbose=0)

                # Decode
                fruit_idx = np.argmax(fruit_pred)
                food = fruit_classes[fruit_idx]

                # Freshness Logic (from snippet: < 0.5 is fresh)
                status = "Fresh" if fresh_pred[0][0] < 0.5 else "Spoiled"
                shelf_life = max(0.0, float(days_pred[0][0]))

                # Color logic
                color = "#06d6a0" if status == "Fresh" else "#ef476f"
                expiry_str = (datetime.now() + timedelta(days=float(shelf_life))).strftime("%b %d")

                # Fake score for consistency
                score = 0.9 if status == "Fresh" else 0.1
                grade, _ = get_grade(score, status)

                stat_idx = 0 if status == "Fresh" else 1

                st.session_state['fruit_res'] = {
                    'food': food, 'status': status, 'days': shelf_life,
                    'score': score, 'color': color, 'expiry': expiry_str,
                    'img_pre': img_batch[0], # Pass for XAI
                    'img_orig': img_orig,
                    'stat_idx': stat_idx,
                    'grade': grade
                }
                if 'fruit_xai_data' in st.session_state: del st.session_state['fruit_xai_data']

        if 'fruit_res' in st.session_state:
            result = st.session_state['fruit_res']

            m1, m2, m3 = st.columns(3)
            m1.markdown(f"<div class='metric-container'><h4>Detected</h4><h2 style='color:#1E3A5F'>{result['food']}</h2></div>", unsafe_allow_html=True)
            m2.markdown(f"<div class='metric-container' style='border-bottom: 5px solid {result['color']}'><h4>Condition</h4><h2 style='color:{result['color']}'>{result['status']}</h2></div>", unsafe_allow_html=True)
            m3.markdown(f"<div class='metric-container'><h4>Shelf Life</h4><h2>{result['days']:.1f} Days</h2><small>Exp: {result['expiry']}</small></div>", unsafe_allow_html=True)

            st.markdown("---")

            # UNIFIED TABS for Fruit
            t1, t2, t3 = st.tabs(["📉 Projections", "🧠 Explainable AI", "🌍 Sustainability"])

            with t1:
                col_a, col_b = st.columns([2,1])
                with col_a: st.plotly_chart(nf.QualityDecayPredictor.predict_decay_curve(result['score'], result['days']), use_container_width=True)
                with col_b:
                    st.write("**AI Storage Advice**")
                    for i, title, desc in nf.SmartStorageAdvisor.generate_recommendations(result['food'], result['score']*100, result['days']):
                        st.info(f"{i} **{title}**: {desc}")

            with t2:
                st.info("Visualizing decision logic with Grad-CAM and LIME.")
                if 'fruit_xai_data' not in st.session_state:
                    if st.button("Generate Explanation (~10s)"):
                        with st.spinner("Computing..."):
                            # Target Freshness Head (Index 1) for Fruit model
                            hm, _ = xai.grad_cam(result['img_pre'])
                            focus = xai.analyze_heatmap_focus(hm)
                            lime = xai.lime_explanation(result['img_orig'], target_head=1, num_samples=50) # Use img_orig
                            st.session_state['fruit_xai_data'] = {'hm': hm, 'focus': focus, 'lime': lime}
                            st.rerun()
                else:
                    xd = st.session_state['fruit_xai_data']
                    v_grad = xai.visualize_grad_cam(result['img_pre'], xd['hm'], alpha=0.4)
                    v_lime = xai.visualize_lime(xd['lime'], result['stat_idx'], num_features=5) # Fixed index
                    r1, r2 = st.columns(2)
                    with r1:
                        st.image(v_grad, caption="Grad-CAM (Attention)", use_container_width=True, clamp=True)
                        grad_text = nf.XAITextGenerator.explain_gradcam(result['food'], result['status'], xd['focus'])
                        st.markdown(f"<div class='xai-box'>{grad_text}</div>", unsafe_allow_html=True)
                    with r2:
                        st.image(v_lime, caption="LIME Features", use_container_width=True, clamp=True)
                        lime_text = nf.XAITextGenerator.explain_lime(result['food'], result['status'])
                        st.markdown(f"<div class='xai-box'>{lime_text}</div>", unsafe_allow_html=True)
                    if st.button("Reset Explanation"): del st.session_state['fruit_xai_data']; st.rerun()

            with t3:
                 co2, miles = nf.CarbonFootprintCalculator.calculate_impact(result['food'], 0.5)
                 st.metric("Estimated Footprint", f"~0.3 kg CO2e", delta="Low Impact")
                 st.markdown(f"<div class='xai-box' style='border-left-color: #28a745'>{nf.CarbonFootprintCalculator.generate_impact_text(result['food'], co2)}</div>", unsafe_allow_html=True)


# --- MAIN ROUTER ---
if st.session_state['page'] == 'home':
    render_home()
elif st.session_state['page'] == 'meat':
    render_meat_page()
elif st.session_state['page'] == 'fruit':
    render_fruit_page()
elif st.session_state['page'] == 'veg':
    render_veg_page()
elif st.session_state['page'] == 'bakery':
    render_bakery_page()
