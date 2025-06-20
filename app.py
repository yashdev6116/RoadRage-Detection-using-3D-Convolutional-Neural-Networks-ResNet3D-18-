
import streamlit as st
import torch
import torch.nn as nn
import tempfile
import cv2
import numpy as np
import pandas as pd
from torchvision import transforms
from torchvision.models.video import r3d_18
from datetime import datetime
import os

# -------------------- Config --------------------
st.set_page_config(page_title="Road Rage Detection", layout="centered")
LOG_PATH = "prediction_log.csv"

# -------------------- Dark Theme Styling --------------------
st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #121212;
        color: #f5f5f5;
    }

    h1, h2, h3, h4, h5, h6, p, div, label, span {
        color: #f5f5f5 !important;
    }

    .stButton>button {
        background-color: #6b4226;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
    }

    .stFileUploader, .stFileUploader>div>div {
        background-color: #1f1f1f;
        border-radius: 10px;
        border: 1px solid #444;
        color: #f5f5f5;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #1f1f1f !important;
        color: #f5f5f5 !important;
        border: none;
    }

    .stTabs [aria-selected="true"] {
        border-bottom: 3px solid #e63946;
        color: #e63946 !important;
    }

    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: center;
        color: #888;
        font-size: 14px;
        padding: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)


# -------------------- Header --------------------
st.markdown("<h1 style='text-align:center;'>🚗 Road Rage Detection using AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload a road video to detect violent behavior using AI</p>", unsafe_allow_html=True)
st.markdown("---")

# -------------------- Load Model --------------------
@st.cache_resource
def load_model():
    model = r3d_18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("road_rage_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -------------------- Preprocess Video --------------------
def preprocess_video(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // num_frames)
    frames = []

    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (112, 112))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        if len(frames) == num_frames: break
    cap.release()

    while len(frames) < num_frames:
        frames.append(np.zeros((112, 112, 3), dtype=np.uint8))

    transform = transforms.ToTensor()
    frames = torch.stack([transform(f) for f in frames])  # (T, C, H, W)
    frames = frames.permute(1, 0, 2, 3)  # (C, T, H, W)
    return frames.unsqueeze(0).to(device)  # (1, C, T, H, W)

# -------------------- Prediction --------------------
def predict(video_tensor):
    with torch.no_grad():
        outputs = model(video_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
        label = "🟥 Fight" if pred == 1 else "🟩 Non-Fight"
        return label, confidence, pred

# -------------------- Logging --------------------
def log_prediction(filename, prediction, confidence):
    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = pd.DataFrame([[filename, prediction, f"{confidence*100:.2f}%", time_now]],
                         columns=["Filename", "Prediction", "Confidence", "Timestamp"])
    if os.path.exists(LOG_PATH):
        log_df = pd.read_csv(LOG_PATH)
        log_df = pd.concat([entry, log_df], ignore_index=True)
    else:
        log_df = entry
    log_df.to_csv(LOG_PATH, index=False)

# -------------------- Tabs --------------------
tab1, tab2 = st.tabs(["📤 Predict", "📜 History"])

# -------------------- Tab 1: Upload & Predict --------------------
with tab1:
    uploaded_file = st.file_uploader("📤 Upload a video (.mp4)", type=["mp4"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        st.video(video_path)

        with st.spinner("🔍 Processing the video..."):
            video_tensor = preprocess_video(video_path)
            label, confidence, pred = predict(video_tensor)

        color = "#d90429" if pred == 1 else "#2a9d8f"
        st.markdown(f"""
        <div style="border:2px solid {color}; padding:20px; border-radius:12px; text-align:center; background-color:#1e1e1e;">
            <h2 style="color:{color};">{label}</h2>
            <p style="font-size:18px; color:white;">Confidence: <strong>{confidence*100:.2f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)

        log_prediction(uploaded_file.name, label, confidence)

        if pred == 1:
            st.warning("⚠️ Violent behavior detected! You may call for help.")
            if st.button("📞 Call Police"):
                st.success("🚓 Alert sent to nearby authorities. Stay safe!")
        else:
            st.success("✅ No violent behavior detected.")

# -------------------- Tab 2: History --------------------
with tab2:
    st.subheader("📜 Previous Predictions")
    if os.path.exists(LOG_PATH):
        history_df = pd.read_csv(LOG_PATH)
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No prediction history found.")

# -------------------- Footer --------------------
st.markdown("""
    <div class="footer">
        Made with ❤️ by <strong>Yash Dev</strong>
    </div>
""", unsafe_allow_html=True)
