import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import urllib.request
import tempfile
from torchvision import transforms
from torchvision.models.video import r3d_18

# -------------------- Streamlit UI Setup --------------------
st.set_page_config(page_title="Road Rage Detection", layout="centered")

st.markdown("""
    <style>
    body {
        background-color: #111;
        color: #eee;
    }
    .stButton > button {
        background-color: #d90429;
        color: white;
        font-size: 16px;
        border-radius: 10px;
        padding: 10px 24px;
    }
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: #111;
        text-align: center;
        color: gray;
        font-size: 14px;
        padding: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center; color:#f5f5f5;'>🚗 Road Rage Detection using AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload a road video to detect violent behavior using AI</p>", unsafe_allow_html=True)
st.markdown("---")

# -------------------- Load Model --------------------

@st.cache_resource
def load_model():
    model_url = "https://your-download-link.com/road_rage_model.pth"  # <-- REPLACE THIS LINK
    model_path = "road_rage_model.pth"

    if not os.path.exists(model_path):
        with st.spinner("🔄 Downloading model..."):
            urllib.request.urlretrieve(model_url, model_path)

    model = r3d_18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model().to(device)

# -------------------- Video Preprocessing --------------------

def preprocess_video(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // num_frames)

    frames = []
    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (112, 112))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        if len(frames) == num_frames:
            break
    cap.release()

    while len(frames) < num_frames:
        frames.append(np.zeros((112, 112, 3), dtype=np.uint8))

    transform = transforms.ToTensor()
    frames = torch.stack([transform(f) for f in frames])
    frames = frames.permute(1, 0, 2, 3)  # (C, T, H, W)
    return frames.unsqueeze(0).to(device)

# -------------------- Prediction --------------------

def predict(video_tensor):
    with torch.no_grad():
        outputs = model(video_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
        label = "🟥 Fight" if pred == 1 else "🟩 Non-Fight"
        return label, confidence, pred

# -------------------- Upload & Run --------------------

uploaded_file = st.file_uploader("📤 Upload a video (.mp4)", type=["mp4"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(video_path)

    with st.spinner("🔍 Processing video..."):
        video_tensor = preprocess_video(video_path)
        label, confidence, pred = predict(video_tensor)

    color = "#d90429" if pred == 1 else "#2a9d8f"
    st.markdown(f"""
    <div style="border:2px solid {color}; padding:20px; border-radius:12px; text-align:center;">
        <h2 style="color:{color};">{label}</h2>
        <p style="font-size:18px;">Confidence: <strong>{confidence*100:.2f}%</strong></p>
    </div>
    """, unsafe_allow_html=True)

    if pred == 1:
        st.warning("⚠️ Fight detected. You may call for help.")
        if st.button("📞 Call Police"):
            st.success("🚓 Alert sent to nearby authorities (mocked). Stay safe!")
    else:
        st.success("✅ No violent behavior detected.")

# -------------------- Footer --------------------

st.markdown("""
    <div class="footer">
        Made with ❤️ by <strong>Yash Dev</strong>
    </div>
""", unsafe_allow_html=True)
