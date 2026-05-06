import streamlit as st

if "app_started" not in st.session_state:
    st.session_state.app_started = True
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2
import time
import os
from collections import deque
import queue
import pandas as pd
import numpy as np
from collections import deque, defaultdict

st.set_page_config(
    page_title="Live Object Detection & Tracing",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="collapsed", 
)

st.markdown("""
<style>
    /* Hide sidebar completely */
    [data-testid="stSidebar"] {
        display: none;
    }
    /* Adjust main content area to take full width */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    #MainMenu, footer, header {visibility: hidden;}

    .stApp {
        background: 
            linear-gradient(rgba(0,0,0,0.04) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,0,0,0.04) 1px, transparent 1px),
            linear-gradient(135deg, #f5f7fa 0%, #e4ebf5 100%);
        background-size: 30px 30px, 30px 30px, auto;
    }

    .custom-card {
        background: white;
        border-radius: 16px;
        padding: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
        margin-bottom: 15px;
    }

    h1 {
        color: #1e3b5a;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    /* Centered video wrapper */
    .video-wrapper {
        background: #ffffff;
        border-radius: 20px;
        padding: 12px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        border: 1px solid #e0e8f0;
        margin: 0 auto;
        max-width: 1100px;
        background-image: 
            linear-gradient(rgba(0,0,0,0.06) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,0,0,0.06) 1px, transparent 1px);
        background-size: 20px 20px;
    }
    
    /* Control panel styling */
    .controls-container {
        background: white;
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        border: 1px solid #eef2f6;
    }
    .alert-log {
        background: #ffffff;
        border-radius: 16px;
        padding: 15px;
        margin-top: 1.5rem;
        border: 1px solid #eef2f6;
        max-height: 300px;
        overflow-y: auto;
    }
    .alert-entry {
        font-family: monospace;
        font-size: 0.85rem;
        padding: 4px 0;
        border-bottom: 1px solid #f0f0f0;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Model Loading ----------
@st.cache_resource(show_spinner=False)
def load_model():
    """Load YOLO model once and cache it to avoid reloading on every interaction."""
    return YOLO("yolo11n.pt") 

model = load_model()

# ---------- Directories & Queues for Alerts and Analytics ----------
SAVE_DIR = "detected_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

alert_queue = queue.Queue()          
analytics_queue = queue.Queue()      

# ---------- Control Widgets ----------
class_names = list(model.names.values())
default_target = "person" if "person" in class_names else class_names[0]

# ---------- SESSION STATE DEFAULTS ----------
defaults = {
    "TARGET_OBJECT": default_target,
    "CONFIDENCE": 0.5,
    "SAVE_COOLDOWN": 3.0,
    "ALERT_CONSISTENCY": 2,
    "CROWD_THRESHOLD": 10,
    "refresh_interval": 1,
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Session state to preserve data across Streamlit reruns
if "alert_log_entries" not in st.session_state:
    st.session_state.alert_log_entries = []

if "chart_history" not in st.session_state:
    st.session_state.chart_history = deque(maxlen=50)

if "latest_tracking" not in st.session_state:
    st.session_state.latest_tracking = []

# ---------- Helper Function: Draw Pill-shaped Text on Frame ----------
def draw_pill_text(img, text, pos, bg_color=(0,0,0,140), text_color=(255,255,255)):
    """
    Draws a text label with a semi-transparent background pill.
    
    Args:
        img: image array (BGR)
        text: string to display
        pos: (x, y) anchor point (bottom-left of text)
        bg_color: (B,G,R,alpha) background color
        text_color: (B,G,R) text color
    Returns:
        Modified image
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    pad = 10

    # Create overlay for transparency
    overlay = img.copy()
    cv2.rectangle(overlay, (x-pad, y-h-pad-4), (x+w+pad, y+pad), bg_color[:3], cv2.FILLED)
    alpha = bg_color[3]/255.0
    img[y-h-pad-4:y+pad, x-pad:x+w+pad] = cv2.addWeighted(
        img[y-h-pad-4:y+pad, x-pad:x+w+pad], 1-alpha,
        overlay[y-h-pad-4:y+pad, x-pad:x+w+pad], alpha, 0
    )
    cv2.putText(img, text, (x, y), font, scale, text_color, thickness, cv2.LINE_AA)
    return img

# ---------- Video Frame Callback ----------
def video_frame_callback(frame):
    """
    Process each video frame: run YOLO detection/tracking, annotate, count objects,
    trigger alerts, and save frames when conditions are met.
    """
    # Convert incoming frame to OpenCV BGR format
    img = frame.to_ndarray(format="bgr24")
    t0 = time.time()

    # Run YOLO tracking (persist=True keeps track IDs across frames)
    results = model.track(img, persist=True, conf=CONFIDENCE, imgsz=640, verbose=False)

    # Annotate frame with bounding boxes, labels, and confidence scores
    annotated = results[0].plot(conf=True, labels=True, boxes=True)

    # ---- Count objects by class ----
    cls_array = []
    if results and results[0].boxes and results[0].boxes.cls is not None:
        cls_array = results[0].boxes.cls.cpu().numpy().astype(int)
    object_counts = {}
    for c in cls_array:
        name = model.names[int(c)]
        object_counts[name] = object_counts.get(name, 0) + 1

    # Store counts in session state for potential display elsewhere
    if "global_counts" not in st.session_state:
        st.session_state.global_counts = {}
    st.session_state.global_counts = object_counts

    # Extract specific metrics
    total_objects = sum(object_counts.values())
    person_count = object_counts.get("person", 0)
    target_present = TARGET_OBJECT in object_counts

    # ---- Extract tracking information ----
    tracking_info = []
    if results and results[0].boxes is not None and results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
        min_len = min(len(ids), len(boxes_xyxy), len(results[0].boxes.cls))
        for i in range(min_len):
            track_id = ids[i]
            cls_idx = int(results[0].boxes.cls[i].item())
            cls_name = model.names[cls_idx]
            tracking_info.append({
                "ID": track_id,
                "Class": cls_name,
                "x1": int(boxes_xyxy[i][0]), "y1": int(boxes_xyxy[i][1]),
                "x2": int(boxes_xyxy[i][2]), "y2": int(boxes_xyxy[i][3])
            })

    # Send data to analytics queue 
    timestamp = time.time()
    analytics_queue.put({
        "timestamp": timestamp,
        "counts": object_counts,
        "tracking": tracking_info
    })

    # ---- Alert Logic ----
    thread_state = st.session_state.setdefault("thread_state", {
        "target_history": deque(maxlen=ALERT_CONSISTENCY),
        "fps": 0.0,
        "last_save_time": 0.0,
    })

    # Update target detection history for consistency check
    thread_state["target_history"].append(target_present)
    target_alert = (len(thread_state["target_history"]) == ALERT_CONSISTENCY and
                    all(thread_state["target_history"]))

    # Crowd alert based on person count threshold
    crowd_alert = person_count >= CROWD_THRESHOLD

    # ---- Draw HUD information on frame ----
    draw_pill_text(annotated, f"People: {person_count}", (12, 32),
                   bg_color=(0,0,0,160), text_color=(255,255,0))
    draw_pill_text(annotated, f"Total Objects: {total_objects}", (12, 72),
                   bg_color=(0,0,0,160), text_color=(0,255,255))

    # Target presence indicator
    status_icon = "✅" if target_alert else "❌"
    status_color = (0,255,0) if target_alert else (255,255,255)
    draw_pill_text(annotated, f"{TARGET_OBJECT.upper()} {status_icon}", (12, 112),
                   bg_color=(0,0,0,160), text_color=status_color)

    # Crowd alert warning
    if crowd_alert:
        draw_pill_text(annotated, "🚨 CROWD ALERT", (12, 152),
                       bg_color=(0,0,200,180), text_color=(255,255,255))

    # Compute and display FPS
    fps = 1 / (time.time() - t0 + 1e-6)
    thread_state["fps"] = 0.9 * thread_state["fps"] + 0.1 * fps  
    fps_text = f"{thread_state['fps']:.1f} FPS"
    (tw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    draw_pill_text(annotated, fps_text,
                   (annotated.shape[1] - tw - 20, 30),
                   bg_color=(0,0,0,160), text_color=(0,255,255))

    # ---- Save frame to disk if alert is active and cooldown passed ----
    now = time.time()
    if target_alert or crowd_alert:
        if now - thread_state["last_save_time"] > SAVE_COOLDOWN:
            trigger = "target" if target_alert else "crowd"
            fname = f"{trigger}_{TARGET_OBJECT.replace(' ', '_')}_{int(now)}.jpg"
            cv2.imwrite(os.path.join(SAVE_DIR, fname), annotated)
            thread_state["last_save_time"] = now
            alert_queue.put(f"📸 {fname} " +
                            ("(crowd alert)" if crowd_alert else "(target alert)"))

    # Return annotated frame to Streamlit-WebRTC
    return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# ---------- Main UI: Full Side Layout ----------
st.title("🎥 Live Object Detection & Tracing")
st.markdown("Point your camera at objects to identify them in real-time.")

left_col, right_col = st.columns([2, 1])

# ----- LEFT: CAMERA -----
with left_col:
    st.markdown('<div class="video-wrapper">', unsafe_allow_html=True)
    webrtc_streamer(
    key="municipal-detection",
    video_frame_callback=video_frame_callback,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)
    st.markdown('</div>', unsafe_allow_html=True)

# ----- RIGHT: CONTROLS + ALERTS -----
with right_col:

    # ==== CONTROLS ====
    st.markdown("### ⚙️ Controls")

    TARGET_OBJECT = st.selectbox(
    "🎯 Alert Object Class",
    class_names,
    key="TARGET_OBJECT"
)
    if "TARGET_OBJECT" not in st.session_state:
     st.session_state["TARGET_OBJECT"] = default_target

    CONFIDENCE = st.slider("🔍 Detection Confidence", 0.1, 1.0, 0.5, 0.05)

    SAVE_COOLDOWN = st.slider("⏱️ Save Cooldown (sec)", 1.0, 10.0, 3.0, 0.5)

    ALERT_CONSISTENCY = st.slider("🔄 Alert Consistency (frames)", 1, 5, 2)

    CROWD_THRESHOLD = st.slider("👥 Crowd Alert (people)", 1, 30, 10, 1)

    refresh_interval = st.slider("⏳ Chart Refresh (sec)", 1, 5, 1)

    st.markdown("---")

    # ==== ALERT LOG ====
    st.markdown("### 🛎️ Alert Log")
    alert_log_placeholder = st.empty()

    while not alert_queue.empty():
        st.session_state.alert_log_entries.append(alert_queue.get_nowait())

    with alert_log_placeholder.container():
        if not st.session_state.alert_log_entries:
            st.info("No alerts yet.")
        else:
            for entry in reversed(st.session_state.alert_log_entries[-10:]):
                st.markdown(
                    f'<div class="alert-entry">📸 {entry}</div>',
                    unsafe_allow_html=True
                )

st.caption("⚡ Real-time detection · Object tracking · Smart alerts")
