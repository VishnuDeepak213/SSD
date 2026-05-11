"""
SDS - Smart Detection & Surveillance Dashboard
Deployment-ready version for Streamlit Cloud
"""
import os
import sys
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# ── Path setup: must happen before any src imports ────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SDS - Crowd Analysis Dashboard",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; color: white; text-align: center;
        padding: 1.5rem;
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
        border-radius: 10px; margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .feature-box-image {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 2rem; border-radius: 10px; margin: 1rem 0;
    }
    .feature-box-video {
        background: linear-gradient(135deg, #FF6B6B 0%, #FFE66D 100%);
        color: white; padding: 2rem; border-radius: 10px; margin: 1rem 0;
    }
    .stButton>button { background-color: #667eea; color: white; border-radius: 8px; border: none; }
</style>
""", unsafe_allow_html=True)

# ── Config loader ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_config():
    import yaml
    config_path = os.path.join(APP_DIR, "config", "config.yaml")
    if not os.path.exists(config_path):
        st.error(f"Config not found at: {config_path}")
        st.stop()
    with open(config_path) as f:
        return yaml.safe_load(f)

# ── Module initialiser ────────────────────────────────────────────────────────
@st.cache_resource
def initialize_modules(_config):
    """Load heavy ML modules once and cache. Underscore prefix skips hashing."""
    from src.detection.detector import PersonDetector
    from src.tracking.tracker import PersonTracker
    from src.threats.detector import ThreatDetector
    detector = PersonDetector(_config['detection'])
    tracker = PersonTracker(_config['tracking'])
    threat_detector = ThreatDetector(_config['threats'])
    return detector, tracker, threat_detector

# ── Image processing ──────────────────────────────────────────────────────────
def process_image(uploaded_file, features, config):
    from src.density.estimator import DensityEstimator
    from src.visualization.renderer import Visualizer

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image. Please upload a valid JPG/PNG.")
    h, w = frame.shape[:2]

    detector, tracker, threat_detector = initialize_modules(config)
    density_estimator = DensityEstimator(config['density'], (w, h))
    visualizer = Visualizer(config['visualization'])

    detections = detector(frame)
    results = {
        'frame': frame,
        'detections': detections,
        'num_persons': len(detections),
        'tracks': [],
        'density': None,
    }

    if features.get('tracking'):
        tracks = tracker.update(frame, detections)
        results['tracks'] = [t for t in tracks if t.is_confirmed()]

    if features.get('density'):
        density_grid, density_heatmap, density_alerts = density_estimator.estimate(detections)
        total_count = int(density_grid.sum())
        thr = config['density']['thresholds']
        if total_count >= thr['critical']:
            level = 'CRITICAL'
        elif total_count >= thr['high']:
            level = 'HIGH'
        elif total_count >= thr['medium']:
            level = 'MEDIUM'
        else:
            level = 'LOW'
        results['density'] = {
            'grid': density_grid, 'heatmap': density_heatmap,
            'level': level, 'count': total_count, 'alerts': density_alerts,
        }

    # Draw bounding boxes
    vis_frame = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])
        conf = det[4]
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis_frame, f'{conf:.2f}', (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    results['visualized'] = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
    results['original'] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return results

# ══════════════════════════════════════════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════════════════════════════════════════

def show_home_page():
    st.markdown("<div class='main-header'>👥 SDS - Smart Detection & Surveillance</div>",
                unsafe_allow_html=True)
    st.markdown("Welcome to the **SDS Crowd Analysis Dashboard**. "
                "This system provides real-time analysis of crowds and individuals in images and videos.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='feature-box-image'>
            <h2>🖼️ IMAGE ANALYSIS</h2>
            <p>Upload a single image and get instant analysis:</p>
            <ul>
                <li>👤 Person Detection (YOLOv8)</li>
                <li>🎯 Individual Tracking (DeepSORT)</li>
                <li>📊 Crowd Density Estimation</li>
                <li>⚠️ Threat Detection</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='feature-box-video'>
            <h2>🎥 VIDEO ANALYSIS</h2>
            <p>Upload a video for comprehensive analysis:</p>
            <ul>
                <li>🎬 Frame-by-frame Detection</li>
                <li>📈 Crowd Density Over Time</li>
                <li>🔄 Optical Flow Analysis</li>
                <li>🚨 Anomaly Detection</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    ### 📋 Key Features
    - **YOLOv8 Detection**: Fast and accurate person detection
    - **DeepSORT Tracking**: Multi-object tracking across frames
    - **Crowd Density**: Grid-based density estimation with heatmap
    - **Optical Flow**: Movement analysis
    - **Threat Detection**: Anomaly and panic detection

    ### 🚀 Getting Started
    1. Select **Image Analysis** or **Video Analysis** from the sidebar
    2. Upload your file
    3. Choose analysis features
    4. View results with visualisations
    """)


def show_image_analysis():
    st.markdown("# 🖼️ Image Analysis")
    st.markdown("Upload an image to detect and analyse crowds")

    config = load_config()

    # Warm up modules in background so first run is faster
    with st.spinner("Loading ML models (first load may take ~30s)..."):
        try:
            initialize_modules(config)
        except Exception as e:
            st.error(f"Failed to load models: {e}")
            st.info("Make sure `yolov8n.pt` is committed to your GitHub repository.")
            return

    uploaded_file = st.file_uploader(
        "Select Image", type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Choose an image file to analyse"
    )

    if uploaded_file:
        st.markdown("### ⚙️ Analysis Features")
        col1, col2, col3 = st.columns(3)
        with col1:
            show_detection = st.checkbox("👤 Person Detection", value=True)
        with col2:
            show_density = st.checkbox("📊 Crowd Density", value=True)
        with col3:
            show_tracking = st.checkbox("🎯 Tracking", value=False)

        if st.button("🔍 Analyse Image", use_container_width=True):
            with st.spinner("⏳ Processing image..."):
                try:
                    features = {
                        'detection': show_detection,
                        'tracking': show_tracking,
                        'density': show_density,
                        'flow': False,
                        'threats': False,
                    }
                    results = process_image(uploaded_file, features, config)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(results['original'], caption="Original Image", use_column_width=True)
                    with col2:
                        st.image(results['visualized'], caption="Detection Result", use_column_width=True)

                    st.markdown("---")
                    st.markdown("### 📊 Analysis Results")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("👤 Persons Detected", results['num_persons'])

                    if results['density']:
                        with col2:
                            level = results['density']['level']
                            color_map = {'LOW': '🟢', 'MEDIUM': '🟡', 'HIGH': '🟠', 'CRITICAL': '🔴'}
                            st.metric("📊 Density Level", f"{color_map.get(level,'')} {level}")
                        with col3:
                            st.metric("🔢 Person Count", results['density']['count'])

                        # Show density heatmap
                        st.markdown("#### 🌡️ Density Heatmap")
                        heatmap_rgb = cv2.cvtColor(results['density']['heatmap'], cv2.COLOR_BGR2RGB)
                        st.image(heatmap_rgb, caption="Crowd Density Heatmap", use_column_width=True)

                        # Show density alerts
                        if results['density']['alerts']:
                            st.markdown("#### ⚠️ Density Alerts")
                            for alert in results['density']['alerts']:
                                if alert['level'] == 'CRITICAL':
                                    st.error(f"🔴 {alert['message']}")
                                elif alert['level'] == 'HIGH':
                                    st.warning(f"🟠 {alert['message']}")
                                else:
                                    st.info(f"🟡 {alert['message']}")

                    # Density grid chart
                    if results['density'] is not None:
                        st.markdown("#### 📈 Density Grid")
                        grid = results['density']['grid']
                        fig = px.imshow(
                            grid, color_continuous_scale='Reds',
                            labels=dict(color="Person Count"),
                            title="Crowd Density Grid"
                        )
                        fig.update_layout(paper_bgcolor='white', font=dict(color='#0f172a'))
                        st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
    else:
        st.info("📤 Upload an image to begin analysis")


def show_video_analysis():
    st.markdown("# 🎥 Video Analysis")
    st.markdown("Upload a video to analyse crowd dynamics over time")

    config = load_config()

    uploaded_file = st.file_uploader(
        "Select Video", type=['mp4', 'avi', 'mov', 'mkv'],
        help="Choose a video file to analyse"
    )

    if uploaded_file:
        st.markdown("### ⚙️ Analysis Options")
        col1, col2 = st.columns(2)
        with col1:
            show_density = st.checkbox("📊 Density Analysis", value=True, key="v_dens")
        with col2:
            sample_rate = st.slider("Sample every N frames", 1, 30, 10,
                                    help="Higher = faster but less detail")
        max_frames = st.slider("Max Frames to Process", 50, 300, 100, step=50)

        if st.button("▶️ Analyse Video", use_container_width=True):
            with st.spinner("Loading ML models..."):
                try:
                    initialize_modules(config)
                except Exception as e:
                    st.error(f"Failed to load models: {e}")
                    return

            # Save upload to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                from src.detection.detector import PersonDetector
                from src.density.estimator import DensityEstimator

                cap = cv2.VideoCapture(tmp_path)
                if not cap.isOpened():
                    st.error("Could not open video file.")
                    return

                ret, first_frame = cap.read()
                if not ret:
                    st.error("Could not read first frame.")
                    return
                h, w = first_frame.shape[:2]

                detector = PersonDetector(config['detection'])
                density_estimator = DensityEstimator(config['density'], (w, h))

                frame_counts = []
                density_levels = []
                person_counts = []
                preview_frames = []

                progress = st.progress(0)
                status = st.empty()
                frame_num = 0
                processed = 0

                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                while processed < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_num += 1
                    if frame_num % sample_rate != 0:
                        continue

                    status.text(f"Processing frame {frame_num}...")
                    detections = detector(frame)
                    density_grid, _, _ = density_estimator.estimate(detections)
                    total = int(density_grid.sum())

                    thr = config['density']['thresholds']
                    if total >= thr['critical']:
                        lvl = 'CRITICAL'
                    elif total >= thr['high']:
                        lvl = 'HIGH'
                    elif total >= thr['medium']:
                        lvl = 'MEDIUM'
                    else:
                        lvl = 'LOW'

                    frame_counts.append(frame_num)
                    person_counts.append(total)
                    density_levels.append(lvl)

                    if len(preview_frames) < 4:
                        vis = frame.copy()
                        for det in detections:
                            x1, y1, x2, y2 = map(int, det[:4])
                            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        preview_frames.append(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

                    processed += 1
                    progress.progress(min(processed / max_frames, 1.0))

                cap.release()
                os.unlink(tmp_path)
                progress.empty()
                status.empty()

                if not frame_counts:
                    st.warning("No frames were processed.")
                    return

                st.success(f"✅ Processed {processed} frames from {frame_num} total")

                # Person count chart
                df = pd.DataFrame({
                    'Frame': frame_counts,
                    'Person Count': person_counts,
                    'Density Level': density_levels
                })
                fig = px.line(df, x='Frame', y='Person Count',
                              title='Person Count Over Time',
                              color_discrete_sequence=['#667eea'])
                fig.update_layout(paper_bgcolor='white', font=dict(color='#0f172a'))
                st.plotly_chart(fig, use_container_width=True)

                # Summary stats
                col1, col2, col3 = st.columns(3)
                col1.metric("Peak Count", max(person_counts))
                col2.metric("Avg Count", f"{sum(person_counts)/len(person_counts):.1f}")
                col3.metric("Frames Analysed", processed)

                # Preview frames
                if preview_frames:
                    st.markdown("#### 🖼️ Sample Frames")
                    cols = st.columns(len(preview_frames))
                    for i, (col, img) in enumerate(zip(cols, preview_frames)):
                        col.image(img, caption=f"Frame {frame_counts[i]}", use_column_width=True)

            except Exception as e:
                st.error(f"❌ Video analysis error: {str(e)}")
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
    else:
        st.info("📤 Upload a video to begin analysis")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    st.sidebar.markdown("# 🎬 SDS Dashboard")
    page = st.sidebar.radio(
        "Select Analysis Type",
        ["🏠 Home", "🖼️ Image Analysis", "🎥 Video Analysis"]
    )
    st.sidebar.divider()
    st.sidebar.info("📧 ml-team@company.com")

    if page == "🏠 Home":
        show_home_page()
    elif page == "🖼️ Image Analysis":
        show_image_analysis()
    elif page == "🎥 Video Analysis":
        show_video_analysis()

if __name__ == "__main__":
    main()
