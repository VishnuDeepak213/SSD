"""
SDS - Smart Detection & Surveillance Dashboard
Deployment-ready version for Streamlit Cloud
"""
import streamlit as st
import cv2
import yaml
import numpy as np
import tempfile
import os
import sys
from pathlib import Path
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ── Path setup BEFORE any src imports ─────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ── Page config must come BEFORE any other st calls ───────────────────────────
st.set_page_config(
    page_title="SDS - Crowd Analysis Dashboard",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Import src modules with clear error messages ───────────────────────────────
try:
    from src.detection.detector import PersonDetector
    from src.tracking.tracker import PersonTracker
    from src.density.estimator import DensityEstimator
    from src.threats.detector import ThreatDetector
    from src.visualization.renderer import Visualizer
except Exception as e:
    st.error(f"❌ Import error: {e}")
    st.error("Make sure all required files are in the src/ folder")
    st.stop()

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 3rem; color: white; text-align: center; padding: 1.5rem;
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
        border-radius: 10px; margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .feature-box-image {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 2rem; border-radius: 10px; margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .feature-box-video {
        background: linear-gradient(135deg, #FF6B6B 0%, #FFE66D 100%);
        color: white; padding: 2rem; border-radius: 10px; margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
    }
    .feature-box {
        background-color: #f0f2f6; padding: 1.5rem; border-radius: 10px;
        margin: 1rem 0; border-left: 4px solid #667eea;
    }
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 1.5rem; border-radius: 10px; text-align: center;
    }
    .stat-value { font-size: 2rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ── Config loader ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_config():
    config_path = ROOT_DIR / "config" / "config.yaml"
    if not config_path.exists():
        st.error(f"❌ Missing config file: {config_path}")
        st.stop()
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# ── Module initialiser — underscore prefix skips st.cache_resource hashing ────
@st.cache_resource
def initialize_modules(_config):
    """Cache heavy ML models. _config prefix prevents unhashable dict error."""
    detector = PersonDetector(_config['detection'])
    tracker = PersonTracker(_config['tracking'])
    threat_detector = ThreatDetector(_config['threats'])
    return detector, tracker, threat_detector

# ── Helpers ────────────────────────────────────────────────────────────────────
def filter_detections_by_confidence(detections, confidence_threshold):
    if len(detections) == 0:
        return detections
    filtered = []
    for det in detections:
        if len(det) >= 5 and det[4] >= confidence_threshold:
            x1, y1, x2, y2 = det[:4]
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0 or w * h < 400:
                continue
            aspect = h / w if w > 0 else 0
            if aspect < 0.3 or aspect > 10:
                continue
            filtered.append(det)
    if not filtered:
        ncols = len(detections[0]) if len(detections) > 0 else 0
        return np.array([]).reshape(0, ncols)
    return np.array(filtered)


def limit_detections(detections, max_count):
    if len(detections) == 0 or max_count is None:
        return detections
    if len(detections) <= max_count:
        return detections
    top_idx = np.argsort(-detections[:, 4])[:max_count]
    return detections[top_idx]


def process_image(uploaded_file, features, config):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image. Please upload a valid JPG/PNG.")
    h, w = frame.shape[:2]

    detector, tracker, threat_detector = initialize_modules(config)
    density_estimator = DensityEstimator(config['density'], (h, w))
    visualizer = Visualizer(config['visualization'])

    detections = detector.detect(frame)
    results = {
        'frame': frame, 'detections': detections,
        'num_persons': len(detections), 'tracks': [], 'density': None
    }

    if features['tracking']:
        tracks = tracker.update(frame, detections)
        confirmed = [t for t in tracks if t.is_confirmed()]
        results['tracks'] = confirmed
        results['num_persons'] = len(confirmed)

    if features['density']:
        density_input = results['tracks'] if results['tracks'] else detections
        density_grid, density_heatmap, density_alerts = density_estimator.estimate(density_input)
        total_count = int(density_grid.sum())
        thr = config['density']['thresholds']
        if total_count >= thr['critical']:   level = 'CRITICAL'
        elif total_count >= thr['high']:     level = 'HIGH'
        elif total_count >= thr['medium']:   level = 'MEDIUM'
        else:                                level = 'LOW'
        results['density'] = {
            'grid': density_grid, 'heatmap': density_heatmap,
            'level': level, 'count': total_count, 'alerts': density_alerts
        }

    vis_frame = visualizer.render(
        frame,
        tracks=results['tracks'] if results['tracks'] else None,
        detections=None if results['tracks'] else detections,
        density_heatmap=results['density']['heatmap'] if results['density'] else None,
        alerts=results['density']['alerts'] if results['density'] else None
    )
    results['visualized'] = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
    results['original']   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return results

# ══════════════════════════════════════════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════════════════════════════════════════

def show_home_page():
    st.markdown("<div class='main-header'>👥 SDS - Smart Detection & Surveillance</div>",
                unsafe_allow_html=True)
    st.markdown("Welcome to the **SDS Crowd Analysis Dashboard**.\n\n"
                "This system provides real-time analysis of crowds and individuals in images and videos.")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='feature-box-image'>
            <h2>🖼️ IMAGE ANALYSIS</h2>
            <p>Upload a single image and get instant analysis:</p>
            <ul>
                <li>👤 Person Detection</li>
                <li>🎯 Individual Tracking</li>
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
                <li>🎬 Real-time Detection</li>
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
    - **Crowd Density**: Grid-based density estimation
    - **Optical Flow**: Movement analysis
    - **Threat Detection**: Anomaly and panic detection

    ### 🚀 Getting Started
    1. Select **Image Analysis** or **Video Analysis** from the sidebar
    2. Upload your file
    3. Choose analysis features
    4. View results with visualizations
    """)


def show_image_analysis():
    st.markdown("# 🖼️ Image Analysis")
    st.markdown("Upload an image to detect and analyze crowds")

    config = load_config()

    # ── Sidebar settings ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Detection Settings")
        st.markdown("*Optimized for custom trained model*")

        available_weights = [p.name for p in ROOT_DIR.glob("*.pt")]
        default_weight = config['detection'].get('model', 'yolov8n.pt')
        selected_weight = st.selectbox(
            "🧠 Model Weights",
            options=available_weights if available_weights else [default_weight],
            index=(available_weights.index(default_weight)
                   if default_weight in available_weights else 0),
            help="Choose YOLO weights."
        )

        # FIX: don't import torch at top level — lazy import here only
        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except Exception:
            cuda_available = False

        device_choice = st.selectbox(
            "⚙️ Device",
            options=["cpu", "cuda"] if cuda_available else ["cpu"],
            index=0,
            help="Use CUDA if available for speed."
        )

        confidence = st.slider(
            "🎯 Confidence Threshold", 0.05, 0.6,
            float(config['detection'].get('confidence', 0.15)), 0.05,
            help="Lower = more detections. Recommended: 0.10–0.20 for crowds"
        )
        max_detections = st.slider(
            "📊 Max Detections", 100, 2000,
            int(config['detection'].get('max_det', 300)), 100,
            help="Maximum persons to detect."
        )
        high_accuracy = st.checkbox(
            "🔬 High Accuracy Mode", value=False,
            help="Enables mild TTA for cleaner results (slower on CPU)."
        )

        config['detection']['model']      = selected_weight
        config['detection']['device']     = device_choice
        config['detection']['confidence'] = confidence
        config['detection']['max_det']    = max_detections
        if high_accuracy:
            config['detection']['tta_scales'] = [1.0, 1.25]
            config['detection']['flip_tta']   = False
            config['detection']['nms_iou']    = 0.55
            config['detection']['augment']    = True
        else:
            config['detection']['tta_scales'] = [1.0]
            config['detection']['flip_tta']   = False
            config['detection']['nms_iou']    = 0.5
            config['detection']['augment']    = False

        st.markdown("---")
        st.info("💡 **Tip**: For dense crowds, use 0.10–0.15 confidence")

    uploaded_file = st.file_uploader(
        "Select Image", type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Choose an image file to analyze", key="image_uploader"
    )

    if uploaded_file:
        if ('last_uploaded_image' not in st.session_state or
                st.session_state.last_uploaded_image != uploaded_file.name):
            st.session_state.image_results = None
            st.session_state.last_uploaded_image = uploaded_file.name

        st.markdown("### ⚙️ Analysis Features")
        col1, col2, col3 = st.columns(3)
        with col1: show_detection = st.checkbox("👤 Person Detection", value=True)
        with col2: show_density   = st.checkbox("📊 Crowd Density",    value=True)
        with col3: show_tracking  = st.checkbox("🎯 Tracking",         value=False)

        if st.button("🔍 Analyze Image", use_container_width=True):
            with st.spinner("⏳ Processing image..."):
                try:
                    features = {
                        'detection': show_detection, 'tracking': show_tracking,
                        'density': show_density, 'flow': False, 'threats': False
                    }
                    st.session_state.image_results = process_image(uploaded_file, features, config)
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

        results = st.session_state.get('image_results')
        if isinstance(results, dict) and 'original' in results and 'visualized' in results:
            density_result = results.get('density')
            h, w = results['original'].shape[:2]
            area_mpx = (h * w) / 1_000_000.0
            density_per_mpx = results.get('num_persons', 0) / max(area_mpx, 1e-6)

            main_col, side_col = st.columns([2, 1])
            with main_col:
                st.image(results['visualized'], caption="Detection Result", use_column_width=True)
                st.caption(f"People: {results.get('num_persons', 0)} | Density/MPx: {density_per_mpx:.2f}")
            with side_col:
                st.metric("People", int(results.get('num_persons', 0)))
                st.metric("Density per MPx", f"{density_per_mpx:.2f}")
                if density_result and density_result.get('heatmap') is not None:
                    heatmap_rgb = cv2.cvtColor(density_result['heatmap'], cv2.COLOR_BGR2RGB)
                    st.image(heatmap_rgb, caption="Spatial Density Heatmap", use_column_width=True)
                else:
                    st.info("Enable Crowd Density to view heatmap")
                if density_result:
                    st.metric("Density Level", density_result.get('level', 'N/A'))
                    st.metric("Count", int(density_result.get('count', 0)))

            with st.expander("Show Original Image"):
                st.image(results['original'], caption="Original Image", use_column_width=True)
    else:
        st.info("📤 Upload an image to begin analysis")


def show_video_analysis():
    st.markdown("# 🎥 Video Analysis")
    st.markdown("Upload a video to analyze crowd dynamics over time")

    config = load_config()

    # ── Show cached results from previous run ────────────────────────────────
    if 'video_results' in st.session_state and st.session_state.video_results:
        results = st.session_state.video_results
        st.info("📊 Displaying results from previous analysis. Upload a new video to start fresh.")

        # FIX: video bytes are stored directly in session_state, not re-read from disk
        if results.get('video_bytes'):
            st.markdown("---")
            st.markdown("### 📥 Download Processed Video")
            st.download_button(
                label="⬇️ Download Video (MP4)",
                data=results['video_bytes'],
                file_name="sds_processed_video.mp4",
                mime="video/mp4"
            )

        _render_video_results(results, config)

        st.markdown("---")
        if st.button("🔄 Clear Results & Upload New Video"):
            st.session_state.video_results = None
            st.rerun()

    # ── File uploader ─────────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Select Video", type=['mp4', 'avi', 'mov', 'mkv'],
        help="Choose a video file to analyze"
    )

    if uploaded_file:
        st.session_state.video_results    = None
        st.session_state.first_frame_data = None

        st.markdown("### ⚙️ Analysis Options")
        col1, col2, col3 = st.columns(3)
        with col1: show_detection = st.checkbox("👤 Detection", value=True,  key="v_det")
        with col2: show_density   = st.checkbox("📊 Density",   value=True,  key="v_dens")
        with col3: show_flow      = st.checkbox("🔄 Flow",      value=False, key="v_flow")

        st.markdown("### ⚙️ Detection Settings")
        st.warning("⚠️ **Seeing empty boxes?** Increase confidence to 0.35–0.45 to remove false positives!")
        conf_threshold  = st.slider("Confidence Threshold", 0.1, 0.9, 0.35, 0.05)
        max_detections  = st.slider("Max Detections per Frame", 10, 400, 100, 10)
        track_specific  = st.checkbox("🎯 Track Specific Person", value=False, key="v_track_person")
        max_frames      = st.slider("Max Frames to Process", 50, 500, 200, step=50)

        selected_person_id = None

        # ── First-frame person selector ───────────────────────────────────────
        if track_specific:
            if 'first_frame_data' not in st.session_state or st.session_state.first_frame_data is None:
                with st.spinner("📸 Loading first frame for person selection..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = tmp.name
                    try:
                        cap = cv2.VideoCapture(tmp_path)
                        ret, first_frame = cap.read()
                        cap.release()
                    finally:
                        os.unlink(tmp_path)

                    if ret:
                        detector, tracker, _ = initialize_modules(config)
                        dets = detector.detect(first_frame)
                        dets = filter_detections_by_confidence(dets, conf_threshold)
                        dets = limit_detections(dets, max_detections)
                        for _ in range(5):
                            tracks = tracker.update(first_frame, dets)
                        confirmed = [t for t in tracks if t.is_confirmed()]
                        display = first_frame.copy()
                        person_ids = []
                        if confirmed:
                            for t in confirmed:
                                x1, y1, x2, y2 = map(int, t.to_tlbr())
                                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                cv2.putText(display, f'ID: {t.track_id}', (x1, y1-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                person_ids.append(t.track_id)
                        else:
                            for i, det in enumerate(dets):
                                x1, y1, x2, y2 = map(int, det[:4])
                                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 3)
                                cv2.putText(display, f'Person {i+1}', (x1, y1-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                person_ids.append(i + 1)
                        st.session_state.first_frame_data = {
                            'frame': cv2.cvtColor(display, cv2.COLOR_BGR2RGB),
                            'person_ids': person_ids,
                            'num_detected': len(dets),
                            'has_tracks': bool(confirmed)
                        }

            fd = st.session_state.get('first_frame_data')
            if fd:
                st.markdown("### 🎯 Select Person to Track")
                st.image(fd['frame'], caption="First Frame — Detected Persons", use_column_width=True)
                if fd['num_detected'] > 0:
                    if fd['has_tracks']:
                        st.info("✅ Using actual tracker IDs")
                    else:
                        st.warning("⚠️ Tracks not yet confirmed — IDs are detection indices")
                    selected_person_id = st.selectbox(
                        "Select Person ID:", options=fd['person_ids'], key="person_id_selector"
                    )
                    st.success(f"✅ Person ID {selected_person_id} selected")
                else:
                    st.warning("⚠️ No persons detected — try lowering the confidence threshold")

        # ── Main processing ───────────────────────────────────────────────────
        if st.button("▶️ Analyze Video", use_container_width=True):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_video_path = tmp.name

            cap = out = output_video_path = None
            try:
                cap = cv2.VideoCapture(tmp_video_path)
                if not cap.isOpened():
                    raise RuntimeError("Failed to open video file.")
                ret_first, first_frame = cap.read()
                if not ret_first:
                    raise RuntimeError("Could not read frames from video.")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                vid_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or first_frame.shape[1]
                vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or first_frame.shape[0]
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                st.success("✅ Video loaded successfully!")
                st.info(f"📹 {vid_width}×{vid_height} | FPS: {fps:.1f} | Frames: {total_frames}")

                detector, tracker, _ = initialize_modules(config)
                density_estimator = DensityEstimator(config['density'], (vid_height, vid_width))

                # Write to temp file
                output_video_path = os.path.join(
                    tempfile.gettempdir(), f"sds_out_{os.getpid()}.mp4"
                )
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (vid_width, vid_height))
                if not out.isOpened():
                    out = None

                progress_bar      = st.progress(0)
                frame_placeholder = st.empty()
                status_placeholder = st.empty()

                processed_frames          = 0
                total_detections          = []
                present_counts            = []
                density_over_time         = []
                unique_track_ids          = set()
                tracked_person_positions  = []
                tracked_person_found      = 0

                if selected_person_id is not None:
                    status_placeholder.info(f"🔍 Looking for Person ID {selected_person_id}...")

                while cap.isOpened() and processed_frames < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    output_frame = frame.copy()
                    dets = detector.detect(frame)
                    dets = filter_detections_by_confidence(dets, conf_threshold)
                    dets = limit_detections(dets, max_detections)
                    total_detections.append(len(dets))

                    tracks = []

                    # Track specific person
                    if selected_person_id is not None and len(dets) > 0:
                        try:
                            tracks = tracker.update(frame, dets)
                            if processed_frames < 10:
                                active = [t.track_id for t in tracks if t.is_confirmed()]
                                if active:
                                    status_placeholder.info(
                                        f"🔍 Frame {processed_frames}: IDs {active} | Looking for {selected_person_id}"
                                    )
                            for t in tracks:
                                if t.is_confirmed() and t.track_id == selected_person_id:
                                    x1, y1, x2, y2 = map(int, t.to_tlbr())
                                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                                    cv2.putText(output_frame, f'TRACKING ID:{t.track_id}',
                                                (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                                    cx, cy = (x1+x2)//2, (y1+y2)//2
                                    cv2.circle(output_frame, (cx, cy), 8, (0,0,255), -1)
                                    tracked_person_positions.append((processed_frames, cx, cy, x2-x1, y2-y1))
                                    tracked_person_found += 1
                                    if tracked_person_found == 1:
                                        status_placeholder.success(f"✅ Found Person ID {selected_person_id}!")
                                    break
                        except Exception:
                            pass

                    # Show all detections
                    elif show_detection and len(dets) > 0:
                        try:
                            tracks = tracker.update(frame, dets)
                            for t in tracks:
                                if t.is_confirmed():
                                    x1, y1, x2, y2 = map(int, t.to_tlbr())
                                    cv2.rectangle(output_frame, (x1,y1),(x2,y2),(0,255,0),2)
                                    cv2.putText(output_frame, f'ID:{t.track_id}', (x1,y1-10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                        except Exception:
                            for det in dets:
                                x1,y1,x2,y2 = map(int, det[:4])
                                cv2.rectangle(output_frame,(x1,y1),(x2,y2),(0,255,0),2)
                                cv2.putText(output_frame,f'{det[4]:.2f}',(x1,y1-5),
                                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

                    confirmed_tracks = [t for t in tracks if t.is_confirmed()] if tracks else []
                    present_counts.append(len(confirmed_tracks) if confirmed_tracks else len(dets))
                    for t in confirmed_tracks:
                        unique_track_ids.add(t.track_id)

                    if show_density and (len(dets) > 0 or confirmed_tracks):
                        try:
                            d_input = confirmed_tracks if confirmed_tracks else dets
                            dg, _, _ = density_estimator.estimate(d_input)
                            dc = dg.sum()
                            density_over_time.append(dc)
                            thr = config['density']['thresholds']
                            if dc >= thr['critical']:   lv, col = "CRITICAL", (0,0,255)
                            elif dc >= thr['high']:     lv, col = "HIGH",     (0,165,255)
                            elif dc >= thr['medium']:   lv, col = "MEDIUM",   (0,255,255)
                            else:                       lv, col = "LOW",      (0,255,0)
                            cv2.putText(output_frame, f'Density: {lv} ({dc:.0f})',
                                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
                        except Exception:
                            density_over_time.append(len(dets))

                    frame_placeholder.image(output_frame, channels="BGR", use_column_width=True)
                    if out:
                        out.write(output_frame)
                    processed_frames += 1
                    if processed_frames % 10 == 0 or processed_frames >= max_frames:
                        progress_bar.progress(min(processed_frames / max_frames, 1.0))

                progress_bar.progress(1.0)
                st.success(f"✅ Done! Analysed {processed_frames} frames")

                # FIX: read video bytes BEFORE closing/deleting file, store in session_state
                video_bytes = None
                if out:
                    out.release()
                    out = None
                if output_video_path and os.path.exists(output_video_path) and \
                        os.path.getsize(output_video_path) > 0:
                    with open(output_video_path, 'rb') as vf:
                        video_bytes = vf.read()
                    st.markdown("---")
                    st.markdown("### 📥 Download Processed Video")
                    st.download_button(
                        label="⬇️ Download Video (MP4)",
                        data=video_bytes,
                        file_name="sds_processed_video.mp4",
                        mime="video/mp4"
                    )

                # Store everything in session_state
                st.session_state.video_results = {
                    'video_bytes':                video_bytes,
                    'tracked_person_positions':   tracked_person_positions,
                    'tracked_person_found_frames': tracked_person_found,
                    'selected_person_id':         selected_person_id,
                    'density_over_time':          density_over_time,
                    'processed_frames':           processed_frames,
                    'total_detections':           total_detections,
                    'present_counts':             present_counts,
                    'unique_ids_count':           len(unique_track_ids),
                    'config':                     config,
                    'show_detection':             show_detection,
                    'show_density':               show_density,
                }

                _render_video_results(st.session_state.video_results, config)

            except Exception as e:
                st.error(f"❌ Error processing video: {e}")
            finally:
                if cap:
                    try: cap.release()
                    except: pass
                if out:
                    try: out.release()
                    except: pass
                if output_video_path and os.path.exists(output_video_path):
                    try: os.unlink(output_video_path)
                    except: pass
                if os.path.exists(tmp_video_path):
                    try: os.unlink(tmp_video_path)
                    except: pass
    else:
        st.info("📤 Upload a video to begin analysis")


def _render_video_results(results, config):
    """Render charts and stats from stored video results."""
    total_detections  = results.get('total_detections', [])
    present_counts    = results.get('present_counts', [])
    density_over_time = results.get('density_over_time', [])
    tracked_positions = results.get('tracked_person_positions', [])
    selected_id       = results.get('selected_person_id')
    tracked_found     = results.get('tracked_person_found_frames', 0)
    processed_frames  = results.get('processed_frames', 1)
    show_detection    = results.get('show_detection', True)
    show_density      = results.get('show_density', True)

    # Tracking summary
    if selected_id is not None and len(tracked_positions) > 0:
        st.markdown("---")
        st.markdown(f"### 🎯 Tracking Results for Person ID {selected_id}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("✅ Frames Tracked", tracked_found)
        col2.metric("📊 Tracking Rate", f"{tracked_found/processed_frames*100:.1f}%")
        col3.metric("📹 Total Frames", processed_frames)
        col4.metric("👤 Person ID", selected_id)

        if len(tracked_positions) > 1:
            df = pd.DataFrame(tracked_positions, columns=['Frame','X','Y','Width','Height'])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['X'], y=df['Y'], mode='lines+markers',
                                     line=dict(color='red', width=2), marker=dict(size=6)))
            fig.update_layout(title='Movement Trajectory', xaxis_title='X (px)',
                              yaxis_title='Y (px)', yaxis=dict(autorange='reversed'), height=400)
            st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                fx = px.line(df, x='Frame', y='X', title='X Position Over Time')
                fx.update_traces(line_color='red'); st.plotly_chart(fx, use_container_width=True)
            with c2:
                fy = px.line(df, x='Frame', y='Y', title='Y Position Over Time')
                fy.update_traces(line_color='blue'); st.plotly_chart(fy, use_container_width=True)

            x_mov = df['X'].max() - df['X'].min()
            y_mov = df['Y'].max() - df['Y'].min()
            disp  = np.sqrt((df['X'].iloc[-1]-df['X'].iloc[0])**2 + (df['Y'].iloc[-1]-df['Y'].iloc[0])**2)
            c1, c2, c3 = st.columns(3)
            c1.metric("↔️ Horizontal Movement", f"{x_mov:.0f} px")
            c2.metric("↕️ Vertical Movement",   f"{y_mov:.0f} px")
            c3.metric("📏 Total Displacement",  f"{disp:.0f} px")
    elif selected_id is not None:
        st.warning(f"⚠️ Person ID {selected_id} was not successfully tracked in the video")

    # People count chart
    if show_detection and len(total_detections) > 0:
        st.markdown("---")
        st.markdown("### 👥 People Count Over Time")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(total_detections))), y=total_detections,
                                 mode='lines', name='Detections (raw)',
                                 line=dict(color='#4CAF50', width=2),
                                 fill='tozeroy', fillcolor='rgba(76,175,80,0.2)'))
        if len(present_counts) == len(total_detections):
            fig.add_trace(go.Scatter(x=list(range(len(present_counts))), y=present_counts,
                                     mode='lines', name='Present (tracked)',
                                     line=dict(color='#2196F3', width=2),
                                     fill='tozeroy', fillcolor='rgba(33,150,243,0.2)'))
        fig.update_layout(title='People Count Per Frame', xaxis_title='Frame',
                          yaxis_title='Count', height=400, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📈 Max Count", max(total_detections))
        c2.metric("📉 Min Count", min(total_detections))
        c3.metric("📊 Avg Count", f"{sum(total_detections)/len(total_detections):.1f}")
        c4.metric("🆔 Unique IDs", results.get('unique_ids_count', 0))

    # Density chart
    if show_density and len(density_over_time) > 0:
        st.markdown("---")
        st.markdown("### 📊 Crowd Density Analysis")
        thr = config['density']['thresholds']
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(density_over_time))), y=density_over_time,
                                 mode='lines', name='Crowd Density',
                                 line=dict(color='#FF6B6B', width=2),
                                 fill='tozeroy', fillcolor='rgba(255,107,107,0.3)'))
        fig.add_hline(y=thr['critical'], line_dash="dash", line_color="red",    annotation_text="Critical")
        fig.add_hline(y=thr['high'],     line_dash="dash", line_color="orange", annotation_text="High")
        fig.add_hline(y=thr['medium'],   line_dash="dash", line_color="yellow", annotation_text="Medium")
        fig.update_layout(title='Crowd Density Over Time', xaxis_title='Frame',
                          yaxis_title='Density Count', height=400, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📈 Max Density",    f"{max(density_over_time):.0f}")
        c2.metric("📉 Min Density",    f"{min(density_over_time):.0f}")
        c3.metric("📊 Avg Density",    f"{sum(density_over_time)/len(density_over_time):.0f}")
        c4.metric("🚨 Critical Frames", sum(1 for d in density_over_time if d >= thr['critical']))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    if "page" not in st.session_state:
        st.session_state.page = "🏠 Home"

    st.sidebar.markdown("# 🎬 SDS Dashboard")
    if st.sidebar.button("🏠 Home",            use_container_width=True, key="nav_home"):
        st.session_state.page = "🏠 Home"
    if st.sidebar.button("🖼️ Image Analysis",  use_container_width=True, key="nav_image"):
        st.session_state.page = "🖼️ Image Analysis"
    if st.sidebar.button("🎥 Video Analysis",  use_container_width=True, key="nav_video"):
        st.session_state.page = "🎥 Video Analysis"

    page = st.session_state.page
    if   page == "🏠 Home":           show_home_page()
    elif page == "🖼️ Image Analysis": show_image_analysis()
    elif page == "🎥 Video Analysis": show_video_analysis()


if __name__ == "__main__":
    main()
