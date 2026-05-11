"""DeepSORT multi-person tracker - cloud safe (no GPU embedder)."""
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

class PersonTracker:
    def __init__(self, config):
        self.config = config
        # use_appearance=True requires GPU embedder which crashes on Streamlit Cloud
        # Force it off; set embedder to None for CPU-safe operation
        use_appearance = config.get('use_appearance', False)
        self.tracker = DeepSort(
            max_age=config['max_age'],
            n_init=config['n_init'],
            max_iou_distance=config['max_iou_distance'],
            max_cosine_distance=config['max_cosine_distance'],
            nn_budget=config['nn_budget'],
            embedder="mobilenet" if use_appearance else None,
            embedder_gpu=False,   # always False - no GPU on Streamlit Cloud
        )
        self.track_history = {}

    def update(self, frame, detections):
        det_list = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            det_list.append(([x1, y1, x2 - x1, y2 - y1], conf, int(cls)))
        tracks = self.tracker.update_tracks(det_list, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            bbox = track.to_ltrb()
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            self.track_history.setdefault(tid, []).append(center)
            if len(self.track_history[tid]) > 30:
                self.track_history[tid].pop(0)
        return tracks

    def __call__(self, frame, detections):
        return self.update(frame, detections)
