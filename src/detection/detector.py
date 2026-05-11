"""YOLOv8 person detector - cloud safe."""
import os
import cv2
import numpy as np
from ultralytics import YOLO

class PersonDetector:
    def __init__(self, config):
        self.config = config
        # Resolve model path: look next to this file's repo root first
        model_name = config['model']
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        local_path = os.path.join(repo_root, model_name)
        self.model = YOLO(local_path if os.path.exists(local_path) else model_name)
        self.confidence = config['confidence']
        self.iou_threshold = config['iou_threshold']
        self.classes = config['classes']
        self.device = config.get('device', 'cpu')
        self.imgsz = config.get('imgsz', 640)
        self.augment = config.get('augment', False)
        self.max_det = config.get('max_det', 300)
        self.tta_scales = config.get('tta_scales', [1.0])
        self.flip_tta = config.get('flip_tta', False)
        self.nms_iou = config.get('nms_iou', 0.55)

    def detect(self, frame):
        """Detect persons in frame. Returns [x1, y1, x2, y2, conf, cls]."""
        h, w = frame.shape[:2]
        all_dets = []

        for scale in self.tta_scales:
            imgsz = int(self.imgsz * float(scale))
            res = self.model(
                frame,
                conf=self.confidence,
                iou=self.iou_threshold,
                classes=self.classes,
                device=self.device,
                imgsz=imgsz,
                augment=self.augment,
                max_det=self.max_det,
                verbose=False,
            )
            if len(res) > 0:
                for box in res[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    all_dets.append([x1, y1, x2, y2, conf, cls])

            if self.flip_tta:
                flipped = cv2.flip(frame, 1)
                res_f = self.model(
                    flipped, conf=self.confidence, iou=self.iou_threshold,
                    classes=self.classes, device=self.device, imgsz=imgsz,
                    augment=self.augment, max_det=self.max_det, verbose=False,
                )
                if len(res_f) > 0:
                    for box in res_f[0].boxes:
                        fx1, fy1, fx2, fy2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        all_dets.append([w - fx2, fy1, w - fx1, fy2, conf, cls])

        if not all_dets:
            return np.empty((0, 6))

        merged = self._nms(np.array(all_dets, dtype=float), self.nms_iou)
        if merged.shape[0] > self.max_det:
            order = np.argsort(-merged[:, 4])
            merged = merged[order[:self.max_det]]
        return merged

    @staticmethod
    def _nms(dets, iou_thr):
        if dets.size == 0:
            return dets
        x1, y1, x2, y2, scores = dets[:,0], dets[:,1], dets[:,2], dets[:,3], dets[:,4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            iou = (w * h) / (areas[i] + areas[order[1:]] - w * h)
            order = order[np.where(iou <= iou_thr)[0] + 1]
        return dets[keep]

    def __call__(self, frame):
        return self.detect(frame)
