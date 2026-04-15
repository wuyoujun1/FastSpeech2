#!/usr/bin/env python3
# jetson_face_v3.py - 完整修复版 (参考简单版逻辑 + 高级功能)
# 工作目录：/opt/Work/Face_detect

import cv2
import numpy as np
import json
import os
import time
import threading
import signal
import sys
import warnings
from datetime import datetime
from queue import Queue, Empty, Full
from collections import deque
from flask import Flask, Response, request, jsonify, url_for
from PIL import Image, ImageDraw, ImageFont

import insightface
from insightface.app import FaceAnalysis
from insightface.app.common import Face as InsightFace

# ✅ 屏蔽警告
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['ORT_LOG_SEVERITY_LEVEL'] = '3'

# ✅ FFmpeg RTSP 低延迟配置
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"

# ==================== 配置 ====================
CONFIG = {
    "base_dir": "/opt/Work/Face_detect",
    "database_path": "/opt/Work/Face_detect/database/faces.json",
    "rtsp_source": "rtsp://192.168.234.1:8554/test",
    "local_ip": "192.168.234.234",
    "web_port": 5000,
    "stream_width": 1280,#1920,1080
    "stream_height": 720,
    "stream_fps": 25,
    "web_width": 1280,
    "web_height": 720,
    "web_fps": 15,
    "jpeg_quality": 75,
    "use_gstreamer": False,
    "gst_latency": 200,
    "model_name": "buffalo_s",
    "detect_resolution": (960, 960),
    
    # ✅ 修复：降低阈值适应机器狗场景
    "threshold": 0.55,
    "min_face_size": 20,
    
    # 录入配置
    "enroll_frames": 7,
    "enroll_interval": 0.3,
    
    # 追踪配置 (机器狗专用)
    "recognize_interval_frames": 2,
    "iou_threshold": 0.2,
    "track_max_age": 30,
    "track_min_hits": 1,
    "kalman_process_noise": 0.1,
    "kalman_measure_noise": 0.5,
    
    # 队列配置
    "max_task_queue_size": 50,
    "max_result_queue_size": 50,
    
    # 字体配置
    "font_paths": [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ],
    
    "debug_mode": False,
}

COLOR_POOL = [
    (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255),
    (0, 128, 255), (128, 0, 255), (255, 255, 0), (0, 255, 128),
]
name_color_map = {}

FONT_PATH = None
for fp in CONFIG["font_paths"]:
    if os.path.exists(fp):
        FONT_PATH = fp
        break

app = Flask(__name__)
latest_frame = None
frame_lock = threading.Lock()
frame_count = 0
face_app = None
db = None
tracker = None
recognizer = None
enroller = None
img_center_x = CONFIG["stream_width"] // 2
img_center_y = CONFIG["stream_height"] // 2

# ==================== 工具函数 ====================
def cv2_put_chinese_text(img, text, position, color=(0, 255, 0), font_size=20, font_path=None):
    if font_path is None:
        font_path = FONT_PATH
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    if font_path and os.path.exists(font_path):
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=tuple(reversed(color)))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def debug_log(msg):
    if CONFIG["debug_mode"]:
        print(f"    [DEBUG] {msg}")

def calculate_iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = bbox1_area + bbox2_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

# ==================== 数据库管理 ====================
class FaceDatabase:
    def __init__(self, path):
        self.path = path
        self.data = self.load()
        self.lock = threading.Lock()

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ 加载数据库失败：{e}")
        return {}

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            print(f"💾 保存数据库：{len(self.data)} 人")
            return True
        except Exception as e:
            print(f"⚠️ 保存数据库失败：{e}")
            return False

    def enroll(self, name, features):
        """录入人脸特征"""
        with self.lock:
            if name not in self.data:
                self.data[name] = {
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "features": []
                }
            # ✅ 确保特征是 list 格式
            for feat in features:
                if isinstance(feat, np.ndarray):
                    self.data[name]["features"].append(feat.tolist())
                elif isinstance(feat, list):
                    self.data[name]["features"].append(feat)
            self.save()
            count = len(self.data[name]["features"])
            if self.data[name]["features"]:
                feat = np.array(self.data[name]["features"][0])
                print(f"\n✅ 录入成功：{name} (共{count}个特征，维度：{feat.shape})")
            return True

    def delete(self, name):
        """✅ 删除人脸"""
        with self.lock:
            if name in self.data:
                del self.data[name]
                self.save()
                print(f"\n🗑️ 已删除：{name}")
                return True
            print(f"\n⚠️ 未找到：{name}")
            return False

    def list_all(self):
        with self.lock:
            return {
                name: {
                    "feature_count": len(info["features"]),
                    "created_at": info["created_at"]
                }
                for name, info in self.data.items()
            }

    def search(self, embedding, debug=False):
        """搜索最相似人脸"""
        with self.lock:
            best_score, best_name = 0.0, "strangers"
            
            # ✅ 检查输入特征
            if embedding is None:
                return "strangers", 0.0
            if isinstance(embedding, list):
                embedding = np.array(embedding, dtype=np.float32)
            if embedding.size == 0 or not np.any(embedding):
                return "strangers", 0.0
            
            norm_embed = np.linalg.norm(embedding)
            if norm_embed == 0 or np.isnan(norm_embed) or np.isinf(norm_embed):
                return "strangers", 0.0
            
            for name, info in self.data.items():
                for stored_feat in info["features"]:
                    try:
                        if stored_feat is None:
                            continue
                        stored_emb = np.array(stored_feat, dtype=np.float32)
                        if stored_emb.size == 0 or not np.any(stored_emb):
                            continue
                        
                        norm_stored = np.linalg.norm(stored_emb)
                        if norm_stored == 0 or np.isnan(norm_stored) or np.isinf(norm_stored):
                            continue
                        
                        # ✅ 安全计算余弦相似度
                        dot_product = np.dot(embedding, stored_emb)
                        denominator = norm_embed * norm_stored
                        if denominator == 0:
                            continue
                        score = dot_product / denominator
                        score = np.clip(score, -1.0, 1.0)
                        
                        if debug and CONFIG["debug_mode"]:
                            debug_log(f"🔍 比对 {name}: 分数={score:.3f}")
                        
                        if score > best_score:
                            best_score, best_name = score, name
                    except Exception as e:
                        if debug and CONFIG["debug_mode"]:
                            debug_log(f"⚠️ 比对 {name} 异常：{e}")
                        continue
            
            result = best_name if best_score > CONFIG["threshold"] else "strangers"
            if CONFIG["debug_mode"]:
                debug_log(f"🎯 识别结果：{result} (最佳分数：{best_score:.3f})")
            return result, best_score

# ==================== 录入器 ====================
class FaceEnroller:
    def __init__(self, database):
        self.database = database
        self.is_enrolling = False
        self.enroll_name = ""
        self.captured_features = []
        self.last_enroll_time = 0
        self.lock = threading.Lock()
        self.detect_count = 0
        self.valid_count = 0
        self.fail_count = 0
        self.center_distance = 0

    def start_enroll(self, name):
        with self.lock:
            if self.is_enrolling:
                return False, "已有录入任务进行中"
            self.is_enrolling = True
            self.enroll_name = name
            self.captured_features = []
            self.last_enroll_time = 0
            self.valid_count = 0
            self.fail_count = 0
            print(f"\n🎯 开始录入：{name}")
        return True, f"📸 请面对摄像头中心，采集 {CONFIG['enroll_frames']} 帧"

    def capture_frame(self, frame, frame_bbox, detected_faces):
        """✅ 参考简单版：直接使用 face.embedding"""
        with self.lock:
            if not self.is_enrolling:
                return False, "未处于录入状态"
            
            current_time = time.time()
            if current_time - self.last_enroll_time < CONFIG["enroll_interval"]:
                return False, f"⏳ 间隔太短"
            
            if not detected_faces:
                self.fail_count += 1
                return False, f"⏳ 未检测到人脸 (失败{self.fail_count}次)"
            
            # ✅ 选择中心人脸
            min_distance = float('inf')
            best_face = None
            for face in detected_faces:
                face_bbox = face.bbox.astype(int)
                face_cx = (face_bbox[0] + face_bbox[2]) / 2
                face_cy = (face_bbox[1] + face_bbox[3]) / 2
                distance = ((face_cx - img_center_x) ** 2 + (face_cy - img_center_y) ** 2) ** 0.5
                if distance < min_distance:
                    min_distance = distance
                    best_face = face
            
            if best_face is None:
                self.fail_count += 1
                return False, f"⚠️ 无法选择人脸"
            
            self.center_distance = min_distance
            self.detect_count = len(detected_faces)
            
            # ✅ 直接使用 face.embedding (参考简单版)
            if best_face.embedding is None:
                self.fail_count += 1
                return False, f"⚠️ 特征为空"
            
            embedding = np.array(best_face.embedding, dtype=np.float32)
            if embedding.size == 0 or np.all(embedding == 0) or np.isnan(embedding).any():
                self.fail_count += 1
                return False, f"⚠️ 特征无效"
            
            # ✅ 采集成功
            self.captured_features.append(embedding)
            self.last_enroll_time = current_time
            self.valid_count += 1
            
            progress = len(self.captured_features)
            total = CONFIG["enroll_frames"]
            
            if progress >= total:
                features_list = [feat.tolist() for feat in self.captured_features]
                if self.database.enroll(self.enroll_name, features_list):
                    enroll_name = self.enroll_name
                    self.is_enrolling = False
                    self.enroll_name = ""
                    self.captured_features = []
                    return True, f"✅ 录入成功！{enroll_name}"
                else:
                    return False, "❌ 保存失败"
            
            return True, f"📸 {progress}/{total} | 检测:{self.detect_count}人 | 距离:{self.center_distance:.0f}px"

    def cancel_enroll(self):
        with self.lock:
            if self.is_enrolling:
                self.is_enrolling = False
                self.enroll_name = ""
                self.captured_features = []
                print("   ❌ 录入已取消")
                return True
        return False

    def get_status(self):
        with self.lock:
            return {
                "active": self.is_enrolling,
                "name": self.enroll_name,
                "progress": len(self.captured_features),
                "target": CONFIG["enroll_frames"],
                "detect_count": self.detect_count,
                "valid_count": self.valid_count,
                "fail_count": self.fail_count,
                "center_distance": self.center_distance
            }

# ==================== 卡尔曼追踪器 ====================
class KalmanTracker:
    def __init__(self, track_id, bbox, kps=None, embedding=None):
        self.track_id = track_id
        self.age = 0
        self.hits = 0
        self.time_since_update = 0
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ], np.float32)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ], np.float32)
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * CONFIG["kalman_process_noise"]
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * CONFIG["kalman_measure_noise"]
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        self.kf.statePost[:4, 0] = np.array([cx, cy, w, h], np.float32)
        self.bbox = bbox
        self.kps = kps
        self.embedding = embedding
        self.is_locked = False
        self.name = "未确定"
        self.score = 0.0
        self.color = (0, 165, 255)
        self.confirmed = False
        self.last_recognize_frame = 0
        self.is_pending = False

    def predict(self, current_frame):
        self.age += 1
        self.time_since_update += 1
        if self.is_locked:
            self.kf.predict()
            cx, cy, w, h = self.kf.statePre[:4, 0]
            x1, y1 = int(cx - w / 2), int(cy - h / 2)
            x2, y2 = int(cx + w / 2), int(cy + h / 2)
            self.predicted_bbox = np.array([max(0, x1), max(0, y1), x2, y2])
        else:
            self.predicted_bbox = self.bbox.astype(np.int32)
        return self.predicted_bbox

    def update(self, bbox, kps=None, embedding=None):
        self.hits += 1
        self.time_since_update = 0
        if self.is_locked:
            cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            measurement = np.array([[cx], [cy], [w], [h]], np.float32)
            self.kf.correct(measurement)
        self.bbox = bbox
        if kps is not None:
            self.kps = kps
        if embedding is not None:
            self.embedding = embedding

    def lock_identity(self, name, score):
        self.is_locked = True
        self.name = name
        self.score = score
        self.color = self._get_color(name)
        self.confirmed = True
        cx, cy = (self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2
        w, h = self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]
        self.kf.statePost[:4, 0] = np.array([cx, cy, w, h], np.float32)
        debug_log(f"🔒 Track-{self.track_id}: 已锁定为 {name} ({score:.2f})")

    def should_submit_recognition(self, current_frame):
        if self.is_locked:
            return False
        if self.is_pending:
            return False
        face_w = self.bbox[2] - self.bbox[0]
        if face_w < CONFIG["min_face_size"]:
            return False
        if current_frame - self.last_recognize_frame < CONFIG["recognize_interval_frames"]:
            return False
        if self.hits < CONFIG["track_min_hits"]:
            return False
        return True

    def mark_pending(self, current_frame):
        self.is_pending = True
        self.last_recognize_frame = current_frame

    def release_pending(self):
        self.is_pending = False

    def mark_submit_failed(self):
        self.is_pending = False

    def is_dead(self):
        return self.time_since_update > CONFIG["track_max_age"]

    def _get_color(self, name):
        if name not in name_color_map:
            idx = len(name_color_map) % len(COLOR_POOL)
            name_color_map[name] = COLOR_POOL[idx]
        return name_color_map[name]

# ==================== 人脸追踪管理 ====================
class FaceTracker:
    def __init__(self):
        self.trackers = {}
        self.next_track_id = 1

    def update(self, bboxes, kpss, det_scores, current_frame):
        n = len(bboxes) if bboxes is not None else 0
        for t in self.trackers.values():
            t.predict(current_frame)
        if n > 0 and self.trackers:
            track_ids = list(self.trackers.keys())
            matched_tracks, matched_dets = set(), set()
            iou_matrix = np.zeros((len(track_ids), n))
            for i, tid in enumerate(track_ids):
                for j in range(n):
                    iou_matrix[i, j] = calculate_iou(self.trackers[tid].bbox, bboxes[j])
            while iou_matrix.size > 0:
                max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                if iou_matrix[max_idx] < CONFIG["iou_threshold"]:
                    break
                i, j = max_idx
                tid = track_ids[i]
                kps = kpss[j] if kpss is not None and j < len(kpss) else None
                self.trackers[tid].update(bboxes[j], kps=kps)
                matched_tracks.add(tid)
                matched_dets.add(j)
                iou_matrix[i, :] = -1
                iou_matrix[:, j] = -1
            for j in range(n):
                if j not in matched_dets:
                    new_t = KalmanTracker(self.next_track_id, bboxes[j], 
                                        kpss[j] if kpss is not None and j < len(kpss) else None)
                    self.next_track_id += 1
                    self.trackers[new_t.track_id] = new_t
            for tid in track_ids:
                if tid not in matched_tracks:
                    self.trackers[tid].time_since_update += 1
        elif n > 0:
            for j in range(n):
                new_t = KalmanTracker(self.next_track_id, bboxes[j],
                                    kpss[j] if kpss is not None and j < len(kpss) else None)
                self.next_track_id += 1
                self.trackers[new_t.track_id] = new_t
        for tid in [t for t, tr in self.trackers.items() if tr.is_dead()]:
            del self.trackers[tid]
        return list(self.trackers.keys())

    def has_unconfirmed(self):
        return any(not t.is_locked for t in self.trackers.values())

    def get_unconfirmed_pending(self, current_frame):
        return [t for t in self.trackers.values() if t.should_submit_recognition(current_frame)]

    def process_recognition_result(self, tid, name, score, current_frame):
        if tid in self.trackers:
            t = self.trackers[tid]
            if t.is_locked:
                return False
            if score > CONFIG["threshold"] and name != "strangers":
                t.lock_identity(name, score)
                return True
            t.name = "未确定"
            t.score = score
            t.color = (0, 165, 255)
        return False

    def update_stats(self):
        stats = {"registered": 0, "unrecognized": 0}
        for t in self.trackers.values():
            if t.is_locked:
                stats["registered"] += 1
            else:
                stats["unrecognized"] += 1
        return stats

# ==================== 异步识别器 (✅ 修复核心) ====================
class AsyncRecognizer:
    def __init__(self, app, db):
        self.app = app  # ✅ face_app 用于自动特征提取
        self.db = db
        self.task_queue = Queue(maxsize=CONFIG["max_task_queue_size"])
        self.result_queue = Queue(maxsize=CONFIG["max_result_queue_size"])
        self.running = True
        self.pending_tids = set()
        self.lock = threading.Lock()
        self.cancelled_tids = set()
        self.cancel_lock = threading.Lock()
        
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        print("   ✅ 异步识别线程已启动")

    def _worker(self):
        """✅ 修复版：参考简单版，使用 face_app.get() 自动提取特征"""
        while self.running:
            try:
                task = self.task_queue.get(timeout=0.1)
            except Empty:
                continue
            
            # ✅ 接收原始图像 + 目标 bbox
            tid, original_frame, target_bbox, target_kps, det_score = task
            
            with self.cancel_lock:
                if tid in self.cancelled_tids:
                    self.cancelled_tids.discard(tid)
                    with self.lock:
                        self.pending_tids.discard(tid)
                    continue
            
            try:
                # ✅ 关键修复：使用 face_app.get() 自动检测 + 提取 (参考简单版)
                faces = self.app.get(original_frame)
                
                # ✅ 通过 IOU 找到与目标 bbox 最匹配的人脸
                best_iou = 0
                best_embedding = None
                
                for face in faces:
                    iou = calculate_iou(target_bbox, face.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_embedding = face.embedding
                
                # ✅ 检查特征有效性
                if best_embedding is None:
                    debug_log(f"⚠️ Track-{tid}: 未找到匹配人脸 (IOU={best_iou:.2f})")
                    self.result_queue.put_nowait((tid, "strangers", 0.0))
                    continue
                
                embedding = np.array(best_embedding, dtype=np.float32)
                if embedding.size == 0 or np.all(embedding == 0) or np.isnan(embedding).any():
                    debug_log(f"⚠️ Track-{tid}: 特征无效")
                    self.result_queue.put_nowait((tid, "strangers", 0.0))
                    continue
                
                # ✅ 调试日志
                if CONFIG["debug_mode"]:
                    debug_log(f"🔍 Track-{tid}: 特征维度={embedding.shape}, 范数={np.linalg.norm(embedding):.3f}, IOU={best_iou:.2f}")
                
                # ✅ 执行搜索
                name, score = self.db.search(embedding, debug=True)
                self.result_queue.put_nowait((tid, name, score))
                
            except Exception as e:
                debug_log(f"⚠️ Track-{tid} 识别异常：{e}")
                try:
                    self.result_queue.put_nowait((tid, "strangers", 0.0))
                except Full:
                    pass

    def submit(self, tid, frame, bbox, kps, det_score):
        """✅ 修复：传入原始图像，不裁剪"""
        with self.lock:
            if tid in self.pending_tids:
                return False, "已在排队"
            if self.task_queue.full():
                return False, "队列已满"
            with self.cancel_lock:
                self.cancelled_tids.discard(tid)
            self.pending_tids.add(tid)
        try:
            # ✅ 关键：传入原始 frame，让 _worker 用 face_app.get() 处理
            self.task_queue.put((tid, frame, bbox, kps, float(det_score)), timeout=0.1)
            return True, "提交成功"
        except Full:
            with self.lock:
                self.pending_tids.discard(tid)
            return False, "提交超时"

    def cancel_task(self, tid):
        with self.cancel_lock:
            self.cancelled_tids.add(tid)
        return True

    def cleanup_dead_tasks(self, active_tids, trackers):
        with self.lock:
            for tid in list(self.pending_tids):
                if tid not in active_tids:
                    self.cancel_task(tid)
                    if tid in trackers:
                        trackers[tid].release_pending()

    def collect_results(self):
        results = []
        while True:
            try:
                results.append(self.result_queue.get_nowait())
            except Empty:
                break
        return results

    def get_pending_count(self):
        with self.lock:
            return len(self.pending_tids)

    def release_pending(self, tid, trackers):
        with self.lock:
            self.pending_tids.discard(tid)
        if tid in trackers:
            trackers[tid].release_pending()

    def clear_all_tasks(self):
        with self.lock:
            count = len(self.pending_tids)
            self.pending_tids.clear()
        with self.cancel_lock:
            self.cancelled_tids.clear()
        while True:
            try:
                self.task_queue.get_nowait()
            except Empty:
                break
        while True:
            try:
                self.result_queue.get_nowait()
            except Empty:
                break
        debug_log(f"🧹 队列已清空 ({count} 任务)")

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)

# ==================== GStreamer 管道 ====================
def create_gstreamer_pipeline(rtsp_url, width, height, fps):
    return (
        f"rtspsrc location={rtsp_url} latency={CONFIG['gst_latency']} "
        f"buffer-size=524288 drop-on-latency=true ! rtph264depay ! h264parse ! "
        f"nvv4l2decoder enable-max-performance=1 ! nvvidconv ! "
        f"video/x-raw,format=BGRx,width={width},height={height} ! videoconvert ! "
        f"video/x-raw,format=BGR ! appsink drop=1 sync=false emit-signals=false max-buffers=1"
    )

# ==================== 无缓存实时视频流 ====================
class RealTimeCamera:
    def __init__(self, rtsp_url, use_gstreamer, width, height, fps):
        self.rtsp_url = rtsp_url
        self.use_gstreamer = use_gstreamer
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = self._create_cap()
        self.ret = False
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        
    def _create_cap(self):
        if self.use_gstreamer:
            pipeline = create_gstreamer_pipeline(self.rtsp_url, self.width, self.height, self.fps)
            return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap

    def _reader(self):
        while self.running:
            if not self.cap.isOpened():
                time.sleep(0.1)
                continue
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
            with self.lock:
                self.ret = ret
                if ret:
                    self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is not None:
                return True, self.frame.copy()
            return False, None
            
    def isOpened(self):
        return self.cap.isOpened()
        
    def release(self):
        self.running = False
        self.thread.join(timeout=1)
        self.cap.release()

# ==================== 视频流读取线程 ====================
def video_reader_thread():
    global latest_frame, frame_count, face_app, tracker, recognizer, enroller, db
    
    print(f"📡 启动 RTSP 拉流：{CONFIG['rtsp_source']}")
    print(f"⏳ 加载人脸模型：{CONFIG['model_name']} (CUDA)...")
    face_app = FaceAnalysis(name=CONFIG["model_name"], 
                          providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=CONFIG["detect_resolution"])
    print("✅ 人脸模型加载完成")
    
    db = FaceDatabase(CONFIG["database_path"])
    tracker = FaceTracker()
    recognizer = AsyncRecognizer(face_app, db)
    enroller = FaceEnroller(db)
    
    cap = RealTimeCamera(CONFIG["rtsp_source"], CONFIG["use_gstreamer"], 
                         CONFIG["stream_width"], CONFIG["stream_height"], 
                         CONFIG["stream_fps"])
    
    if not cap.isOpened():
        print("❌ 无法打开 RTSP 流")
        return
    print("✅ RTSP 连接成功")
    
    local_count = 0
    last_log = time.time()
    last_frame_time = time.time()
    
    while True:
        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                if time.time() - last_frame_time > 3.0:
                    print("⚠️ 拉流超时，重新连接...")
                    cap.release()
                    cap = RealTimeCamera(CONFIG["rtsp_source"], CONFIG["use_gstreamer"], 
                                         CONFIG["stream_width"], CONFIG["stream_height"], 
                                         CONFIG["stream_fps"])
                    last_frame_time = time.time()
                time.sleep(0.01)
                continue
            
            last_frame_time = time.time()
            frame_count += 1
            current_time = time.time()
            
            # 1. 收集识别结果
            for tid, name, score in recognizer.collect_results():
                if tid in tracker.trackers and not tracker.trackers[tid].is_locked:
                    tracker.process_recognition_result(tid, name, score, frame_count)
                    recognizer.release_pending(tid, tracker.trackers)
            
            # 2. 人脸检测
            bboxes, kpss = face_app.det_model.detect(frame, max_num=0, metric='default')
            n = len(bboxes) if bboxes is not None else 0
            det_scores = bboxes[:, 4] if n > 0 else None
            bbox_coords = bboxes[:, 0:4] if n > 0 else np.empty((0, 4))
            
            # 3. 更新追踪器
            track_ids = tracker.update(bbox_coords, kpss, det_scores, frame_count)
            
            # 4. 清理无效任务
            recognizer.cleanup_dead_tasks(set(tracker.trackers.keys()), tracker.trackers)
            
            # 5. 队列清空检查
            if not tracker.has_unconfirmed():
                recognizer.clear_all_tasks()
            else:
                for t in tracker.get_unconfirmed_pending(frame_count):
                    best_iou, best_idx = 0, -1
                    for j in range(n):
                        iou = calculate_iou(t.bbox, bbox_coords[j])
                        if iou > best_iou:
                            best_iou, best_idx = iou, j
                    if best_idx >= 0:
                        success, _ = recognizer.submit(
                            t.track_id, frame, bbox_coords[best_idx],
                            kpss[best_idx] if kpss is not None else None,
                            det_scores[best_idx] if det_scores is not None else 1.0
                        )
                        if success:
                            t.mark_pending(frame_count)
                    else:
                        t.mark_submit_failed()
            
            # 6. 录入模式 (参考简单版逻辑)
            if enroller.is_enrolling and n > 0:
                # ✅ 使用 face_app.get() 获取人脸对象列表 (与识别一致)
                faces = face_app.get(frame)
                if faces:
                    min_dist = float('inf')
                    ref_bbox = None
                    for bbox in bbox_coords:
                        cx, cy = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
                        dist = ((cx-img_center_x)**2 + (cy-img_center_y)**2)**0.5
                        if dist < min_dist:
                            min_dist, ref_bbox = dist, bbox
                    success, msg = enroller.capture_frame(frame, ref_bbox, faces)
                    if not success and "间隔太短" not in msg and CONFIG["debug_mode"]:
                        debug_log(f"录入：{msg}")
            
            # 7. 渲染绘制
            for tid in track_ids:
                t = tracker.trackers[tid]
                bbox = t.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                is_small = (x2-x1) < CONFIG["min_face_size"]
                color = (0,0,255) if is_small else t.color
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                if is_small:
                    label = "太远"
                elif t.is_locked:
                    label = f"{t.name} 🔒"
                else:
                    label = f"未确定 ({t.score:.2f})"
                if FONT_PATH:
                    frame = cv2_put_chinese_text(frame, label, (x1, y1-25), color=color, font_size=20)
                else:
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # 8. 录入视觉反馈
            if enroller.is_enrolling:
                cv2.line(frame, (img_center_x-30, img_center_y), (img_center_x+30, img_center_y), (0,255,0), 2)
                cv2.line(frame, (img_center_x, img_center_y-30), (img_center_x, img_center_y+30), (0,255,0), 2)
                if n > 0:
                    min_d, best_b = float('inf'), None
                    for bbox in bbox_coords:
                        cx, cy = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
                        d = ((cx-img_center_x)**2 + (cy-img_center_y)**2)**0.5
                        if d < min_d:
                            min_d, best_b = d, bbox
                    if best_b is not None:
                        cv2.rectangle(frame, (int(best_b[0]), int(best_b[1])), 
                                    (int(best_b[2]), int(best_b[3])), (0,255,0), 3)
                status = enroller.get_status()
                debug_lines = [
                    f"📸 录入：{enroller.enroll_name}",
                    f"{status['progress']}/{status['target']} | 检测:{status['detect_count']}人",
                    f"有效:{status['valid_count']} | 失败:{status['fail_count']}",
                    f"距离中心:{status['center_distance']:.0f}px"
                ]
                for i, line in enumerate(debug_lines):
                    cv2_put_chinese_text(frame, line, (50, 50+i*30), 
                                       color=(0,255,255), font_size=22)
            
            # 9. 更新全局帧
            with frame_lock:
                latest_frame = frame.copy()
                local_count += 1
            
            # 10. 日志
            if current_time - last_log >= 10:
                fps = local_count / (current_time - last_log)
                stats = tracker.update_stats()
                print(f"📊 FPS:{fps:.1f} | 追踪:{len(tracker.trackers)} | "
                      f"锁定:{stats['registered']} | 未确定:{stats['unrecognized']} | "
                      f"队列:{recognizer.get_pending_count()} | 数据库:{len(db.data)}人")
                local_count = 0
                last_log = current_time
                
        except Exception as e:
            print(f"⚠️ 读取线程异常：{e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)

# ==================== MJPEG 流生成器 ====================
def generate_frames():
    global latest_frame
    try:
        while True:
            try:
                with frame_lock:
                    if latest_frame is not None:
                        if latest_frame.shape[:2] != (CONFIG["web_height"], CONFIG["web_width"]):
                            scaled = cv2.resize(latest_frame, 
                                              (CONFIG["web_width"], CONFIG["web_height"]),
                                              interpolation=cv2.INTER_AREA)
                        else:
                            scaled = latest_frame
                        ret, jpeg = cv2.imencode('.jpg', scaled, 
                                               [int(cv2.IMWRITE_JPEG_QUALITY), CONFIG["jpeg_quality"]])
                        if ret:
                            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + 
                                  jpeg.tobytes() + b'\r\n')
                time.sleep(1.0 / CONFIG["web_fps"])
            except GeneratorExit:
                debug_log("🔌 视频流生成器被客户端关闭")
                break
            except Exception as e:
                debug_log(f"⚠️ 推流异常：{e}")
                time.sleep(0.1)
    finally:
        debug_log("🛑 视频流生成器已退出")

# ==================== HTML 页面 ====================
HTML_TEMPLATE = """
<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>🤖 机器狗人脸系统</title><style>
*{{margin:0;padding:0;box-sizing:border-box}}body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;
background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);min-height:100vh;padding:20px}}
.container{{max-width:900px;margin:0 auto}}.header{{text-align:center;color:white;margin-bottom:20px}}
.header h1{{font-size:28px;margin-bottom:10px}}.video-container{{background:white;border-radius:15px;
padding:20px;box-shadow:0 10px 30px rgba(0,0,0,0.3);margin-bottom:20px}}
.video-wrapper{{position:relative;width:100%;background:#000;border-radius:10px;overflow:hidden}}
#videoStream{{width:100%;height:auto;display:block}}.status{{display:flex;justify-content:space-between;
align-items:center;margin-top:15px;padding:10px;background:#f8f9fa;border-radius:8px;flex-wrap:wrap;gap:10px}}
.status-item{{display:flex;align-items:center;font-size:14px;color:#333}}.legend{{margin-top:10px;
display:flex;gap:20px;justify-content:center}}.legend-item{{display:flex;align-items:center;font-size:13px}}
.legend-box{{width:20px;height:20px;margin-right:8px;border-radius:3px}}.legend-box.green{{background:#00ff00}}
.legend-box.orange{{background:#ffa500}}.enroll-panel{{background:white;border-radius:15px;padding:20px;
box-shadow:0 10px 30px rgba(0,0,0,0.3);margin-bottom:20px}}.enroll-panel h2{{color:#667eea;margin-bottom:15px}}
.form-group{{margin-bottom:15px}}.form-group label{{display:block;margin-bottom:5px;font-weight:bold}}
.form-group input{{width:100%;padding:10px;border:2px solid #ddd;border-radius:5px;font-size:16px}}
.form-group input:focus{{border-color:#667eea;outline:none}}.btn{{padding:12px 24px;border:none;
border-radius:5px;font-size:16px;cursor:pointer;margin-right:10px}}.btn-primary{{background:#667eea;color:white}}
.btn-primary:hover{{background:#5568d3}}.btn-success{{background:#28a745;color:white}}.btn-success:hover{{background:#218838}}
.btn-danger{{background:#dc3545;color:white}}.btn-danger:hover{{background:#c82333}}.btn-sm{{padding:6px 12px;font-size:12px}}
.enroll-status{{margin-top:15px;padding:15px;border-radius:5px;font-size:14px}}.enroll-status.waiting{{background:#fff3cd;color:#856404}}
.enroll-status.collecting{{background:#d1ecf1;color:#0c5460}}.enroll-status.success{{background:#d4edda;color:#155724}}
.enroll-status.error{{background:#f8d7da;color:#721c24}}.progress-bar{{width:100%;height:20px;background:#e9ecef;
border-radius:10px;overflow:hidden;margin-top:10px}}.progress-fill{{height:100%;background:linear-gradient(90deg,#667eea,#764ba2);
transition:width 0.3s}}.db-info{{margin-top:15px;padding:15px;background:#f8f9fa;border-radius:5px;font-size:14px}}
.db-info h3{{margin-bottom:10px;color:#667eea}}.db-list{{max-height:200px;overflow-y:auto}}.db-item{{padding:8px;
border-bottom:1px solid #ddd;display:flex;justify-content:space-between;align-items:center}}.db-item:last-child{{border-bottom:none}}
.ip-highlight{{background:#28a745;color:white;padding:2px 8px;border-radius:3px;font-weight:bold}}
</style></head><body><div class="container"><div class="header">
<h1>🤖 机器狗实时视频</h1><p>访问地址：<span class="ip-highlight">http://{LOCAL_IP}:{WEB_PORT}</span></p></div>
<div class="video-container"><div class="video-wrapper">
<img id="videoStream" src="{VIDEO_FEED_URL}" alt="视频流加载中..."></div>
<div class="legend"><div class="legend-item"><div class="legend-box green"></div><span>🔒 已锁定</span></div>
<div class="legend-item"><div class="legend-box orange"></div><span>❓ 未确定</span></div></div>
<div class="status"><div class="status-item"><span>分辨率：<strong>{WEB_WIDTH}x{WEB_HEIGHT}</strong></span></div>
<div class="status-item"><span id="dbCount">数据库：0 人</span></div></div></div>
<div class="enroll-panel"><h2>📸 人脸录入</h2><div class="form-group">
<label for="enrollName">姓名：</label><input type="text" id="enrollName" placeholder="输入要录入的姓名"></div>
<div><button class="btn btn-primary" onclick="startEnroll()">开始录入</button>
<button class="btn btn-danger" onclick="cancelEnroll()">取消</button></div>
<div id="enrollStatus" class="enroll-status waiting" style="display:none;">
<div id="enrollMessage">准备就绪</div><div class="progress-bar"><div id="progressFill" class="progress-fill" style="width:0%"></div></div></div>
<div class="db-info"><h3>📋 已录入人员</h3>
<button class="btn btn-success" onclick="refreshDB()" style="margin-bottom:10px">刷新列表</button>
<div id="dbList" class="db-list">加载中...</div></div></div></div>
<script>
const img=document.getElementById('videoStream'),enrollStatus=document.getElementById('enrollStatus'),
enrollMessage=document.getElementById('enrollMessage'),progressFill=document.getElementById('progressFill'),
dbList=document.getElementById('dbList'),dbCount=document.getElementById('dbCount');let enrollTimer=null;
img.onload=function(){{console.log('视频流已连接')}};img.onerror=function(){{
console.log('视频流断开，5 秒后重试...');setTimeout(()=>{{img.src=img.src+'?t='+Date.now()}},5000)}};
async function startEnroll(){{const name=document.getElementById('enrollName').value.trim();
if(!name){{alert('请输入姓名！');return}}try{{const resp=await fetch('/api/enroll/start',{{
method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{name:name}})}}),
data=await resp.json();if(data.success){{enrollStatus.style.display='block';enrollStatus.className='enroll-status collecting';
enrollMessage.textContent=data.message;progressFill.style.width='0%';enrollTimer=setInterval(checkEnrollStatus,500)}}
else{{alert(data.message)}}}}catch(e){{alert('请求失败：'+e)}}}}
async function checkEnrollStatus(){{try{{const resp=await fetch('/api/enroll/status'),data=await resp.json();
enrollMessage.textContent=data.message;progressFill.style.width=(data.progress/data.target)*100+'%';
if(!data.active){{clearInterval(enrollTimer);enrollStatus.className=data.message.includes('成功')?'enroll-status success':'enroll-status error';
if(data.message.includes('成功'))refreshDB()}}}}catch(e){{console.error(e)}}}}
async function cancelEnroll(){{try{{await fetch('/api/enroll/cancel',{{method:'POST'}});
clearInterval(enrollTimer);enrollStatus.style.display='none';document.getElementById('enrollName').value=''}}
catch(e){{console.error(e)}}}}async function refreshDB(){{try{{const resp=await fetch('/api/database'),
data=await resp.json();dbCount.textContent='数据库：'+data.total_people+' 人';
if(data.total_people===0)dbList.innerHTML='<div style="color:#999">暂无录入人员</div>';
else dbList.innerHTML=data.names.map(name=>{{const info=data.details[name];
return`<div class="db-item"><span>👤 ${{name}} (${{info.feature_count}}特征)</span>
<button class="btn btn-danger btn-sm" onclick="deleteFace('${{name}}')">删除</button></div>`}}).join('')}}
catch(e){{dbList.innerHTML='<div style="color:red">加载失败</div>'}}}}
async function deleteFace(name){{if(!confirm(`确定删除 "${{name}}" 吗？`))return;
try{{const resp=await fetch('/api/database/delete',{{method:'POST',headers:{{'Content-Type':'application/json'}},
body:JSON.stringify({{name:name}})}}),data=await resp.json();
if(data.success){{alert(data.message);refreshDB()}}else{{alert('删除失败：'+data.message)}}}}
catch(e){{alert('请求失败：'+e)}}}}
refreshDB();setInterval(refreshDB,10000);
</script></body></html>
"""

def render_html():
    return HTML_TEMPLATE.format(
        LOCAL_IP=CONFIG["local_ip"],
        WEB_PORT=CONFIG["web_port"],
        WEB_WIDTH=CONFIG["web_width"],
        WEB_HEIGHT=CONFIG["web_height"],
        VIDEO_FEED_URL=url_for('video_feed')
    )

# ==================== Flask 路由 ====================
@app.route('/')
def index():
    return render_html()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    stats = tracker.update_stats() if tracker else {"registered": 0, "unrecognized": 0}
    return jsonify({
        'status': 'online',
        'local_ip': CONFIG["local_ip"],
        'port': CONFIG["web_port"],
        'frame_count': frame_count,
        'database_size': len(db.data) if db else 0,
        'tracking': {
            'total': len(tracker.trackers) if tracker else 0,
            'registered': stats['registered'],
            'unrecognized': stats['unrecognized']
        }
    })

@app.route('/api/enroll/start', methods=['POST'])
def api_enroll_start():
    data = request.json
    name = data.get('name', '').strip()
    if not name:
        return jsonify({'success': False, 'message': '姓名不能为空'})
    success, msg = enroller.start_enroll(name)
    return jsonify({'success': success, 'message': msg})

@app.route('/api/enroll/status', methods=['GET'])
def api_enroll_status():
    return jsonify(enroller.get_status())

@app.route('/api/enroll/cancel', methods=['POST'])
def api_enroll_cancel():
    enroller.cancel_enroll()
    return jsonify({'success': True})

@app.route('/api/database', methods=['GET'])
def api_database():
    data = db.list_all() if db else {}
    return jsonify({
        'total_people': len(data),
        'names': list(data.keys()),
        'details': data
    })

@app.route('/api/database/reload', methods=['POST'])
def api_database_reload():
    if db:
        db.data = db.load()
        return jsonify({'success': True, 'message': '数据库已重新加载'})
    return jsonify({'success': False, 'message': '数据库未初始化'})

@app.route('/api/database/delete', methods=['POST'])
def api_database_delete():
    if not db:
        return jsonify({'success': False, 'message': '数据库未初始化'})
    data = request.json
    name = data.get('name', '').strip()
    if not name:
        return jsonify({'success': False, 'message': '姓名不能为空'})
    if db.delete(name):
        return jsonify({'success': True, 'message': f'已删除：{name}'})
    return jsonify({'success': False, 'message': f'未找到：{name}'})

# ==================== 主函数 ====================
def main():
    print("=" * 70)
    print("🚀 机器狗人脸系统启动 (完整修复版)")
    print("=" * 70)
    print(f"📁 工作目录：{CONFIG['base_dir']}")
    print(f"📂 数据库：{CONFIG['database_path']}")
    print(f"✅ IP: {CONFIG['local_ip']}:{CONFIG['web_port']}")
    print(f"📡 RTSP: {CONFIG['rtsp_source']}")
    print(f"🤖 模型：{CONFIG['model_name']} (det:{CONFIG['detect_resolution']})")
    print(f"🎯 锁定阈值：{CONFIG['threshold']}")
    print(f"⏱️ 识别间隔：每 {CONFIG['recognize_interval_frames']} 帧")
    print(f"📸 录入：{CONFIG['enroll_frames']} 帧 / {CONFIG['enroll_interval']}s 间隔")
    print(f"🐛 调试：{'✅' if CONFIG['debug_mode'] else '❌'}")
    print("=" * 70)
    print(f"\n📱 访问：http://{CONFIG['local_ip']}:{CONFIG['web_port']}")
    print("\n💡 修复说明：")
    print("   • ✅ 特征提取参考简单版：face_app.get() + IOU 匹配")
    print("   • ✅ 传入原始图像，不裁剪，避免坐标不匹配")
    print("   • ✅ 添加删除人脸功能 (Web + API)")
    print("   • ✅ 修复生成器 BrokenPipeError")
    print("=" * 70)
    
    reader = threading.Thread(target=video_reader_thread, daemon=True)
    reader.start()
    time.sleep(2)
    app.run(host='0.0.0.0', port=CONFIG["web_port"], threaded=True)

if __name__ == "__main__":
    main()
