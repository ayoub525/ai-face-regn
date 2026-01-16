"""
ULTRA-FAST Face Recognition
===========================
Maximum FPS with background processing.
Press 'q' to quit.
"""

import cv2
import numpy as np
import pickle
from pathlib import Path
import onnxruntime as ort
import threading
import time
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).parent
RECOGNITION_DB_PATH = SCRIPT_DIR / 'recognition_database_arcface.pkl'

# Use the ArcFace model directly with ONNX Runtime (faster than InsightFace wrapper)
MODEL_PATH = Path.home() / '.insightface' / 'models' / 'buffalo_l' / 'w600k_r50.onnx'

print("🚀 Loading models...")

# Face detector - use OpenCV's fast DNN detector
detector_proto = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(detector_proto)

# ArcFace recognizer via ONNX Runtime
if MODEL_PATH.exists():
    arcface = ort.InferenceSession(str(MODEL_PATH), providers=['CPUExecutionProvider'])
    print("✅ ArcFace model loaded")
else:
    print(f"❌ Model not found: {MODEL_PATH}")
    arcface = None

# Database
face_db = {}

# Shared state for threading
current_faces = []
processing_lock = threading.Lock()
is_processing = False


def load_database():
    global face_db
    face_db = {}
    
    if RECOGNITION_DB_PATH.exists():
        with open(RECOGNITION_DB_PATH, 'rb') as f:
            db = pickle.load(f)
        
        for key, data in db['persons'].items():
            parts = key.split('_', 1)
            if len(parts) >= 2:
                face_db[key] = {
                    'id': int(parts[0]),
                    'name': parts[1].replace('_', ' '),
                    'embedding': data['mean_embedding'].flatten()
                }
        print(f"✅ Database: {len(face_db)} persons")


def get_embedding(face_img):
    """Get 512D embedding from face image"""
    if arcface is None:
        return None
    
    # Preprocess for ArcFace
    face = cv2.resize(face_img, (112, 112))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = (face.astype(np.float32) - 127.5) / 127.5
    face = np.transpose(face, (2, 0, 1))
    face = np.expand_dims(face, 0)
    
    # Get embedding
    embedding = arcface.run(None, {'input.1': face})[0].flatten()
    return embedding


def match_face(embedding):
    """Find best match in database"""
    if embedding is None or not face_db:
        return None, 0.0
    
    query = embedding / (np.linalg.norm(embedding) + 1e-6)
    
    best_match, best_sim = None, 0.0
    for data in face_db.values():
        db_emb = data['embedding'] / (np.linalg.norm(data['embedding']) + 1e-6)
        sim = np.dot(query, db_emb)
        if sim > best_sim:
            best_sim = sim
            best_match = data
    
    if best_sim < 0.45:
        return None, best_sim
    return best_match, best_sim


def detect_and_recognize(frame):
    """Detect faces and recognize them"""
    global current_faces, is_processing
    is_processing = True
    
    results = []
    
    # Fast grayscale detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
    
    for (x, y, w, h) in faces:
        # Expand slightly
        pad = int(w * 0.1)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)
        
        # Get embedding
        face_img = frame[y1:y2, x1:x2]
        embedding = get_embedding(face_img)
        match, conf = match_face(embedding)
        
        results.append({
            'box': (x, y, x+w, y+h),
            'match': match,
            'conf': conf
        })
    
    with processing_lock:
        current_faces = results
    
    is_processing = False


def run_camera():
    print("\n📷 Camera starting... (Press 'q' to quit)\n")
    
    # Use DirectShow on Windows for better performance
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    frame_count = 0
    fps_time = time.time()
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Calculate FPS
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_time)
            fps_time = time.time()
        
        # Process every 15 frames in background
        if frame_count % 15 == 0 and not is_processing:
            t = threading.Thread(target=detect_and_recognize, args=(frame.copy(),))
            t.daemon = True
            t.start()
        
        # Draw current results
        with processing_lock:
            faces_to_draw = current_faces.copy()
        
        for face in faces_to_draw:
            x1, y1, x2, y2 = face['box']
            match = face['match']
            conf = face['conf']
            
            if match:
                color = (0, 255, 0)
                label = f"{match['name']} {conf:.0%}"
            else:
                color = (0, 0, 255)
                label = f"Unknown {conf:.0%}"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Show FPS
        cv2.putText(frame, f"FPS: {fps:.0f} | Press 'q' to quit", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.imshow('Face Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Done!")


if __name__ == "__main__":
    load_database()
    run_camera()
