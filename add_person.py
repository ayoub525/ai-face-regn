"""
Add New Person to Face Recognition Database
============================================
1. Captures photos from webcam
2. Applies data augmentation
3. Extracts ArcFace embeddings
4. Adds to database

Usage: python add_person.py
"""

import cv2
import numpy as np
import pickle
from pathlib import Path
import onnxruntime as ort
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).parent
DATABASE_PATH = SCRIPT_DIR / 'recognition_database_arcface.pkl'
MODEL_PATH = Path.home() / '.insightface' / 'models' / 'buffalo_l' / 'w600k_r50.onnx'

# Load ArcFace model
print("🚀 Loading ArcFace model...")
arcface = ort.InferenceSession(str(MODEL_PATH), providers=['CPUExecutionProvider'])
print("✅ Model loaded!")

# Face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def get_embedding(face_img):
    """Get 512D embedding from face image"""
    face = cv2.resize(face_img, (112, 112))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = (face.astype(np.float32) - 127.5) / 127.5
    face = np.transpose(face, (2, 0, 1))
    face = np.expand_dims(face, 0)
    embedding = arcface.run(None, {'input.1': face})[0].flatten()
    return embedding


def augment_image(image):
    """Apply data augmentation to create variations"""
    augmented = [image]  # Original
    
    # 1. Horizontal flip
    augmented.append(cv2.flip(image, 1))
    
    # 2. Brightness variations
    for beta in [-30, -15, 15, 30]:
        bright = cv2.convertScaleAbs(image, alpha=1.0, beta=beta)
        augmented.append(bright)
    
    # 3. Contrast variations
    for alpha in [0.8, 0.9, 1.1, 1.2]:
        contrast = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        augmented.append(contrast)
    
    # 4. Slight rotations
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    for angle in [-10, -5, 5, 10]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        augmented.append(rotated)
    
    # 5. Gaussian blur (slight)
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    augmented.append(blurred)
    
    # 6. Add slight noise
    noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
    noisy = cv2.add(image, noise)
    augmented.append(noisy)
    
    return augmented


def capture_faces(person_name, num_photos=15):
    """Capture face photos from webcam"""
    print(f"\n📷 Capturing photos for: {person_name}")
    print("   - Press SPACE to capture a photo")
    print("   - Press 'q' when done (minimum 5 photos)")
    print("   - Move your head slightly between captures\n")
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return []
    
    captured_faces = []
    
    while len(captured_faces) < num_photos:
        ret, frame = cap.read()
        if not ret:
            break
        
        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(100, 100))
        
        face_crop = None
        for (x, y, w, h) in faces:
            # Expand box
            pad = int(w * 0.2)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)
            
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            face_crop = frame[y1:y2, x1:x2]
            break  # Only first face
        
        # Show instructions
        cv2.putText(display, f"Photos: {len(captured_faces)}/{num_photos}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display, "SPACE=Capture | Q=Done", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if face_crop is not None:
            cv2.putText(display, "Face detected!", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        cv2.imshow('Capture Face', display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and face_crop is not None:
            captured_faces.append(face_crop)
            print(f"   ✓ Captured photo {len(captured_faces)}")
        elif key == ord('q'):
            if len(captured_faces) >= 5:
                break
            else:
                print(f"   ⚠️ Need at least 5 photos (have {len(captured_faces)})")
    
    cap.release()
    cv2.destroyAllWindows()
    
    return captured_faces


def add_person_to_database(person_id, person_name, face_images):
    """Process faces and add to database"""
    print(f"\n🔄 Processing {len(face_images)} photos...")
    
    all_embeddings = []
    
    for i, face in enumerate(face_images):
        # Apply augmentation
        augmented = augment_image(face)
        print(f"   Photo {i+1}: {len(augmented)} augmented versions")
        
        # Extract embeddings
        for aug_face in augmented:
            try:
                emb = get_embedding(aug_face)
                all_embeddings.append(emb)
            except:
                pass
    
    if len(all_embeddings) == 0:
        print("❌ No embeddings extracted!")
        return False
    
    # Calculate mean embedding
    embeddings_array = np.array(all_embeddings)
    mean_embedding = np.mean(embeddings_array, axis=0)
    mean_embedding = mean_embedding / (np.linalg.norm(mean_embedding) + 1e-6)
    
    print(f"✅ Generated {len(all_embeddings)} embeddings")
    print(f"   Mean embedding shape: {mean_embedding.shape}")
    
    # Load existing database
    if DATABASE_PATH.exists():
        with open(DATABASE_PATH, 'rb') as f:
            database = pickle.load(f)
    else:
        database = {'persons': {}}
    
    # Create person key
    person_key = f"{person_id}_{person_name.replace(' ', '_')}"
    
    # Add to database
    database['persons'][person_key] = {
        'mean_embedding': mean_embedding,
        'embeddings': embeddings_array
    }
    
    # Save database
    with open(DATABASE_PATH, 'wb') as f:
        pickle.dump(database, f)
    
    print(f"\n✅ Added '{person_name}' (ID: {person_id}) to database!")
    print(f"   Total persons in database: {len(database['persons'])}")
    
    return True


def get_next_id():
    """Get next available ID"""
    if DATABASE_PATH.exists():
        with open(DATABASE_PATH, 'rb') as f:
            database = pickle.load(f)
        
        max_id = 0
        for key in database['persons'].keys():
            try:
                id_num = int(key.split('_')[0])
                max_id = max(max_id, id_num)
            except:
                pass
        return max_id + 1
    return 1


def main():
    print("=" * 50)
    print("   ADD NEW PERSON TO DATABASE")
    print("=" * 50)
    
    # Get person info
    name = input("\n👤 Enter person's name: ").strip()
    if not name:
        print("❌ Name cannot be empty!")
        return
    
    # Auto-generate ID
    person_id = get_next_id()
    print(f"   Assigned ID: {person_id}")
    
    # Capture photos
    faces = capture_faces(name, num_photos=15)
    
    if len(faces) < 5:
        print(f"❌ Not enough photos captured ({len(faces)}). Need at least 5.")
        return
    
    print(f"\n📸 Captured {len(faces)} photos")
    
    # Process and add to database
    success = add_person_to_database(person_id, name, faces)
    
    if success:
        print("\n" + "=" * 50)
        print("✅ DONE! Your friend has been added.")
        print("   Run 'python face_recognition_camera.py' to test!")
        print("=" * 50)


if __name__ == "__main__":
    main()
