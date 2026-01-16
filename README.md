# 🎭 AI Face Recognition System

A real-time face recognition system built with Python, OpenCV, and ArcFace deep learning model. This project allows you to register faces and recognize them in real-time through your webcam.

---

## 🌟 Features

- **Real-time Face Recognition** - Recognize faces from webcam feed with high FPS
- **Face Registration** - Add new people to the database with data augmentation
- **Database Management** - View and manage registered faces
- **Threaded Processing** - Background processing for smooth video display
- **Data Augmentation** - Automatically creates variations of captured photos for better recognition

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python 3** | Core programming language |
| **OpenCV (cv2)** | Camera capture, image processing, and face detection |
| **ONNX Runtime** | High-performance inference engine for deep learning models |
| **ArcFace (w600k_r50)** | State-of-the-art face recognition model that generates 512D embeddings |
| **NumPy** | Numerical computing for embedding operations |
| **Pickle** | Database serialization for storing face embeddings |
| **Threading** | Parallel processing for smooth real-time performance |

---

## 🤖 How the AI Recognition Works

### 1. Face Detection
The system uses **OpenCV's Haar Cascade Classifier** (`haarcascade_frontalface_default.xml`) to detect faces in each frame. This is a fast, classical computer vision approach.

### 2. Face Embedding Extraction
Once a face is detected, it's passed through the **ArcFace** deep learning model:
- The face image is resized to **112x112 pixels**
- Preprocessed (RGB conversion, normalization)
- Fed into the ArcFace ONNX model
- Outputs a **512-dimensional embedding vector** that uniquely represents the face

### 3. Face Matching
Recognition works by comparing embeddings:
- **Cosine similarity** is calculated between the detected face embedding and all stored embeddings in the database
- If similarity > **0.45** (45%), the face is recognized
- Higher similarity = more confident match

### 4. Data Augmentation (for registration)
When adding a new person, the system applies augmentation to improve recognition accuracy:
- Horizontal flips
- Brightness variations (+/- 15, 30)
- Contrast variations (0.8x - 1.2x)
- Slight rotations (-10° to +10°)
- Gaussian blur
- Random noise

---

## 📁 Project Structure

```
ai-face--regn-main/
├── face_recognition_camera.py   # Main real-time recognition script
├── add_person.py                # Register new people to database
├── view_database.py             # View all registered people
├── recognition_database_arcface.pkl  # Stored face embeddings database
├── arcface_model.onnx           # Small ArcFace model
├── w600k_r50.onnx               # Large ArcFace model (174MB)
└── .gitignore
```

---

## 🚀 Installation

### Prerequisites
- Python 3.7+
- Webcam

### Install Dependencies

```bash
pip install opencv-python numpy onnxruntime
```

### Download ArcFace Model
The system expects the ArcFace model at:
```
~/.insightface/models/buffalo_l/w600k_r50.onnx
```

Alternatively, you can use the included `w600k_r50.onnx` file by updating the `MODEL_PATH` in the scripts.

---

## 📖 Usage

### 1. Add a New Person
```bash
python add_person.py
```
- Enter the person's name
- Press **SPACE** to capture photos (minimum 5, recommended 15)
- Move your head slightly between captures
- Press **Q** when done

### 2. Run Face Recognition
```bash
python face_recognition_camera.py
```
- Faces will be detected and recognized in real-time
- **Green box** = Recognized person (with name and confidence %)
- **Red box** = Unknown person
- Press **Q** to quit

### 3. View Database
```bash
python view_database.py
```
Shows all registered people with their IDs and embedding counts.

---

## ⚙️ Configuration

### Recognition Threshold
In `face_recognition_camera.py`, line 98:
```python
if best_sim < 0.45:  # Adjust threshold (0.0 - 1.0)
```
- **Lower** = More lenient (may have false positives)
- **Higher** = Stricter (may miss valid matches)

### Frame Processing Interval
In `face_recognition_camera.py`, line 170:
```python
if frame_count % 15 == 0:  # Process every 15 frames
```
- **Lower** = More responsive but slower FPS
- **Higher** = Faster FPS but less responsive

---

## 📊 Performance

- Camera resolution: **640x480** @ **30 FPS**
- Processing runs in background threads
- Recognition typically runs at **20-30+ FPS** depending on hardware

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera not opening | Make sure no other app is using the webcam |
| Model not found | Check the `MODEL_PATH` points to a valid `.onnx` file |
| Low recognition accuracy | Capture more photos with varied angles/lighting |
| Slow performance | Increase the frame processing interval or use a smaller detection `minSize` |

---

## 📜 License

This project is open source and available for personal and educational use.

---

## 🙏 Acknowledgments

- **InsightFace** - For the ArcFace model
- **OpenCV** - For face detection and image processing
- **ONNX Runtime** - For efficient model inference
