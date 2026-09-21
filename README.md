# 🔍 FaceID – Face Recognition System

A real-time face detection and recognition application built using Python, OpenCV, and Streamlit. FaceTrace AI allows users to enroll individuals, recognize faces from uploaded images or a live camera, and manage enrolled identities.

## ✨ Features

* **Face Enrollment:** Register a person using an uploaded image or live camera.
* **Duplicate Detection:** Check whether a newly enrolled face may match an existing identity.
* **Face Recognition:** Identify enrolled individuals from uploaded images.
* **Live Recognition:** Recognize faces in a real-time webcam stream.
* **Unknown Face Detection:** Label faces that do not meet the matching threshold as `Unknown`.
* **Enrollment Management:** View enrolled individuals and their stored face samples.
* **Deletion Confirmation:** Remove an enrolled identity and its saved enrollment photos.
* **Interactive UI:** Streamlit-based interface with image previews and recognition results.

## 🛠️ Technologies Used

| Technology       | Purpose                                 |
| ---------------- | --------------------------------------- |
| Python           | Core programming language               |
| Streamlit        | Web application interface               |
| OpenCV           | Image processing and face visualization |
| NumPy            | Numerical and array operations          |
| streamlit-webrtc | Real-time webcam streaming              |
| Threading        | Synchronization during live processing  |

## 🧠 Model Used

FaceTrace AI uses a modular face-recognition pipeline consisting of:

1. **Face Detection – `FaceDetector`**
   Detects faces in an input image or video frame.

2. **Face Embedding Extraction – `FaceEmbedder`**
   Converts a detected face into a numerical feature vector, called a face embedding.

3. **Face Matching – `FaceMatcher`**
   Compares the input embedding against stored embeddings to identify a possible match.

4. **Face Database – `FaceDatabase`**
   Stores and retrieves enrolled identities and their face embeddings.

The application initializes these modules from the `src` directory.

> **Model architecture:** The exact detector and embedding model architecture should be specified after verifying the implementations in `src/detector.py` and `src/embedder.py`. The main application code alone does not establish the underlying model name or architecture.

## ⚙️ Matching Threshold

FaceTrace AI uses cosine similarity-based matching.

| Operation                    | Threshold |
| ---------------------------- | --------: |
| Face recognition             |    `0.45` |
| Possible duplicate detection |    `0.60` |

### How it works

* During recognition, the system compares the input face embedding with stored embeddings.
* A match is accepted according to the matching logic and the recognition threshold.
* If no enrolled identity meets the matching condition, the face is labeled **Unknown**.
* During enrollment, a separate duplicate threshold is used to flag a possible match with an existing identity.

**Note:** A similarity score is not proof of identity. Thresholds are configuration values and should be validated against representative test data before using the system in real-world settings.

## 🔄 System Workflow

1. The user uploads an image or accesses the camera.
2. The face detector identifies faces in the input.
3. The embedder extracts a feature vector for each detected face.
4. The matcher compares the embedding with the enrolled database.
5. The system displays the predicted identity and similarity score.
6. Faces that do not meet the matching condition are labeled `Unknown`.

## 📊 Evaluation Results

The application implements face enrollment, recognition, duplicate checking, and live webcam processing. However, **formal evaluation metrics have not been established from the available project code**.

The following metrics should be measured using a separate, labeled test dataset.

| Evaluation Metric           | Result           |
| --------------------------- | ---------------- |
| Face detection accuracy     | Not yet measured |
| Face recognition accuracy   | Not yet measured |
| Precision                   | Not yet measured |
| Recall                      | Not yet measured |
| False Acceptance Rate (FAR) | Not yet measured |
| False Rejection Rate (FRR)  | Not yet measured |
| Average inference time      | Not yet measured |

### Suggested evaluation procedure

1. Prepare a labeled dataset containing enrolled and unenrolled individuals.
2. Test images under different lighting, poses, and distances.
3. Compare predicted identities with the actual labels.
4. Record true matches, false matches, missed matches, and correctly rejected unknown faces.
5. Evaluate multiple thresholds and document the results.

Do not report accuracy or other performance values until they have been measured.

## ⚠️ Failure Cases and Limitations

Potential failure cases to evaluate include:

* **Poor lighting:** Dark or overexposed images may reduce face detection and matching reliability.
* **Face occlusion:** Masks, sunglasses, or objects covering the face may affect recognition.
* **Pose variation:** Side-facing or tilted faces may be harder to recognize.
* **Low image quality:** Blurry or low-resolution images may produce unreliable embeddings.
* **Unregistered individuals:** Faces not present in the database should be labeled `Unknown`.
* **Similar-looking individuals:** The system may confuse people with similar facial features.
* **Enrollment quality:** Poor enrollment images may reduce recognition performance.
* **Multiple faces:** Enrollment requires an image containing exactly one detected face.
* **Live camera conditions:** Webcam quality, frame rate, and processing delays may affect real-time performance.

These are potential limitations; their frequency and severity should be confirmed through testing.

## 🚀 Possible Improvements

* Evaluate and document the exact face detection and embedding architectures.
* Tune matching thresholds using a representative validation dataset.
* Measure accuracy, precision, recall, FAR, and FRR.
* Improve robustness to lighting, pose, occlusion, and image quality.
* Optimize real-time processing speed.
* Add face alignment and image-quality checks.
* Improve database security and protect stored face embeddings.
* Add secure user authentication and access controls.
* Add configurable enrollment and database management.
* Evaluate performance across diverse demographic groups.
* Add clear consent, retention, and deletion procedures for biometric data.

## 🔒 Privacy and Security

FaceTrace AI is a demonstration project and is **not a secure authentication system**.

* Obtain consent before enrolling or recognizing anyone.
* Avoid uploading real biometric data to public repositories.
* Keep enrollment photos and face embeddings out of GitHub.
* Do not use the application as the sole basis for security-sensitive decisions.
* Add appropriate access controls, retention limits, and deletion safeguards before any real-world deployment.

## 📁 Project Structure

```text
FaceTrace-AI/
│
├── app.py
├── src/
│   ├── detector.py
│   ├── embedder.py
│   ├── matcher.py
│   └── database.py
│
├── data/
│   └── enrolled/
│
├── requirements.txt
├── README.md
└── .gitignore
```

*Update `app.py` or the folder names if your repository uses different filenames.*

## ▶️ Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/Khushi-Hegde/FaceTrace-AI.git
cd FaceTrace-AI
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal.

> Ensure the repository contains the required model files and source modules before running the application.

## 🔮 Future Scope

FaceTrace AI can be extended with improved model evaluation, stronger biometric-data protection, optimized real-time inference, and more robust recognition under varied environmental conditions.

## 👩‍💻 Author

**Khushi S Hegde**
B.E. – Artificial Intelligence and Data Science
SDM Institute of Technology, Ujire

GitHub: [Khushi-Hegde](https://github.com/Khushi-Hegde)

---

*FaceTrace AI is an educational demonstration project for face detection and recognition.*
