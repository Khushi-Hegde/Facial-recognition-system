
import streamlit as st
import cv2
import numpy as np
import re

from pathlib import Path
from datetime import datetime

from src.detector import FaceDetector
from src.embedder import FaceEmbedder
from src.matcher import FaceMatcher
from src.database import FaceDatabase
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import threading

# --------------------------------
# Project paths
# --------------------------------
BASE_DIR = Path(__file__).resolve().parent

ENROLLED_IMAGES_DIR = (
    BASE_DIR / "data" / "enrolled"
)

ENROLLED_IMAGES_DIR.mkdir(
    parents=True,
    exist_ok=True
)



# --------------------------------
# Page configuration
# --------------------------------
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="🔍",
    layout="centered"
)


# --------------------------------
# Clean & Attractive Light UI
# --------------------------------
st.markdown("""
<style>

/* Page background */
.stApp {
    background: linear-gradient(
        135deg,
        #f8faff,
        #edf2ff,
        #f8f5ff
    );
    color: #273451;
}

/* Content spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}

/* Main title */
h1 {
    color: #3446a8 !important;
    font-weight: 750 !important;
}

/* Section headings */
h2, h3 {
    color: #4657b8 !important;
    font-weight: 650 !important;
}

/* Body text */
p, label {
    color: #46536b;
}

/* Input boxes */
.stTextInput input {
    background: #ffffff !important;
    color: #273451 !important;
    border: 1px solid #d6def2 !important;
    border-radius: 10px !important;
}

/* Upload area */
[data-testid="stFileUploader"] section {
    background: #ffffff !important;
    border: 1.5px dashed #aab8ed !important;
    border-radius: 14px !important;
    padding: 18px !important;
}

/* Upload button */
[data-testid="stFileUploader"] button {
    background: #eef1ff !important;
    color: #4355b5 !important;
    border: 1px solid #d5dcff !important;
    border-radius: 8px !important;
}

/* Camera area */
[data-testid="stCameraInput"] {
    border-radius: 14px !important;
    overflow: hidden;
}

/* Primary buttons */
.stButton > button {
    background: linear-gradient(
        100deg,
        #536dfe,
        #8265e8
    ) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.2rem !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

/* Button hover */
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 14px #687cf044;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #53617c !important;
    font-weight: 600 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #536dfe !important;
}

/* Rounded images */
[data-testid="stImage"] img {
    border-radius: 14px;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Metrics and bordered sections */
[data-testid="stMetric"],
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px;
}

/* Gentle entrance animation */
@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(8px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.block-container {
    animation: fadeIn 0.6s ease-out;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------
# Load models and database
# --------------------------------
@st.cache_resource
def load_system():

    detector = FaceDetector()
    embedder = FaceEmbedder()
    database = FaceDatabase()

    matcher = FaceMatcher(
        embedder.recognizer,
        threshold=0.45
    )

    return detector, embedder, database, matcher


try:
    detector, embedder, database, matcher = (
        load_system()
    )

except Exception as error:
    st.error(
        f"Could not load the system: {error}"
    )
    st.stop()


# --------------------------------
# Image processing
# --------------------------------
def read_image(uploaded_file):

    file_bytes = np.frombuffer(
        uploaded_file.getvalue(),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    return image


def get_faces_and_embeddings(image):

    faces = detector.detect_faces(image)

    results = []

    for face in faces:

        embedding = embedder.get_embedding(
            image,
            face
        )

        results.append(
            (face, embedding)
        )

    return results


# --------------------------------
# Save enrollment photo
# --------------------------------
def save_enrollment_image(name, image):

    safe_name = re.sub(
        r"[^a-zA-Z0-9_-]",
        "_",
        name.strip()
    )

    if not safe_name:
        raise ValueError(
            "Please enter a valid name."
        )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    person_dir = (
        ENROLLED_IMAGES_DIR / safe_name
    )

    person_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    image_path = (
        person_dir / f"{timestamp}.jpg"
    )

    success = cv2.imwrite(
        str(image_path),
        image
    )

    if not success:
        raise RuntimeError(
            "Could not save the enrollment photo."
        )

    return image_path


# --------------------------------
# Interface tabs
# --------------------------------
st.title("🔍 Face Recognition System")

st.caption(
    "✨ Smart face detection and identification powered by AI"
)

st.info(
    "Use images only with the person's consent. "
    "This is a local demonstration, not a secure authentication system."
)
enroll_tab, recognize_tab, live_tab = st.tabs(
    [
        "👤 Enroll Person",
        "🔎 Recognize Face",
        "📹 Live Recognition"
    ]
)



# =================================
# ENROLLMENT
# =================================
with enroll_tab:

    st.subheader("Enroll a Person")

    st.caption(
        "You can enroll multiple photos of the same person "
        "to improve recognition across different appearances."
    )

    person_name = st.text_input(
        "Enter person's name",
        key="person_name"
    )

    enroll_source = st.radio(
        "Choose enrollment method",
        ["Upload Image", "Live Camera"],
        key="enroll_source",
        horizontal=True
    )

    if enroll_source == "Upload Image":
        enroll_image = st.file_uploader(
            "Upload a clear face image",
            type=["jpg", "jpeg", "png"],
            key="enroll_image"
        )
    else:
        enroll_image = st.camera_input(
            "Capture enrollment photo",
            key="enroll_camera"
        )

    # Process the selected image
    if enroll_image is not None:

        image = read_image(enroll_image)

        if image is None:
            st.error("Could not read this image.")

        else:
            st.image(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                caption="Enrollment photo",
                width="stretch"
            )

            if st.button(
                "Check & Enroll",
                key="enroll_button"
            ):

                name = person_name.strip()

                if not name:
                    st.warning("Please enter a person's name.")

                else:
                    faces = detector.detect_faces(image)

                    if len(faces) == 0:
                        st.error(
                            "No face detected. "
                            "Try a clearer image."
                        )

                    elif len(faces) > 1:
                        st.warning(
                            "Multiple faces detected. "
                            "Please use an image containing "
                            "only one person."
                        )

                    else:
                        try:
                            embedding = embedder.get_embedding(
                                image,
                                faces[0]
                            )

                            # Check whether this face may already
                            # exist under a different person's name.
                            duplicate_name, duplicate_score = (
                                matcher.check_duplicate(
                                    embedding,
                                    database.get_all(),
                                    exclude_name=name,
                                    duplicate_threshold=0.60
                                )
                            )

                            # Keep the pending enrollment in session
                            # state until the user confirms it.
                            st.session_state["pending_enrollment"] = {
                                "name": name,
                                "embedding": np.asarray(
                                    embedding,
                                    dtype=float
                                ).flatten().tolist(),
                                "image": image.copy(),
                                "duplicate_name": duplicate_name,
                                "duplicate_score": duplicate_score
                            }

                        except Exception as error:
                            st.error(
                                f"Could not process face: {error}"
                            )

    # --------------------------------
    # Duplicate warning and confirmation
    # --------------------------------
    pending = st.session_state.get("pending_enrollment")

    if pending is not None:

        st.divider()
        st.subheader("Enrollment Confirmation")

        if pending["duplicate_name"] is not None:

            st.warning(
                "This face may be similar to an existing "
                "enrollment under another name."
            )

            st.write(
                f"Possible match: "
                f"**{pending['duplicate_name']}**"
            )

            st.write(
                f"Similarity score: "
                f"{pending['duplicate_score']:.4f}"
            )

            st.caption(
                "Similarity is not proof of identity. "
                "Check the images before proceeding."
            )

            st.write(
                "If this is genuinely a different person, "
                "you can confirm enrollment."
            )

        else:
            st.info(
                "No possible duplicate was detected. "
                "Confirm to save this enrollment."
            )

        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "Confirm Enrollment",
                key="confirm_enrollment"
            ):

                try:
                    # Add this sample to the person's existing
                    # list of embeddings.
                    database.enroll(
                        pending["name"],
                        pending["embedding"]
                    )

                    saved_path = save_enrollment_image(
                        pending["name"],
                        pending["image"]
                    )

                    st.session_state.pop(
                        "pending_enrollment",
                        None
                    )

                    st.success(
                        f"{pending['name']} enrolled successfully!"
                    )

                    st.write("Enrollment photo saved:")
                    st.code(
                        str(saved_path.relative_to(BASE_DIR))
                    )

                except Exception as error:
                    st.error(
                        f"Enrollment failed: {error}"
                    )

        with col2:
            if st.button(
                "Cancel",
                key="cancel_enrollment"
            ):
                st.session_state.pop(
                    "pending_enrollment",
                    None
                )
                st.info("Enrollment cancelled.")

# =================================
# RECOGNITION
# =================================
with recognize_tab:

    st.subheader("Recognize a Person")

    recognize_source = st.radio(
        "Choose recognition method",
        [
            "Upload Image",
            "Live Camera"
        ],
        key="recognize_source",
        horizontal=True
    )

    # Choose recognition image source
    if recognize_source == "Upload Image":

        recognize_image = st.file_uploader(
            "Upload an image to identify",
            type=["jpg", "jpeg", "png"],
            key="recognize_image"
        )

    else:

        recognize_image = st.camera_input(
            "Capture face for recognition",
            key="recognize_camera"
        )

    # Process recognition image
    if recognize_image is not None:

        image = read_image(
            recognize_image
        )

        if image is None:

            st.error(
                "Could not read this image."
            )

        else:

            st.image(
                cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                ),
                caption="Image for recognition",
                width="stretch"
            )

            if st.button(
                "Recognize Face",
                key="recognize_button"
            ):

                results = (
                    get_faces_and_embeddings(
                        image
                    )
                )

                if len(results) == 0:

                    st.error(
                        "No face detected. "
                        "Try another image."
                    )

                else:

                    display_image = image.copy()

                    enrolled_faces = (
                        database.get_all()
                    )

                    if not enrolled_faces:

                        st.warning(
                            "No individuals are enrolled "
                            "in the database yet."
                        )

                    else:

                        for face, embedding in results:

                            name, score = matcher.match(
                                embedding,
                                enrolled_faces
                            )

                            x, y, w, h = (
                                face[:4].astype(int)
                            )

                            color = (
                                (0, 180, 0)
                                if name != "Unknown"
                                else (0, 0, 255)
                            )

                            # Draw bounding box
                            cv2.rectangle(
                                display_image,
                                (x, y),
                                (x + w, y + h),
                                color,
                                2
                            )

                            # Display predicted name
                            cv2.putText(
                                display_image,
                                name,
                                (
                                    x,
                                    max(y - 10, 20)
                                ),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                color,
                                2
                            )

                            st.markdown(
                                f"### Result: {name}"
                            )

                            st.write(
                                "Cosine similarity: "
                                f"{score:.4f}"
                            )

                            if name == "Unknown":

                                st.warning(
                                    "No enrolled identity "
                                    "met the matching threshold."
                                )

                            else:

                                st.success(
                                    "Matching identity found."
                                )

                    # Display final annotated image
                    st.image(
                        cv2.cvtColor(
                            display_image,
                            cv2.COLOR_BGR2RGB
                        ),
                        caption="Recognition result",
                        width="stretch"
                    )



# =================================
# ENROLLED INDIVIDUALS
# =================================

st.divider()
st.subheader("👥 Enrolled Individuals")

people = database.get_all()

if people:

    st.write(
        f"Total registered people: {len(people)}"
    )

    for name, embeddings in people.items():

        with st.expander(f"👤 {name}"):

            st.write(
                f"Face samples: {len(embeddings)}"
            )

            if st.button(
                f"🗑️ Delete {name}",
                key=f"delete_{name}"
            ):
                st.session_state["delete_person"] = name

    # --------------------------------
    # Deletion confirmation
    # --------------------------------

    person_to_delete = st.session_state.get(
        "delete_person"
    )

    if person_to_delete in people:

        st.warning(
            f"Are you sure you want to delete "
            f"{person_to_delete}?"
        )

        st.caption(
            "This will remove their face embeddings "
            "and saved enrollment photos."
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "Yes, delete",
                key="confirm_delete"
            ):

                # Remove embeddings from database
                database.remove(person_to_delete)

                # Remove saved enrollment photos
                safe_name = re.sub(
                    r"[^a-zA-Z0-9_-]",
                    "_",
                    person_to_delete.strip()
                )

                person_dir = (
                    ENROLLED_IMAGES_DIR / safe_name
                )

                if person_dir.exists():
                    import shutil
                    shutil.rmtree(person_dir)

                st.session_state.pop(
                    "delete_person",
                    None
                )

                st.success(
                    f"{person_to_delete} was deleted."
                )

                st.rerun()

        with col2:
            if st.button(
                "Cancel",
                key="cancel_delete"
            ):
                st.session_state.pop(
                    "delete_person",
                    None
                )

                st.rerun()

else:

    st.info(
        "No individuals enrolled yet."
    )


# =================================
# LIVE CAMERA RECOGNITION
# =================================

class LiveFaceProcessor(VideoProcessorBase):

    def __init__(self, detector, embedder, matcher, database):
        self.detector = detector
        self.embedder = embedder
        self.matcher = matcher
        self.database = database
        self.lock = threading.Lock()

    def recv(self, frame):

        # Convert incoming video frame to OpenCV format.
        image = frame.to_ndarray(format="bgr24")

        try:
            with self.lock:

                # Detect faces in the current frame.
                faces = self.detector.detect_faces(image)

                # Get current enrolled identities.
                enrolled_faces = self.database.get_all()

                for face in faces:

                    try:
                        # Extract face embedding.
                        embedding = self.embedder.get_embedding(
                            image,
                            face
                        )

                        # Match against enrolled identities.
                        name, score = self.matcher.match(
                            embedding,
                            enrolled_faces
                        )

                        x, y, w, h = face[:4].astype(int)

                        if name == "Unknown":
                            color = (0, 0, 255)
                            label = f"Unknown ({score:.2f})"
                        else:
                            color = (0, 200, 0)
                            label = f"{name} ({score:.2f})"

                        # Draw face bounding box.
                        cv2.rectangle(
                            image,
                            (x, y),
                            (x + w, y + h),
                            color,
                            2
                        )

                        # Draw name and similarity score.
                        cv2.putText(
                            image,
                            label,
                            (x, max(y - 10, 25)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            color,
                            2
                        )

                    except Exception:
                        # Skip a problematic face and continue.
                        continue

        except Exception as error:
            print(f"Live recognition error: {error}")

        return frame.from_ndarray(
            image,
            format="bgr24"
        )


with live_tab:

    st.subheader("📹 Real-Time Face Recognition")

    st.write(
        "Start your webcam to identify enrolled people "
        "or label unregistered faces as Unknown."
    )

    st.info(
        "Allow camera access when your browser asks. "
        "Use this feature only with people's consent."
    )

    # Create the live webcam stream.
    webrtc_streamer(
        key="live-face-recognition",

        video_processor_factory=lambda: LiveFaceProcessor(
            detector,
            embedder,
            matcher,
            database
        ),

        rtc_configuration={
            "iceServers": [
                {
                    "urls": ["stun:stun.l.google.com:19302"]
                }
            ]
        },

        media_stream_constraints={
            "video": True,
            "audio": False
        },

        async_processing=True
    )