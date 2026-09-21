import streamlit as st
import cv2
import numpy as np
import re
import threading
import shutil

from pathlib import Path
from datetime import datetime

from src.detector import FaceDetector
from src.embedder import FaceEmbedder
from src.matcher import FaceMatcher
from src.database import FaceDatabase
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import random


# --------------------------------
# Dot-mesh face illustration
# --------------------------------
def face_mesh_svg(width=340, height=360, seed=7):
    """Builds a front-facing, point-cloud style face as inline SVG:
    a dense grid of dots clipped to a face silhouette, lit from the
    right, with a few feature outlines (eyes, brows, nose, mouth)."""

    random.seed(seed)

    cx, cy = width / 2, height / 2 - 6
    half_w, half_h = width * 0.30, height * 0.42

    def width_factor(v):
        # v runs -1 (forehead top) .. 1 (chin tip)
        if v < -0.6:
            t = (v + 1) / 0.4
            return 0.82 + 0.18 * t
        elif v < 0.45:
            return 1.0
        else:
            t = (v - 0.45) / 0.55
            return 1.0 - 0.62 * (t ** 1.3)

    dots = []
    y = -1.0

    while y <= 1.0:

        wf = width_factor(y)
        x = -1.0

        while x <= 1.0:

            if abs(x) <= wf:

                jx = x + random.uniform(-0.015, 0.015)
                jy = y + random.uniform(-0.015, 0.015)

                px = cx + jx * half_w
                py = cy + jy * half_h

                edge_dist = wf - abs(x)
                lit = (x + 1) / 2

                opacity = 0.32 + 0.5 * lit + 0.15 * min(edge_dist, 0.15) / 0.15
                opacity = max(0.22, min(opacity, 0.95))

                radius = 0.9 + 0.65 * lit
                color = "#22d3ee" if lit > 0.55 else "#6366f1"

                dots.append((px, py, radius, opacity, color))

            x += 0.10

        y += 0.095

    circles = "".join(
        f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{r:.2f}" '
        f'fill="{c}" opacity="{o:.2f}"/>'
        for px, py, r, o, c in dots
    )

    def pt(nx, nv):
        return cx + nx * half_w, cy + nv * half_h

    lex, ley = pt(-0.30, -0.15)
    rex, rey = pt(0.30, -0.15)
    eye_rx, eye_ry = 0.14 * half_w, 0.075 * half_h

    nose_top_x, nose_top_y = pt(0, -0.12)
    nose_bot_x, nose_bot_y = pt(0.02, 0.20)

    mouth_lx, mouth_ly = pt(-0.20, 0.42)
    mouth_rx, mouth_ry = pt(0.20, 0.42)
    mouth_mx, mouth_my = pt(0.0, 0.48)

    features = f"""
    <g stroke="#67e8f9" stroke-width="1" fill="none" opacity="0.75">
        <ellipse cx="{lex:.1f}" cy="{ley:.1f}" rx="{eye_rx:.1f}" ry="{eye_ry:.1f}"/>
        <ellipse cx="{rex:.1f}" cy="{rey:.1f}" rx="{eye_rx:.1f}" ry="{eye_ry:.1f}"/>
        <path d="M {lex - eye_rx:.1f} {ley - eye_ry * 1.9:.1f}
                 Q {lex:.1f} {ley - eye_ry * 2.6:.1f} {lex + eye_rx:.1f} {ley - eye_ry * 1.9:.1f}"/>
        <path d="M {rex - eye_rx:.1f} {rey - eye_ry * 1.9:.1f}
                 Q {rex:.1f} {rey - eye_ry * 2.6:.1f} {rex + eye_rx:.1f} {rey - eye_ry * 1.9:.1f}"/>
        <path d="M {nose_top_x:.1f} {nose_top_y:.1f}
                 L {nose_bot_x:.1f} {nose_bot_y:.1f}
                 Q {nose_bot_x + 10:.1f} {nose_bot_y + 8:.1f} {nose_bot_x:.1f} {nose_bot_y + 12:.1f}"/>
        <path d="M {mouth_lx:.1f} {mouth_ly:.1f}
                 Q {mouth_mx:.1f} {mouth_my:.1f} {mouth_rx:.1f} {mouth_ry:.1f}"/>
    </g>
    <circle cx="{lex:.1f}" cy="{ley:.1f}" r="1.6" fill="#e8feff"/>
    <circle cx="{rex:.1f}" cy="{rey:.1f}" r="1.6" fill="#e8feff"/>
    """

    return f"""
    <svg viewBox="0 0 {width} {height}" width="100%" height="300"
         xmlns="http://www.w3.org/2000/svg" role="img"
         aria-label="Point-cloud illustration of a face">
        {circles}
        {features}
    </svg>
    """


# --------------------------------
# Project paths
# --------------------------------
BASE_DIR = Path(__file__).resolve().parent

ENROLLED_IMAGES_DIR = BASE_DIR / "data" / "enrolled"

ENROLLED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------
# Page configuration
# --------------------------------
st.set_page_config(
    page_title="FaceID — Face Recognition System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------
# Modern Dark (Cyber) theme
# --------------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg-deep:    #070b14;
    --bg-panel:   #0d1524;
    --bg-raised:  #121c30;
    --line:       #1e2c47;
    --cyan:       #22d3ee;
    --indigo:     #6366f1;
    --text:       #e8eefc;
    --muted:      #8ea0c0;
}

/* ---- Page shell ---- */
.stApp {
    background:
        radial-gradient(900px 500px at 78% 6%, #12325111 0%, transparent 70%),
        radial-gradient(700px 400px at 10% 90%, #4f46e514 0%, transparent 70%),
        var(--bg-deep);
    color: var(--text);
    font-family: 'Inter', system-ui, sans-serif;
}

.block-container {
    padding-top: 3.5rem;
    padding-bottom: 3rem;
    max-width: 1220px;
}

/* Streamlit's fixed top toolbar sits above the content;
   make sure it never overlaps the page heading. */
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 3rem;
}

h1, h2, h3, h4 {
    font-family: 'Sora', system-ui, sans-serif !important;
    color: var(--text) !important;
    letter-spacing: -0.015em;
}

p, label, span, li {
    color: var(--text);
}

[data-testid="stCaptionContainer"] p,
.stMarkdown small {
    color: var(--muted) !important;
}

hr, [data-testid="stDivider"] {
    border-color: var(--line) !important;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: var(--bg-panel);
    border-right: 1px solid var(--line);
}

[data-testid="stSidebar"] .block-container {
    padding-top: 1.2rem;
}

.brand {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.35rem;
    padding: 0.4rem 0 1.1rem 0;
    border-bottom: 1px solid var(--line);
    margin-bottom: 1rem;
}

.brand-mark {
    width: 54px;
    height: 54px;
    border-radius: 16px;
    display: grid;
    place-items: center;
    font-size: 26px;
    background: linear-gradient(145deg, #1b2b4d, #0f1a2e);
    border: 1px solid #2a3f66;
    box-shadow: 0 0 22px #22d3ee22;
}

.brand-name {
    font-family: 'Sora', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text);
}

.brand-tag {
    font-size: 0.68rem;
    line-height: 1.5;
    text-align: center;
    color: var(--muted);
}

/* Sidebar nav buttons */
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    text-align: left !important;
    justify-content: flex-start !important;
    background: transparent !important;
    color: var(--muted) !important;
    border: 1px solid transparent !important;
    border-radius: 10px !important;
    padding: 0.55rem 0.8rem !important;
    font-weight: 500 !important;
    box-shadow: none !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: #16223a !important;
    color: var(--text) !important;
    transform: none;
}

/* Active nav item */
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(100deg, var(--indigo), #3b82f6) !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: 0 6px 18px #4f46e544 !important;
}

.side-quote {
    margin-top: 1.4rem;
    padding-top: 1rem;
    border-top: 1px solid var(--line);
    font-size: 0.75rem;
    font-style: italic;
    color: var(--muted);
    line-height: 1.6;
}

.side-rule {
    width: 34px;
    height: 3px;
    border-radius: 3px;
    background: var(--cyan);
    margin-top: 0.8rem;
}

/* ---- Hero ---- */
.hero {
    position: relative;
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 2.6rem 2.4rem;
    overflow: hidden;
    background:
        radial-gradient(520px 320px at 88% 50%, #0e3a5522 0%, transparent 72%),
        linear-gradient(120deg, #0c1425 0%, #0a1120 100%),
        var(--bg-panel);
}

.hero h1 {
    font-size: 2.4rem;
    margin: 0 0 0.5rem 0;
}

.hero .lede {
    color: var(--cyan);
    font-weight: 600;
    font-size: 0.98rem;
    margin-bottom: 0.4rem;
}

.hero p.sub {
    color: var(--muted);
    max-width: 52ch;
    line-height: 1.7;
    margin-bottom: 0.4rem;
}

/* ---- Feature cards ---- */
.feature {
    height: 100%;
    background: var(--bg-raised);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 1.25rem 1.1rem;
    text-align: center;
}

.feature .ic {
    font-size: 1.5rem;
    display: block;
    margin-bottom: 0.55rem;
}

.feature .ft {
    font-family: 'Sora', sans-serif;
    font-weight: 600;
    font-size: 0.92rem;
    margin-bottom: 0.25rem;
}

.feature .fd {
    font-size: 0.76rem;
    color: var(--muted);
    line-height: 1.55;
}

/* ---- Section panel ---- */
.panel-title {
    font-family: 'Sora', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}

.panel-sub {
    color: var(--muted);
    font-size: 0.88rem;
    margin-bottom: 1.2rem;
}

/* ---- Inputs ---- */
.stTextInput input {
    background: var(--bg-raised) !important;
    color: var(--text) !important;
    border: 1px solid var(--line) !important;
    border-radius: 10px !important;
}

.stTextInput input:focus {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 0 2px #22d3ee33 !important;
}

[data-testid="stFileUploader"] section {
    background: var(--bg-raised) !important;
    border: 1.5px dashed #2e4268 !important;
    border-radius: 14px !important;
    padding: 18px !important;
}

[data-testid="stFileUploader"] button {
    background: #1a2843 !important;
    color: var(--text) !important;
    border: 1px solid var(--line) !important;
    border-radius: 8px !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: var(--muted) !important;
}

[data-testid="stCameraInput"] {
    border-radius: 14px !important;
    overflow: hidden;
}

div[role="radiogroup"] label p {
    color: var(--text) !important;
}

/* ---- Buttons (main area) ---- */
.main .stButton > button,
[data-testid="stAppViewContainer"] .stButton > button {
    background: linear-gradient(100deg, var(--indigo), #4f7cf7) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 11px !important;
    padding: 0.55rem 1.3rem !important;
    font-weight: 600 !important;
    transition: transform 0.18s ease, box-shadow 0.18s ease !important;
}

[data-testid="stAppViewContainer"] .stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 20px #4f46e544;
}

[data-testid="stAppViewContainer"] .stButton > button[kind="secondary"] {
    background: var(--bg-raised) !important;
    color: var(--text) !important;
    border: 1px solid var(--line) !important;
}

/* ---- Images, alerts, expanders, code ---- */
[data-testid="stImage"] img {
    border-radius: 14px;
    border: 1px solid var(--line);
}

[data-testid="stAlert"] {
    background: var(--bg-raised) !important;
    border: 1px solid var(--line) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
}

[data-testid="stAlert"] p {
    color: var(--text) !important;
}

[data-testid="stExpander"] {
    background: var(--bg-raised);
    border: 1px solid var(--line) !important;
    border-radius: 12px !important;
}

[data-testid="stExpander"] summary p {
    color: var(--text) !important;
}

.stCodeBlock, pre {
    background: #0a1120 !important;
    border: 1px solid var(--line) !important;
    border-radius: 10px !important;
}

/* Reduced motion */
@media (prefers-reduced-motion: reduce) {
    * { transition: none !important; animation: none !important; }
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
    detector, embedder, database, matcher = load_system()

except Exception as error:
    st.error(f"Could not load the system: {error}")
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

        embedding = embedder.get_embedding(image, face)

        results.append((face, embedding))

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
        raise ValueError("Please enter a valid name.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    person_dir = ENROLLED_IMAGES_DIR / safe_name

    person_dir.mkdir(parents=True, exist_ok=True)

    image_path = person_dir / f"{timestamp}.jpg"

    success = cv2.imwrite(str(image_path), image)

    if not success:
        raise RuntimeError("Could not save the enrollment photo.")

    return image_path


# --------------------------------
# Navigation state
# --------------------------------
PAGES = [
    ("Home", "🏠"),
    ("Enroll Face", "👤"),
    ("Recognize", "🔎"),
    ("Live Recognition", "📹"),
    ("Database", "🗂️"),
    ("About", "ℹ️"),
]

if "page" not in st.session_state:
    st.session_state["page"] = "Home"


def go_to(page_name):
    st.session_state["page"] = page_name


# --------------------------------
# Sidebar
# --------------------------------
with st.sidebar:

    st.markdown("""
    <div class="brand">
        <div class="brand-mark">🧠</div>
        <div class="brand-name">FaceID</div>
        <div class="brand-tag">Recognize today<br>Secure tomorrow</div>
    </div>
    """, unsafe_allow_html=True)

    for page_name, icon in PAGES:

        is_active = st.session_state["page"] == page_name

        st.button(
            f"{icon}  {page_name}",
            key=f"nav_{page_name}",
            type="primary" if is_active else "secondary",
            on_click=go_to,
            args=(page_name,),
            use_container_width=True
        )

    st.markdown("""
    <div class="side-quote">
        "Faces are unique,<br>just like your story."
        <div class="side-rule"></div>
    </div>
    """, unsafe_allow_html=True)


page = st.session_state["page"]


# =================================
# HOME
# =================================
if page == "Home":

    left, right = st.columns([1.35, 1], gap="large")

    with left:
        st.markdown("""
        <div class="hero">
            <div class="lede">AI-powered. Secure. Simple.</div>
            <h1>Face Recognition System</h1>
            <p class="sub">
                Enroll, recognize and manage faces with ease using
                YuNet detection and SFace embeddings — running entirely
                on your own machine.
            </p>
        </div>
        """, unsafe_allow_html=True)

        act1, act2, _ = st.columns([1, 1, 0.6])

        with act1:
            st.button(
                "👤  Enroll a face",
                key="cta_enroll",
                on_click=go_to,
                args=("Enroll Face",),
                use_container_width=True
            )

        with act2:
            st.button(
                "📷  Recognize a face",
                key="cta_recognize",
                on_click=go_to,
                args=("Recognize",),
                use_container_width=True
            )

    with right:
        st.markdown(face_mesh_svg(), unsafe_allow_html=True)

    st.write("")

    features = [
        ("👥", "Fast & accurate", "Powered by YuNet detection and SFace embeddings"),
        ("🎯", "Easy to use", "Simple, guided screens for every step"),
        ("🛡️", "Private by design", "Your images never leave this machine"),
        ("⚡", "Real time", "Live webcam recognition with instant labels"),
    ]

    cols = st.columns(4, gap="medium")

    for col, (icon, title, desc) in zip(cols, features):
        with col:
            st.markdown(f"""
            <div class="feature">
                <span class="ic">{icon}</span>
                <div class="ft">{title}</div>
                <div class="fd">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")

    st.info(
        "Use images only with the person's consent. This is a local "
        "demonstration, not a secure authentication system."
    )


# =================================
# ENROLLMENT
# =================================
elif page == "Enroll Face":

    st.markdown("""
    <div class="panel-title">Enroll a person</div>
    <div class="panel-sub">
        Add several photos of the same person to improve recognition
        across different lighting and appearances.
    </div>
    """, unsafe_allow_html=True)

    person_name = st.text_input(
        "Person's name",
        key="person_name",
        placeholder="e.g. Khushi"
    )

    enroll_source = st.radio(
        "Enrollment method",
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
            st.error("Could not read this image. Try a JPG or PNG file.")

        else:
            preview, _ = st.columns([1, 1])

            with preview:
                st.image(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    caption="Enrollment photo",
                    use_container_width=True
                )

            if st.button("Check & enroll", key="enroll_button"):

                name = person_name.strip()

                if not name:
                    st.warning("Enter a name before enrolling.")

                else:
                    faces = detector.detect_faces(image)

                    if len(faces) == 0:
                        st.error(
                            "No face detected. Use a brighter, "
                            "front-facing image."
                        )

                    elif len(faces) > 1:
                        st.warning(
                            "Multiple faces detected. Use an image "
                            "containing only one person."
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
                            st.error(f"Could not process face: {error}")

    # --------------------------------
    # Duplicate warning and confirmation
    # --------------------------------
    pending = st.session_state.get("pending_enrollment")

    if pending is not None:

        st.divider()
        st.markdown(
            '<div class="panel-title">Confirm enrollment</div>',
            unsafe_allow_html=True
        )

        if pending["duplicate_name"] is not None:

            st.warning(
                "This face looks similar to an existing enrollment "
                "under another name."
            )

            st.write(f"Possible match: **{pending['duplicate_name']}**")

            st.write(
                f"Similarity score: {pending['duplicate_score']:.4f}"
            )

            st.caption(
                "Similarity is not proof of identity. Compare the "
                "images before continuing."
            )

            st.write(
                "If this is genuinely a different person, "
                "confirm to enroll them."
            )

        else:
            st.info(
                "No possible duplicate found. Confirm to save "
                "this enrollment."
            )

        col1, col2, _ = st.columns([1, 1, 2])

        with col1:
            if st.button("Confirm enrollment", key="confirm_enrollment"):

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

                    st.session_state.pop("pending_enrollment", None)

                    st.success(f"{pending['name']} enrolled.")

                    st.write("Enrollment photo saved:")
                    st.code(str(saved_path.relative_to(BASE_DIR)))

                except Exception as error:
                    st.error(f"Enrollment failed: {error}")

        with col2:
            if st.button(
                "Cancel",
                key="cancel_enrollment",
                type="secondary"
            ):
                st.session_state.pop("pending_enrollment", None)
                st.info("Enrollment cancelled.")


# =================================
# RECOGNITION
# =================================
elif page == "Recognize":

    st.markdown("""
    <div class="panel-title">Recognize a person</div>
    <div class="panel-sub">
        Upload or capture an image and match every detected face
        against your enrolled identities.
    </div>
    """, unsafe_allow_html=True)

    recognize_source = st.radio(
        "Recognition method",
        ["Upload Image", "Live Camera"],
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

        image = read_image(recognize_image)

        if image is None:

            st.error("Could not read this image. Try a JPG or PNG file.")

        else:

            in_col, out_col = st.columns(2, gap="large")

            with in_col:
                st.image(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    caption="Input image",
                    use_container_width=True
                )

            if st.button("Recognize face", key="recognize_button"):

                results = get_faces_and_embeddings(image)

                if len(results) == 0:

                    st.error("No face detected. Try another image.")

                else:

                    display_image = image.copy()

                    enrolled_faces = database.get_all()

                    if not enrolled_faces:

                        st.warning(
                            "No one is enrolled yet. Add a person "
                            "from the Enroll Face screen first."
                        )

                    else:

                        for face, embedding in results:

                            name, score = matcher.match(
                                embedding,
                                enrolled_faces
                            )

                            x, y, w, h = face[:4].astype(int)

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
                                (x, max(y - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                color,
                                2
                            )

                            st.markdown(f"### Result: {name}")

                            st.write(f"Cosine similarity: {score:.4f}")

                            if name == "Unknown":

                                st.warning(
                                    "No enrolled identity met the "
                                    "matching threshold."
                                )

                            else:

                                st.success("Matching identity found.")

                    # Display final annotated image
                    with out_col:
                        st.image(
                            cv2.cvtColor(
                                display_image,
                                cv2.COLOR_BGR2RGB
                            ),
                            caption="Recognition result",
                            use_container_width=True
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

        return frame.from_ndarray(image, format="bgr24")


if page == "Live Recognition":

    st.markdown("""
    <div class="panel-title">Real-time recognition</div>
    <div class="panel-sub">
        Start your webcam to label enrolled people as they appear,
        and mark unregistered faces as Unknown.
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "Allow camera access when your browser asks. Use this "
        "feature only with people's consent."
    )

    stream_col, _ = st.columns([1.4, 1])

    with stream_col:
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
                    {"urls": ["stun:stun.l.google.com:19302"]}
                ]
            },

            media_stream_constraints={
                "video": True,
                "audio": False
            },

            async_processing=True
        )


# =================================
# DATABASE
# =================================
elif page == "Database":

    st.markdown(
        '<div class="panel-title">Enrolled individuals</div>',
        unsafe_allow_html=True
    )

    people = database.get_all()

    if people:

        total_samples = sum(len(v) for v in people.values())

        st.markdown(
            f'<div class="panel-sub">{len(people)} people · '
            f'{total_samples} face samples stored</div>',
            unsafe_allow_html=True
        )

        for name, embeddings in people.items():

            with st.expander(f"👤  {name}"):

                st.write(f"Face samples: {len(embeddings)}")

                if st.button(
                    f"🗑️  Delete {name}",
                    key=f"delete_{name}",
                    type="secondary"
                ):
                    st.session_state["delete_person"] = name

        # --------------------------------
        # Deletion confirmation
        # --------------------------------
        person_to_delete = st.session_state.get("delete_person")

        if person_to_delete in people:

            st.divider()

            st.warning(f"Delete {person_to_delete}?")

            st.caption(
                "This removes their face embeddings and saved "
                "enrollment photos. It cannot be undone."
            )

            col1, col2, _ = st.columns([1, 1, 2])

            with col1:
                if st.button("Yes, delete", key="confirm_delete"):

                    # Remove embeddings from database
                    database.remove(person_to_delete)

                    # Remove saved enrollment photos
                    safe_name = re.sub(
                        r"[^a-zA-Z0-9_-]",
                        "_",
                        person_to_delete.strip()
                    )

                    person_dir = ENROLLED_IMAGES_DIR / safe_name

                    if person_dir.exists():
                        shutil.rmtree(person_dir)

                    st.session_state.pop("delete_person", None)

                    st.success(f"{person_to_delete} deleted.")

                    st.rerun()

            with col2:
                if st.button(
                    "Cancel",
                    key="cancel_delete",
                    type="secondary"
                ):
                    st.session_state.pop("delete_person", None)
                    st.rerun()

    else:

        st.markdown(
            '<div class="panel-sub">Nobody is enrolled yet.</div>',
            unsafe_allow_html=True
        )

        st.button(
            "👤  Enroll the first person",
            key="empty_enroll",
            on_click=go_to,
            args=("Enroll Face",)
        )


# =================================
# ABOUT
# =================================
elif page == "About":

    st.markdown("""
    <div class="panel-title">About FaceID</div>
    <div class="panel-sub">
        A local face recognition demo built with OpenCV and Streamlit.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
**How it works**

1. YuNet finds faces in an image or video frame.
2. SFace turns each face into a 128-dimension embedding.
3. Cosine similarity compares that embedding against enrolled
   identities; anything below the 0.45 threshold is labelled Unknown.

**Where data lives**

Embeddings stay in the local face database and enrollment photos are
written to `data/enrolled/<name>/`. Nothing is uploaded anywhere.

**Limits**

Similarity is not proof of identity. Treat this as a demonstration
of the pipeline, not a secure authentication system, and only use
images people have agreed to share.
    """)