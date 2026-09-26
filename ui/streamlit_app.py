"""
Streamlit Web Interface for Lunar Image Registration MVP.
SIH 2026 Problem Statement: Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images.
"""

import os
import sys
import time
import urllib.error
import urllib.request
import numpy as np
import cv2
from PIL import Image
import streamlit as st

# Add project root to Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.preprocessing import ImagePreprocessor, PreprocessingError
from core.feature_detection import SIFTFeatureDetector, ORBFeatureDetector, AKAZEFeatureDetector
from core.feature_matching import FLANNMatcher, BFMatcher
from core.geometric_verification import RANSACVerifier
from core.registration import ImageRegistrar
NASA_STEREO_PAIR_URL = (
    "https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/"
    "LROLRC_2001/EXTRAS/ANAGLYPH/NAC_M1181613435_M1181606332/"
    "NAC_ANAGLYPH_M1181613435_M1181606332.TIF"
)


@st.cache_data(show_spinner=False, ttl=86400)
def fetch_nasa_stereo_pair():
    """Fetch and split the NASA LROC stereo product into two viewing-angle images."""
    request = urllib.request.Request(
        NASA_STEREO_PAIR_URL,
        headers={"User-Agent": "LunarImageRegistrationPrototype/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        encoded = np.frombuffer(response.read(), dtype=np.uint8)

    anaglyph = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if anaglyph is None:
        raise ValueError("NASA's stereo image could not be decoded.")

    # The LROC product stores one view in red and the other in green/blue.
    reference = anaglyph[:, :, 1]
    source = anaglyph[:, :, 2]
    max_dimension = max(reference.shape)
    if max_dimension > 2200:
        scale = 2200 / max_dimension
        size = (round(reference.shape[1] * scale), round(reference.shape[0] * scale))
        reference = cv2.resize(reference, size, interpolation=cv2.INTER_AREA)
        source = cv2.resize(source, size, interpolation=cv2.INTER_AREA)

    return reference, source


def main():
    st.set_page_config(
        page_title="Chandrayaan-2 Lunar Image Registration System",
        page_icon="🌙",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom Styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1E293B;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #475569;
            margin-bottom: 1.5rem;
        }
        .stButton>button {
            width: 100%;
            background-color: #2563EB;
            color: white;
            font-size: 1.1rem;
            font-weight: 600;
            padding: 0.6rem 1rem;
            border-radius: 8px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #1D4ED8;
            color: white;
        }
        .metric-card {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title & Subtitle Banner
    st.markdown('<div class="main-header">🌙 Chandrayaan-2 Lunar Image Registration MVP</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">SIH 2026 Problem Statement: Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images</div>',
        unsafe_allow_html=True
    )

    # Sidebar Options & Hyperparameters
    st.sidebar.header("⚙️ Registration Pipeline Settings")

    detector_choice = st.sidebar.selectbox(
        "Feature Detector",
        ["SIFT (Scale-Invariant Baseline)", "ORB (Fast Binary)", "AKAZE (Non-Linear Scale)"],
        index=0
    )

    matcher_choice = st.sidebar.selectbox(
        "Feature Matcher",
        ["FLANN (Fast Approximate Nearest Neighbor)", "BFMatcher (Brute-Force)"],
        index=0
    )

    transform_choice = st.sidebar.selectbox(
        "Geometric Model",
        ["Homography (Perspective 3x3)", "Affine (Partial 2x3)"],
        index=0
    )

    st.sidebar.subheader("Hyperparameters")
    ratio_thresh = st.sidebar.slider("Lowe Ratio Threshold", min_value=0.50, max_value=0.95, value=0.75, step=0.05,
                                    help="Ratio test threshold for filtering ambiguous matches.")

    ransac_thresh = st.sidebar.slider("RANSAC Reprojection Threshold (px)", min_value=0.5, max_value=10.0, value=3.0, step=0.5,
                                     help="Maximum allowed reprojection error in pixels for RANSAC inliers.")

    st.sidebar.subheader("Preprocessing")
    use_clahe = st.sidebar.checkbox("Enable CLAHE (Sun-angle illumination enhancement)", value=True)
    use_blur = st.sidebar.checkbox("Enable Gaussian Blur (Noise suppression)", value=False)

    # Session State Initialization for Images
    if "ref_img_data" not in st.session_state:
        st.session_state.ref_img_data = None
    if "src_img_data" not in st.session_state:
        st.session_state.src_img_data = None


    st.sidebar.subheader("Test Pair")
    st.sidebar.caption("Fetch a real NASA LROC stereo pair of the same terrain viewed from different angles.")
    if st.sidebar.button("Fetch NASA Multi-Angle Test Pair"):
        try:
            with st.spinner("Fetching NASA stereo images..."):
                reference, source = fetch_nasa_stereo_pair()
                st.session_state.ref_img_data = Image.fromarray(reference)
                st.session_state.src_img_data = Image.fromarray(source)
            st.sidebar.success("NASA stereo pair loaded. Click Register Images to run it.")
        except (OSError, urllib.error.URLError, TimeoutError, ValueError) as exc:
            st.sidebar.error(f"Could not fetch the NASA test pair: {exc}")

    # Two-Column Input Section
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("1. Reference / Fixed Image")
        ref_file = st.file_uploader("Upload Reference Lunar Image", type=["png", "jpg", "jpeg", "tif", "tiff"], key="ref_uploader")
        if ref_file is not None:
            try:
                st.session_state.ref_img_data = Image.open(ref_file)
            except Exception as e:
                st.error(f"Error loading Reference image: {str(e)}")

        if st.session_state.ref_img_data is not None:
            st.image(st.session_state.ref_img_data, caption="Reference / Fixed Image", use_container_width=True)

    with col_right:
        st.subheader("2. Source / Moving Image")
        src_file = st.file_uploader("Upload Source Lunar Image", type=["png", "jpg", "jpeg", "tif", "tiff"], key="src_uploader")
        if src_file is not None:
            try:
                st.session_state.src_img_data = Image.open(src_file)
            except Exception as e:
                st.error(f"Error loading Source image: {str(e)}")

        if st.session_state.src_img_data is not None:
            st.image(st.session_state.src_img_data, caption="Source / Moving Image", use_container_width=True)

    # Center Action Button
    st.markdown("<br>", unsafe_allow_html=True)
    btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
    with btn_col2:
        run_registration = st.button("⚡ Register Images", key="run_btn")

    if run_registration:
        if st.session_state.ref_img_data is None or st.session_state.src_img_data is None:
            st.warning("Please upload both images or fetch the NASA multi-angle test pair from the sidebar.")
            return

        with st.spinner("Executing Image Registration Pipeline..."):
            # Instantiate modular core components based on UI selection
            preproc = ImagePreprocessor(use_clahe=use_clahe, use_blur=use_blur)

            if "SIFT" in detector_choice:
                det = SIFTFeatureDetector()
            elif "ORB" in detector_choice:
                det = ORBFeatureDetector()
            else:
                det = AKAZEFeatureDetector()

            if "FLANN" in matcher_choice:
                mat = FLANNMatcher(ratio_threshold=ratio_thresh)
            else:
                mat = BFMatcher(ratio_threshold=ratio_thresh)

            model_type = "homography" if "Homography" in transform_choice else "affine"
            ver = RANSACVerifier(transform_type=model_type, ransac_reproj_threshold=ransac_thresh)

            registrar = ImageRegistrar(
                detector=det,
                matcher=mat,
                verifier=ver,
                preprocessor=preproc
            )

            # Execute pipeline
            result = registrar.register(st.session_state.ref_img_data, st.session_state.src_img_data)
            metrics = result['metrics']

        st.markdown("---")
        st.header("📊 Registration Results & Metrics")

        # Display Clear Status Banner
        if metrics.success:
            st.success(f"✅ **Registration Successful!** {metrics.status_message}")
        else:
            st.error(f"❌ **{metrics.status_message}**")

        # Metrics Panel Cards
        m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
        m_col1.metric("Total Matches", f"{metrics.total_matches}")
        m_col2.metric("Inlier Matches", f"{metrics.inlier_matches}")
        m_col3.metric("Inlier Ratio", f"{metrics.inlier_ratio * 100:.1f}%")
        m_col4.metric("RMSE Error", f"{metrics.rmse_pixels:.3f} px" if metrics.success else "N/A")
        m_col5.metric("Transform Model", f"{metrics.transformation_type}")
        m_col6.metric("Runtime", f"{metrics.execution_time_ms:.1f} ms")

        st.markdown("<br>", unsafe_allow_html=True)

        # Result Visualizations Section
        tab1, tab2, tab3 = st.tabs([
            "🔍 Feature Correspondences & Inliers",
            "🖼️ Registered Image & Differences",
            "🔀 Interactive Overlay & Blink Inspection"
        ])

        with tab1:
            st.subheader("Feature Matches and RANSAC Outlier Rejection")
            c1, c2 = st.columns(2)
            with c1:
                st.write("#### 1. All Initial Matches (Passing Lowe Ratio Test)")
                if result.get('all_matches_vis') is not None:
                    st.image(cv2.cvtColor(result['all_matches_vis'], cv2.COLOR_BGR2RGB), use_container_width=True)
            with c2:
                st.write("#### 2. Geometrically Verified Inliers (RANSAC)")
                if result.get('inlier_matches_vis') is not None:
                    st.image(cv2.cvtColor(result['inlier_matches_vis'], cv2.COLOR_BGR2RGB), use_container_width=True)

        with tab2:
            st.subheader("Alignment Verification")
            if metrics.success and result.get('registered_source') is not None:
                r1, r2, r3 = st.columns(3)
                with r1:
                    st.write("#### Reference Image (Fixed)")
                    st.image(cv2.cvtColor(result['reference_image'], cv2.COLOR_BGR2RGB), use_container_width=True)
                with r2:
                    st.write("#### Registered Source Image (Warped)")
                    st.image(cv2.cvtColor(result['registered_source'], cv2.COLOR_BGR2RGB), use_container_width=True)
                with r3:
                    st.write("#### Absolute Difference Heatmap")
                    st.image(cv2.cvtColor(result['difference_map'], cv2.COLOR_BGR2RGB), use_container_width=True,
                             caption="Darker = Perfect Alignment; Brighter = Disparities")
            else:
                st.warning("Registration did not produce a valid warped image due to insufficient inlier matches or degenerate homography.")

        with tab3:
            st.subheader("Visual Overlay & Blink Comparison")
            if metrics.success and result.get('registered_source') is not None:
                o_col1, o_col2 = st.columns([1, 1])

                with o_col1:
                    st.write("#### Interactive Alpha Blending")
                    alpha_val = st.slider("Blend Opacity (Reference <---> Registered Source)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
                    custom_blend = registrar.create_blended_overlay(result['reference_image'], result['registered_source'], alpha=alpha_val)
                    st.image(cv2.cvtColor(custom_blend, cv2.COLOR_BGR2RGB), use_container_width=True, caption=f"Alpha = {alpha_val:.2f}")

                with o_col2:
                    st.write("#### False-Color Red-Cyan Anaglyph")
                    st.image(cv2.cvtColor(result['anaglyph_overlay'], cv2.COLOR_BGR2RGB), use_container_width=True,
                             caption="Red = Reference, Cyan = Registered Source. Perfectly aligned regions appear grey/white.")

                st.markdown("---")
                st.write("#### Interactive Blink Toggle Inspection")
                blink_state = st.radio("Select View to Toggle:", ["Reference Image", "Registered Source Image"], horizontal=True)
                if blink_state == "Reference Image":
                    st.image(cv2.cvtColor(result['reference_image'], cv2.COLOR_BGR2RGB), use_container_width=True, caption="Reference (Fixed)")
                else:
                    st.image(cv2.cvtColor(result['registered_source'], cv2.COLOR_BGR2RGB), use_container_width=True, caption="Registered Source (Warped)")
            else:
                st.info("Overlay inspection available only after successful registration.")

if __name__ == "__main__":
    main()
