# Chandrayaan-2 Lunar Image Registration Prototype (SIH 2026)

> **Prototype for the SIH 2026 problem statement listed on the SIH portal:** “Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images.”

This repository contains an early MVP baseline built to explore that problem statement. It demonstrates a classical computer-vision registration pipeline; it is a prototype, not a finished or officially validated SIH solution.

This prototype implements a baseline MVP (minimum viable product) for automatic registration of lunar orbital imagery. It establishes a robust computer vision pipeline (Preprocessing + Feature Extraction + Feature Matching + RANSAC Outlier Rejection + Geometric Warping + RMSE Metric Evaluation) with an interactive Streamlit Web UI and a clean modular architecture designed for deep-learning upgrades.

---

## 1. What Problem This MVP Solves

Orbital sensors aboard Chandrayaan-2 capture images of the lunar surface at different orbits, times, sun angles (illumination conditions), and viewing geometry. Before multi-temporal analysis, change detection, or crater mapping can occur, the images must be accurately aligned to a common spatial coordinate system.

This MVP automatically finds spatial correspondences between two lunar images, rejects unreliable matches, calculates the geometric transformation (Homography or Affine), warps the moving image to align with the fixed reference image, and provides visual and quantitative metrics to verify alignment quality.

---

## 2. Reference vs. Source Images

- **Reference (Fixed) Image**: The anchor image whose spatial coordinate system, pixel grid, and resolution are preserved.
- **Source (Moving) Image**: The target image that contains geometric distortions, rotation, scaling, or perspective tilt. It is warped and resampled to match the coordinate grid of the Reference image.

---

## 3. Pipeline Architecture

```
                                  [ User Input ]
                         Reference Image   Source Image
                                │               │
                                ▼               ▼
                      ┌──────────────────────────────────┐
                      │    1. Image Preprocessing        │
                      │ (Grayscale, CLAHE, Filtering)   │
                      └──────────────────────────────────┘
                                │               │
                                ▼               ▼
                      ┌──────────────────────────────────┐
                      │   2. Feature Detection           │
                      │ (SIFT / ORB / AKAZE Extractor)   │
                      └──────────────────────────────────┘
                                │               │
                                ▼               ▼
                      ┌──────────────────────────────────┐
                      │   3. Feature Matching            │
                      │ (FLANN / BFMatcher + Ratio Test) │
                      └──────────────────────────────────┘
                                        │
                                        ▼
                      ┌──────────────────────────────────┐
                      │  4. Geometric Verification       │
                      │ (RANSAC Homography / Affine)     │
                      └──────────────────────────────────┘
                                        │
                                        ▼
                      ┌──────────────────────────────────┐
                      │   5. Image Warping & Metrics     │
                      │ (WarpPerspective + RMSE Calc)    │
                      └──────────────────────────────────┘
                                        │
                                        ▼
                      ┌──────────────────────────────────┐
                      │     6. UI & Visualization        │
                      │ (Side-by-side, Blend, Anaglyph) │
                      └──────────────────────────────────┘
```

---

## 4. Algorithms Used in MVP

1. **Preprocessing**: Contrast Limited Adaptive Histogram Equalization (CLAHE) enhances crater shadows and terrain texture under contrasting solar illumination angles.
2. **Feature Detection & Description**:
   - **SIFT** (Scale-Invariant Feature Transform): Primary baseline invariant to scale and rotation.
   - **ORB / AKAZE**: Binary alternatives for fast processing.
3. **Correspondence Matching**:
   - **FLANN** (Fast Library for Approximate Nearest Neighbors): Nearest-neighbor search in descriptor space.
   - **Lowe's Ratio Test**: Filters ambiguous matches where distance to 1st nearest neighbor is not significantly smaller than 2nd nearest neighbor ($d_1 < 0.75 \times d_2$).
4. **Geometric Verification & Outlier Rejection**:
   - **RANSAC / MAGSAC**: Estimates $3 \times 3$ Homography matrix $H$ or $2 \times 3$ Affine matrix $M$ while identifying and rejecting outlier correspondences.
   - **Degeneracy Detection**: Validates condition number, matrix determinant, and scale preservation to prevent mirror flipping or singular collapses.
5. **Image Warping**: Bilinear perspective warping (`cv2.warpPerspective`) maps the Source image onto the Reference coordinate frame.

---

## 5. Installation & Setup

### Prerequisites
- Python 3.10 or higher
- `pip` package manager

### Installation Steps

1. Clone or navigate into the project root directory:
   ```bash
   cd lunar_registration
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 6. How to Run the Application

### Option A: Launch Interactive Streamlit Web UI

Run the main application script:
```bash
python app/main.py --ui
```
or directly via Streamlit:
```bash
streamlit run ui/streamlit_app.py
```
Open your browser at `http://localhost:8501`.

**Features in UI**:
- Upload your own Reference & Source images (PNG, JPG, TIFF).
- Or click **Fetch NASA Multi-Angle Test Pair** in the sidebar to load a public LROC stereo pair showing the same terrain from different viewing angles.
- Adjust Lowe ratio, RANSAC thresholds, CLAHE enhancement toggles.
- Inspect KPI metric cards: Total Matches, Inliers, Inlier Ratio %, Reprojection RMSE (px), Runtime (ms).
- Visual tabs:
  - Raw Matches vs. RANSAC Inliers.
  - Reference, Registered Source, and Absolute Difference Heatmap.
  - Interactive Alpha Blend slider & Red-Cyan Anaglyph toggle.

### Option B: Headless Command Line (CLI) Batch Run


Run registration headlessly from command line:
```bash
python app/main.py --ref path/to/reference_image.png --src path/to/source_image.png --out outputs/
```

The test pair is downloaded on demand from the NASA/USGS LROC archive; it is not stored in this repository.

Run unit tests:
```bash
python -m unittest discover -s tests
```

---

## 7. Mathematical Formulations (RMSE & Inlier Ratio)

### Reprojection RMSE (Root Mean Square Error)
For each RANSAC inlier pair consisting of source point $p_s = (x_s, y_s)$ and reference point $p_r = (x_r, y_r)$:

1. Transform $p_s$ using estimated Homography $H$:
   $$\begin{bmatrix} X' \\ Y' \\ Z' \end{bmatrix} = H \cdot \begin{bmatrix} x_s \\ y_s \\ 1 \end{bmatrix}$$
2. Convert from homogeneous coordinates to 2D projected point $\hat{p}_r$:
   $$\hat{x}_r = \frac{X'}{Z'}, \quad \hat{y}_r = \frac{Y'}{Z'}$$
3. Euclidean reprojection error distance for inlier point $i$:
   $$e_i = \sqrt{(\hat{x}_r^{(i)} - x_r^{(i)})^2 + (\hat{y}_r^{(i)} - y_r^{(i)})^2}$$
4. Overall Reprojection RMSE (reported in pixels):
   $$\text{RMSE} = \sqrt{\frac{1}{N_{\text{inliers}}} \sum_{i=1}^{N_{\text{inliers}}} e_i^2}$$

### Inlier Ratio
$$\text{Inlier Ratio} = \frac{N_{\text{inliers}}}{N_{\text{total\_good\_matches}}}$$
where $N_{\text{total\_good\_matches}}$ is the count of matches passing Lowe's ratio test.

---

## 8. Known Limitations of the Baseline MVP

1. **Extreme Sun-Angle Shift Sensitivity**: Classical SIFT gradient descriptors degrade when shadow orientation shifts by >60 degrees due to inverted light-dark crater gradients.
2. **Terrain Homogeneity**: Featureless lunar mare regions yield sparse keypoints compared to crater-dense highlands.
3. **Sub-pixel Precision Limit**: Pixel-grid feature detectors operate on integer keypoint coordinates before RANSAC fitting, limiting baseline accuracy to ~0.4–0.8 pixels.
4. **Planar Homography Assumption**: Assumes locally planar terrain or small elevation relief relative to orbital altitude.

---

## 9. Why This Is an MVP & Roadmap for Final SIH Solution

This MVP provides the fundamental verification baseline. To satisfy the complete SIH 2026 challenge requirements, the codebase is modularly structured so the following deep-learning and remote sensing upgrades can be plugged in:

### Deep Learning Extensions (`core/`)
- **Replace SIFT**: Plug in learned feature extractors (**SuperPoint**, **DISK**, **ALIKE**) that extract illumination-invariant keypoints trained on satellite imagery.
- **Replace FLANN**: Plug in learned correspondence matchers (**LightGlue**, **SuperGlue**) that perform graph-neural-network attention over spatial context.
- **Illumination Normalization**: Add multi-scale Phase Congruency or Shadow-Invariant Color Transforms in `core/preprocessing.py`.
- **Spatially Uniform Match Selection**: Implement quadtree or grid-based bucketing to enforce uniform correspondence distribution across featureless terrain.
- **Sub-pixel Refinement**: Integrate patch-based Lucas-Kanade gradient optimization or deep optical flow for sub-pixel correspondence alignment.
- **Advanced Transformation Models**: Support Thin Plate Splines (TPS) and rational polynomial coefficients (RPC) for non-planar lunar topography.
