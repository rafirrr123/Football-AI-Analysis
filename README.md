# ⚽ Football AI Analysis: Real-Time Tactical Pitch Radar & Voronoi Minimap

An end-to-end Computer Vision pipeline that transforms broadcast football match footage into a synchronized 2D tactical minimap and Voronoi space-control visualization.

---

## 🚀 Key Features
- **Player & Ball Detection**: YOLO fine-tuned on football pitch entities.
- **Multi-Object Tracking**: ByteTrack integration for persistent player track IDs across occlusions.
- **Dynamic Team Assignment**: Unsupervised KMeans clustering on jersey RGB histograms (with pre-fit calibration to eliminate frame flicker).
- **Metric Coordinate Projection**: Planar Homography ($3 \times 3$) using pitch keypoint landmarks mapped to FIFA standard pitch dimensions via OpenCV.
- **Tactical Space Control**: Real-time Voronoi diagrams visualizing spatial dominance and passing options.

---

## 🛠️ Pipeline Architecture
1. **Object Detection & Tracking**: Detects players, referees, and the ball using YOLO.
2. **Keypoint Landmark Detection**: Extracts field line intersections.
3. **Planar Homography**: Maps broadcast camera pixel coordinates to real-world metric pitch space.
4. **Voronoi Tessellation**: Computes territory ownership between opposing squads dynamically.

---

## 📦 Setup & Installation

```bash
git clone [https://github.com/rafirrr123/Football-AI-Analysis.git](https://github.com/rafirrr123/Football-AI-Analysis.git)
cd Football-AI-Analysis
pip install -r requirements.txt
