# Computer Vision Project Repository 🏙️

A comprehensive collection of computer vision implementations, from fundamental concepts to advanced applications. This repository contains educational materials, practical implementations, and real-world projects covering various aspects of computer vision.

## 📁 Project Structure

```
CV/
├── fundamentals/          # Basic CV concepts and operations
│   ├── io_image.ipynb    # Image I/O operations tutorial
│   ├── temp.py          # Visualization utilities
│   ├── img_outputs/     # Generated images and results
│   └── README.md        # Fundamentals documentation
├── Operators/            # Image processing operators
│   ├── sobel_operator.py # Sobel edge detection implementation
│   ├── sobel_filter.ipynb # Interactive Sobel filter tutorial
│   ├── utils.py         # Operator utilities
│   └── images/          # Test images for operators
├── projects/            # Real-world CV applications
│   ├── face_anonimizer.py      # Image-based face anonymization
│   ├── face_anonimizer_video.py # Real-time video face anonymization
│   ├── utils.py         # Project utilities
│   └── color_detection.py # Color detection implementation
├── images/              # Test images and datasets
└── requirements.txt     # Python dependencies
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Virtual environment (recommended)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mr-1o1/CV
   cd CV
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📚 Fundamentals

The `fundamentals/` directory contains essential computer vision concepts and operations.

### Image I/O Operations (`io_image.ipynb`)
- **Purpose**: Introduction to basic image handling
- **Topics Covered**:
  - Loading images with OpenCV and PIL
  - Saving images in different formats
  - Image properties (dimensions, channels, data types)
  - Image display and visualization
  - Format conversion between different image types

## 🔧 Image Processing Operators

The `Operators/` directory contains implementations of fundamental image processing operators.

### Sobel Edge Detection (`sobel_operator.py`)

**Implementation Details:**
- **Purpose**: Edge detection using Sobel operators
- **Algorithm**: Gradient-based edge detection
- **Kernels**: 
  - Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
  - Gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

**Usage:**
```python
from Operators.sobel_operator import sobel_operator, gradient_angle_in_color

# Apply Sobel operator
gradient_magnitude, gradient_angle = sobel_operator(image)

# Visualize gradient angle in color
color_angle = gradient_angle_in_color(gradient_angle, gradient_magnitude)
```

**Features:**
- Gradient magnitude calculation
- Gradient angle computation
- Color-coded angle visualization
- OpenCV-optimized implementation

### Interactive Tutorial (`sobel_filter.ipynb`)
- Interactive Jupyter notebook for learning Sobel edge detection
- Step-by-step implementation guide
- Visual demonstrations and examples

## 🎯 Real-World Projects

The `projects/` directory contains practical computer vision applications.

### Face Anonymization

#### Image-Based Anonymization (`face_anonimizer.py`)
- **Purpose**: Automatically detect and blur faces in images
- **Technology**: MediaPipe Face Detection
- **Features**:
  - High-accuracy face detection
  - Configurable blur intensity
  - Bounding box padding for better coverage
  - Support for multiple faces in single image

**Usage:**
```python
python projects/face_anonimizer.py
```

#### Real-Time Video Anonymization (`face_anonimizer_video.py`)
- **Purpose**: Real-time face anonymization from webcam feed
- **Features**:
  - Live video processing
  - Real-time face detection and blurring
  - Selfie-view display (horizontal flip)
  - ESC key to exit

**Usage:**
```python
python projects/face_anonimizer_video.py
```

### Project Utilities (`utils.py`)
Shared utility functions for project implementations:

```python
from projects.utils import to_rgb, to_bgr, blur_img_segment, get_bbox_values

# Color space conversions
rgb_img = to_rgb(bgr_img)
bgr_img = to_bgr(rgb_img)

# Bounding box processing
x1, y1, w, h = get_bbox_values(bbox, height, width, padding=0.01)

# Image segment blurring
blurred_img = blur_img_segment(img, x1, y1, w, h, ksize=55)
```

## 🛠️ Dependencies

### Core Dependencies
- **OpenCV** (`opencv-python==4.6.0.66`): Computer vision library
- **NumPy** (`numpy<2`): Numerical computing (version constraint for compatibility)
- **MediaPipe** (`mediapipe==0.10.5`): ML pipeline framework
- **Matplotlib** (`matplotlib`): Plotting and visualization
- **IPyMPL** (`ipympl`): Interactive matplotlib in Jupyter

### Development Dependencies
- **Jupyter**: Interactive notebooks
- **IPython**: Enhanced Python shell

## 🔧 Troubleshooting

### NumPy Compatibility Issues
If you encounter NumPy 2.x compatibility issues with OpenCV:
```bash
pip install "numpy<2"
```

### MediaPipe Installation
For MediaPipe installation issues on macOS:
```bash
pip install mediapipe==0.10.5
```

## 📖 Learning Path

### For Beginners
1. Start with `fundamentals/io_image.ipynb` to learn basic image operations
2. Explore `Operators/sobel_filter.ipynb` for edge detection concepts
3. Try the face anonymization projects for practical applications

### For Intermediate Users
1. Study the Sobel operator implementation in `Operators/sobel_operator.py`
2. Examine the utility functions for best practices
3. Modify project parameters to experiment with different settings

### For Advanced Users
1. Extend the face anonymization with additional privacy features
2. Implement additional edge detection operators
3. Create new computer vision applications

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenCV**: For the comprehensive computer vision library
- **MediaPipe**: For the face detection capabilities
- **NumPy**: For numerical computing support
- **Matplotlib**: For visualization tools

---

**Happy Computer Vision Coding! 🚀**
