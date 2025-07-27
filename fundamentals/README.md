# Computer Vision Fundamentals

This directory contains Jupyter notebooks covering the fundamental concepts and operations in computer vision.

## Contents

### `io_image.ipynb`
- **Purpose**: Introduction to image input/output operations
- **Topics Covered**:
  - Loading images using OpenCV and PIL
  - Saving images in different formats
  - Basic image properties (dimensions, channels, data types)
  - Image display and visualization techniques
  - Converting between different image formats

## Utilities

### `temp.py`
A utility module containing helper functions for image visualization:

- `show_imgs_in_grid()`: Dynamic function to display multiple images in an automatically sized grid
- Legacy wrapper functions for 2-6 images with backward compatibility
- Automatic grayscale detection and colormap handling
- Flexible title and figure size options

## Usage

1. Start with `io_image.ipynb` to learn basic image handling
2. Use `temp.py` functions for consistent image visualization across notebooks
3. Check `img_outputs/` directory for generated images and results

## Dependencies

- OpenCV (`cv2`)
- Matplotlib (`matplotlib`)
- NumPy (`numpy`)
- PIL/Pillow (`PIL`)

## Getting Started

```python
# Import utilities
from temp import show_imgs_in_grid, get_img_dim_str

# Load and display images
import cv2
img = cv2.imread('path/to/image.jpg')
show_imgs_in_grid([img], titles=['My Image'])
``` 