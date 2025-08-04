# Sign Language Recognition System

A real-time sign language recognition system that uses computer vision and machine learning to detect and classify hand gestures for the 26 letters of the alphabet (A-Z). This project leverages MediaPipe for hand landmark detection and Random Forest classification for gesture recognition.

## 🎯 Project Overview

This system captures hand gestures through a webcam, extracts hand landmarks using MediaPipe, and classifies them into the corresponding alphabet letters. The project is designed for educational purposes and demonstrates the integration of computer vision, machine learning, and real-time processing.

## 🏗️ Architecture

The project follows a modular pipeline approach:

1. **Data Collection** (`collect_imgs.py`) - Captures training images
2. **Feature Extraction** (`create_dataset.py`) - Extracts hand landmarks
3. **Model Training** (`train_classifier.py`) - Trains the classification model
4. **Real-time Inference** (`inference_classifier.py`) - Performs live recognition
5. **Hand Landmark Visualization** (`webcame_hand_landmark.py`) - Debugging tool

## 📁 Project Structure

```
sign-lang/
├── collect_imgs.py              # Data collection script
├── create_dataset.py            # Feature extraction script
├── train_classifier.py          # Model training script
├── inference_classifier.py      # Real-time inference
├── webcame_hand_landmark.py    # Hand landmark visualization
├── data/                        # Training data directory
│   ├── 0/                      # Images for letter 'A'
│   ├── 1/                      # Images for letter 'B'
│   └── ...                     # (0-25 for A-Z)
├── data.pickle                 # Extracted features and labels
├── model.p                     # Trained model file
└── README.md                   # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- Webcam
- Good lighting conditions for hand detection

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mr-1o1/CV/tree/main/sign-lang
   cd sign-lang
   ```

2. **Install required dependencies:**
   ```bash
   pip install opencv-python mediapipe scikit-learn matplotlib numpy
   ```

### Usage

#### 1. Data Collection

Collect training data for each letter of the alphabet:

```bash
python collect_imgs.py
```

**Instructions:**
- The script will cycle through letters A-Z (0-25)
- For each letter, press 'Q' to start capturing
- The system will automatically capture 100 images per letter
- Ensure your hand is clearly visible and well-lit
- Use consistent hand positioning for better training results

#### 2. Feature Extraction

Extract hand landmarks from collected images:

```bash
python create_dataset.py
```

This script:
- Processes all images in the `data/` directory
- Extracts 21 hand landmarks per image using MediaPipe
- Saves features and labels to `data.pickle`

#### 3. Model Training

Train the Random Forest classifier:

```bash
python train_classifier.py
```

The script will:
- Split data into training and testing sets (80/20)
- Train a Random Forest classifier
- Display accuracy score
- Save the trained model to `model.p`

#### 4. Real-time Recognition

Run the live sign language recognition:

```bash
python inference_classifier.py
```

**Features:**
- Real-time hand detection and landmark extraction
- Live classification of hand gestures
- Visual bounding box around detected hands
- Display of predicted letter on screen
- Press 'Q' to quit

#### 5. Hand Landmark Visualization (Optional)

For debugging and understanding hand detection:

```bash
python webcame_hand_landmark.py
```

This tool shows:
- Real-time hand landmark detection
- Visual connections between landmarks
- Press 'ESC' to exit

## 🔧 Technical Details

### Hand Landmark Detection

The system uses MediaPipe Hands to detect 21 hand landmarks:
- 4 landmarks per finger (5 fingers × 4 = 20)
- 1 landmark for the wrist
- Each landmark has x, y coordinates (normalized 0-1)

### Feature Vector

For each image, the system extracts:
- 42 features (21 landmarks × 2 coordinates)
- Normalized coordinates for scale invariance

### Machine Learning Model

- **Algorithm:** Random Forest Classifier
- **Features:** 42-dimensional hand landmark vectors
- **Classes:** 26 (A-Z)
- **Data Split:** 80% training, 20% testing
- **Evaluation:** Accuracy score

## 📊 Performance

The system typically achieves:
- **Training Accuracy:** 85-95% (depending on data quality)
- **Real-time Performance:** 15-30 FPS
- **Detection Range:** 0.5-2 meters from camera

## 🎯 Best Practices

### Data Collection Tips

1. **Lighting:** Ensure good, consistent lighting
2. **Background:** Use plain, uncluttered backgrounds
3. **Hand Position:** Keep hands clearly visible and centered
4. **Consistency:** Use similar hand positioning for each letter
5. **Variation:** Collect data from different angles and distances

### Recognition Tips

1. **Distance:** Keep hands 0.5-2 meters from camera
2. **Lighting:** Ensure adequate lighting
3. **Background:** Avoid cluttered backgrounds
4. **Hand Movement:** Minimize rapid movements
5. **Gesture Clarity:** Make clear, distinct hand shapes

## 🔍 Troubleshooting

### Common Issues

1. **No Hand Detection:**
   - Check lighting conditions
   - Ensure hand is clearly visible
   - Try adjusting camera distance

2. **Poor Recognition Accuracy:**
   - Collect more training data
   - Improve data quality (better lighting, clearer gestures)
   - Retrain the model

3. **Low Frame Rate:**
   - Close other applications
   - Reduce camera resolution if needed
   - Check system resources

### Debugging

- Use `webcame_hand_landmark.py` to verify hand detection
- Check console output for error messages
- Verify camera permissions

## 🛠️ Customization

### Adding New Gestures

1. Modify `number_of_classes` in `collect_imgs.py`
2. Update `labels_dict` in `inference_classifier.py`
3. Collect training data for new gestures
4. Retrain the model

### Model Parameters

Adjust in `train_classifier.py`:
- Test split ratio
- Random Forest parameters
- Feature engineering

### Detection Parameters

Modify in scripts:
- `min_detection_confidence`
- `max_num_hands`
- `min_tracking_confidence`

## 📚 Dependencies

- **OpenCV:** Computer vision and image processing
- **MediaPipe:** Hand landmark detection
- **scikit-learn:** Machine learning algorithms
- **NumPy:** Numerical computing
- **Matplotlib:** Data visualization (optional)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- MediaPipe team for hand landmark detection
- OpenCV community for computer vision tools
- scikit-learn team for machine learning algorithms

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review the code comments
- Open an issue on GitHub

---

**Note:** This project is designed for educational purposes. For production use, consider additional validation, error handling, and security measures. 