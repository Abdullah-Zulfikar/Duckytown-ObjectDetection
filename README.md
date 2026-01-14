# Duckietown Object Detection

A YOLOv8-based object detection system for identifying objects in the Duckietown environment.

## Project Overview

This project uses YOLOv8 (You Only Look Once v8) to perform real-time object detection on Duckietown images. The model is trained to detect various objects present in the Duckietown simulator/environment.

## Dataset

The dataset is organized into three splits:
- **Training Set** (`dataset/train/`): Used for model training
- **Validation Set** (`dataset/val/`): Used for hyperparameter tuning and validation
- **Test Set** (`dataset/test/`): Used for final evaluation

Each split contains:
- `images/`: RGB image files
- `labels/`: YOLO format annotation files

### Dataset Source
- Roboflow Dataset: `Duckietown Object Detection.v1-duckietown-v1.yolov8/`
- Raw annotations: `raw_data/Annotations.csv`

## Models

Pre-trained models are available in the `model/` directory:
- `best.pt`: PyTorch model weights (primary model)
- `best.onnx`: ONNX format model (for deployment/inference)

## Project Structure

```
duckeytown/
├── data.yml                          # Dataset configuration
├── main.ipynb                        # Main analysis notebook
├── Train.ipynb                       # Training notebook
├── model/                            # Trained model weights
│   ├── best.pt
│   └── best.onnx
├── dataset/                          # Processed dataset
│   ├── train/
│   ├── val/
│   └── test/
├── raw_data/                         # Original data
│   ├── Annotations.csv
│   └── images/
├── Duckietown Object Detection.v1-duckietown-v1.yolov8/  # Roboflow dataset
└── runs/                             # Training outputs and predictions
    └── detect/
```

## Setup & Installation

### Requirements
- Python 3.8+
- PyTorch
- YOLOv8
- OpenCV
- Jupyter Notebook

### Installation

```bash
# Clone or setup the project
cd duckeytown

# Install dependencies
pip install torch torchvision torchaudio
pip install ultralytics opencv-python jupyter numpy pandas

# (Optional) For ONNX inference
pip install onnx onnxruntime
```

## Usage

### Training
Open and run `Train.ipynb` to train the YOLOv8 model:
- Loads the dataset from `dataset/`
- Configures training parameters
- Trains the model and saves weights to `model/best.pt`

### Inference/Detection
Open and run `main.ipynb` for:
- Loading pre-trained models
- Running inference on images
- Visualizing detection results
- Post-processing predictions

### Using Pre-trained Model

```python
from ultralytics import YOLO

# Load model
model = YOLO('model/best.pt')

# Run inference
results = model.predict(source='path/to/image.jpg')

# Display results
results[0].show()
```

## Configuration

Dataset configuration is specified in `data.yaml`:
- Dataset paths
- Class names
- Number of classes
- Training/validation/test splits

## Model Performance

The trained YOLOv8 model has been evaluated on the test set. Detailed metrics and visualization results are available in:
- Training logs: `runs/detect/`
- Prediction outputs: `runs/detect/predict/`

## Notes

- Raw annotations are in `raw_data/Annotations.csv` format
- Images are stored with `.txt` metadata files in the raw data directory
- The Roboflow dataset includes pre-split and pre-formatted data ready for training
- Model weights are saved after training completion

## Future Improvements

- Model quantization for deployment
- Real-time inference optimization
- Integration with Duckietown simulator
- Performance benchmarking on different hardware

## Author

Abdullah Zulfiqar

## License

[Specify your license here]
# Duckytown-ObjectDetection
