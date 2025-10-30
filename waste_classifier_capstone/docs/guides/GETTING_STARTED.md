# Getting Started with Waste Classification Capstone Project

This guide will help you set up and run the Waste Classification capstone project on your local machine.

---

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Python 3.8 or higher** - The project is built using Python.
- **pip** - Python package installer.
- **Git** - For cloning the repository.
- **Jupyter Notebook** - For running the interactive notebooks.
- **(Optional) CUDA-enabled GPU** - For faster training and inference.

---

## Installation Steps

### Step 1: Clone the Repository

If you received this project as an archive, extract it to your desired location. Otherwise, clone it from GitHub:

```bash
git clone https://github.com/AnHgPham/waste_classifier.git
cd waste_classifier_capstone
```

### Step 2: Create a Virtual Environment

Creating a virtual environment is highly recommended to avoid dependency conflicts:

```bash
# Create virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```

### Step 3: Install Dependencies

Install all required Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

This will install TensorFlow, PyTorch, OpenCV, Ultralytics (YOLOv8), and other necessary libraries.

### Step 4: Download the Dataset

The project uses the **Garbage Classification v2** dataset from Kaggle. You need to download it manually:

1. Go to [Kaggle Dataset Page](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)
2. Download the dataset (you may need to create a Kaggle account)
3. Extract the downloaded archive
4. Copy all class folders (battery, biological, cardboard, etc.) to `data/raw/`

Your `data/raw/` directory should look like this:

```
data/raw/
├── battery/
├── biological/
├── cardboard/
├── clothes/
├── glass/
├── metal/
├── paper/
├── plastic/
├── shoes/
└── trash/
```

---

## Running the Project

The project is organized into 4 weeks, each building upon the previous one. You should follow them sequentially.

### Week 1: Data Preparation and Baseline CNN

Navigate to the Week 1 assignments folder and start Jupyter Notebook:

```bash
cd Week1_Data_and_Baseline/assignments
jupyter notebook
```

Open and run the notebooks in order:

1. **W1_Data_Exploration.ipynb** - Explore and visualize the dataset
2. **W1_Preprocessing.ipynb** - Split data and create data generators
3. **W1_Baseline_CNN.ipynb** - Build and train a baseline CNN model

### Week 2: Transfer Learning with MobileNetV2

Navigate to the Week 2 assignments folder:

```bash
cd ../../Week2_Transfer_Learning/assignments
jupyter notebook
```

Run the notebooks:

1. **W2_Feature_Extraction.ipynb** - Use MobileNetV2 as a feature extractor
2. **W2_Fine_Tuning.ipynb** - Fine-tune the model for better performance

### Week 3: Real-time Detection with YOLOv8

Navigate to the Week 3 assignments folder:

```bash
cd ../../Week3_Realtime_Detection/assignments
jupyter notebook
```

Run the notebook:

1. **W3_Integration.ipynb** - Integrate YOLOv8 with the classifier for real-time detection

**Note:** This week requires a webcam for real-time detection. If you don't have one, you can modify the code to use a video file instead.

### Week 4: Model Optimization and Deployment

Navigate to the Week 4 assignments folder:

```bash
cd ../../Week4_Deployment/assignments
jupyter notebook
```

Run the notebook:

1. **W4_Model_Optimization.ipynb** - Convert and quantize the model for deployment

---

## Project Structure

Understanding the project structure will help you navigate and modify the code:

```
waste_classifier_capstone/
│
├── config.py                        # Central configuration file
├── requirements.txt                 # Python dependencies
├── README.md                        # Project overview
├── GETTING_STARTED.md              # This file
│
├── Week1_Data_and_Baseline/
│   ├── assignments/                 # Jupyter notebooks
│   ├── utils/                       # Helper functions
│   └── slides/                      # Lecture slides (optional)
│
├── Week2_Transfer_Learning/
│   ├── assignments/
│   ├── utils/
│   └── slides/
│
├── Week3_Realtime_Detection/
│   ├── assignments/
│   ├── utils/
│   └── slides/
│
├── Week4_Deployment/
│   ├── assignments/
│   ├── utils/
│   └── slides/
│
├── data/
│   ├── raw/                         # Original dataset (you need to download)
│   └── processed/                   # Preprocessed data (auto-generated)
│
└── outputs/
    ├── models/                      # Saved models
    ├── reports/                     # Evaluation reports
    └── screenshots/                 # Screenshots from real-time detection
```

---

## Configuration

All hyperparameters and settings are centralized in `config.py`. You can modify them to experiment with different configurations:

- **Image size:** `IMG_SIZE = (224, 224)`
- **Batch size:** `BATCH_SIZE = 32`
- **Learning rates:** `LEARNING_RATE_BASELINE`, `LEARNING_RATE_TRANSFER_PHASE1`, etc.
- **Number of epochs:** `EPOCHS_BASELINE`, `EPOCHS_TRANSFER_PHASE1`, etc.
- **Data augmentation:** `USE_AUGMENTATION = True`

---

## Troubleshooting

### Issue: "No module named 'tensorflow'"

**Solution:** Make sure you have activated your virtual environment and installed all dependencies:

```bash
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Issue: "CUDA not found" or "GPU not detected"

**Solution:** This is normal if you don't have an NVIDIA GPU. The project will run on CPU, but training will be slower. If you have a GPU but it's not detected, ensure you have installed the CUDA-enabled version of TensorFlow:

```bash
pip install tensorflow[and-cuda]
```

### Issue: "Dataset not found"

**Solution:** Make sure you have downloaded the dataset from Kaggle and placed it in the `data/raw/` directory as described in Step 4.

### Issue: Webcam not working in Week 3

**Solution:** 
- Check if your webcam is connected and not being used by another application.
- Try changing the camera index in `config.py`: `CAMERA_INDEX = 1` (or 2, 3, etc.)
- Alternatively, modify the code to use a video file instead of a live webcam feed.

---

## Next Steps

After completing all 4 weeks, you will have:

1. A trained waste classification model with 85-92% accuracy
2. A real-time detection system using YOLOv8
3. An optimized model ready for deployment on edge devices
4. A comprehensive understanding of the entire deep learning pipeline

You can now:

- **Deploy the model** to a web application using Flask or FastAPI
- **Integrate with IoT devices** like Raspberry Pi or Jetson Nano
- **Extend the project** by adding more waste categories or improving accuracy
- **Use this as a portfolio project** to showcase your deep learning skills

---

## Support

If you encounter any issues or have questions, please:

1. Check the troubleshooting section above
2. Review the code comments and docstrings
3. Consult the README.md for additional information
4. Open an issue on the GitHub repository

---

**Happy Learning!**

This project is designed to provide a hands-on, comprehensive learning experience in deep learning and computer vision. Take your time with each week, experiment with the code, and don't hesitate to modify and extend it to suit your needs.
