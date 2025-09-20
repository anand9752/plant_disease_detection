# 🌿 Plant Disease Detection System

An advanced deep learning-powered web application for real-time plant disease detection using Convolutional Neural Networks (CNN) and TensorFlow Lite compression techniques.

![Plant Disease Detection](https://img.shields.io/badge/Plant-Disease%20Detection-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30.0-red)
![Python](https://img.shields.io/badge/Python-3.10+-blue)

## 📋 Table of Contents

- [🌟 Features](#-features)
- [🏗️ Architecture](#️-architecture)
- [📊 Model Information](#-model-information)
- [🗂️ Dataset & Classes](#️-dataset--classes)
- [⚙️ Model Compression](#️-model-compression)
- [🚀 Installation](#-installation)
- [💻 Usage](#-usage)
- [📁 Project Structure](#-project-structure)
- [📚 Libraries Used](#-libraries-used)
- [🔬 Technical Details](#-technical-details)
- [📈 Performance](#-performance)
- [🤝 Contributing](#-contributing)

## 🌟 Features

### Core Functionality
- **Real-time Disease Detection**: Upload plant images and get instant disease classification
- **38 Disease Classes**: Comprehensive coverage of common plant diseases across multiple crops
- **Confidence Scoring**: Visual confidence indicators with medal-based ranking system
- **Treatment Recommendations**: Detailed organic and chemical treatment options
- **Prevention Guidelines**: Proactive measures to prevent disease occurrence
- **Emergency Actions**: Critical steps for severe disease cases

### Advanced Features
- **Model Compression**: TensorFlow Lite optimization for efficient deployment
- **Responsive UI**: Modern, mobile-friendly interface with attractive styling
- **Interactive Components**: Expandable sections, tabbed treatment options
- **Severity Assessment**: Color-coded severity indicators (Low/Moderate/High/Severe)
- **Multi-format Support**: Support for various image formats (JPG, PNG, JPEG)
- **Memory Optimization**: Efficient model loading with caching mechanisms

### User Experience
- **Intuitive Interface**: Clean, professional design with gradient backgrounds
- **Visual Feedback**: Loading animations and progress indicators
- **Comprehensive Information**: Detailed disease information including scientific names
- **Cultural Practices**: Traditional and modern farming recommendations
- **Accessibility**: High contrast colors and clear typography

## 🏗️ Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          Frontend Layer                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   Image Upload  │  │  Results Display │  │ Treatment Info  │  │
│  │     Component   │  │    Component     │  │   Component     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                     Streamlit Web App                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Model Manager  │  │ Image Processor │  │ Disease Database│  │
│  │     System      │  │     Module      │  │     Handler     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                       AI/ML Layer                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   CNN Model     │  │  TFLite Model   │  │  Preprocessing  │  │
│  │   (Original)    │  │  (Compressed)   │  │     Pipeline    │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### CNN Model Architecture

The model follows a classic CNN architecture optimized for image classification:

```python
Model Architecture:
├── Input Layer: (224, 224, 3) - RGB Images
├── Conv2D(32, 3x3) + ReLU + MaxPooling(2x2)
├── Conv2D(64, 3x3) + ReLU + MaxPooling(2x2)
├── Flatten Layer
├── Dense(256) + ReLU
└── Dense(38) + Softmax - Output Layer
```

**Model Specifications:**
- **Input Shape**: 224×224×3 (RGB images)
- **Total Parameters**: 47,845,734
- **Trainable Parameters**: 47,845,734
- **Model Size**: ~547 MB (original), ~45.6 MB (compressed)
- **Architecture Type**: Sequential CNN
- **Activation Functions**: ReLU (hidden layers), Softmax (output)

## 📊 Model Information

### Original Model
- **Format**: Keras H5 (.h5)
- **Size**: 547 MB
- **Location**: `trained_model/plant_disease_prediction_model.h5`
- **Download Source**: [Google Drive](https://drive.google.com/file/d/1rKh-IElSdHTqax7XdfSdZTn-r8T_qWPf/view)

### Compressed Model
- **Format**: TensorFlow Lite (.tflite)
- **Size**: 45.6 MB (91.7% size reduction)
- **Location**: `compressed_models/dynamic_quantized_model.tflite`
- **Compression Technique**: Dynamic Range Quantization
- **Performance**: Maintained accuracy with significant size reduction

### Training Dataset
- **Source**: [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- **Total Images**: 54,305 images
- **Image Resolution**: 224×224 pixels
- **Data Split**: Train/Validation/Test
- **Augmentation**: Applied during training for better generalization

## 🗂️ Dataset & Classes

The model is trained to detect **38 different plant diseases** across **14 crop types**:

### Supported Crops & Diseases

| **Crop** | **Diseases** | **Count** |
|----------|--------------|-----------|
| **Apple** | Apple Scab, Black Rot, Cedar Apple Rust, Healthy | 4 |
| **Blueberry** | Healthy | 1 |
| **Cherry** | Powdery Mildew, Healthy | 2 |
| **Corn (Maize)** | Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy | 4 |
| **Grape** | Black Rot, Esca (Black Measles), Leaf Blight, Healthy | 4 |
| **Orange** | Huanglongbing (Citrus Greening) | 1 |
| **Peach** | Bacterial Spot, Healthy | 2 |
| **Pepper (Bell)** | Bacterial Spot, Healthy | 2 |
| **Potato** | Early Blight, Late Blight, Healthy | 3 |
| **Raspberry** | Healthy | 1 |
| **Soybean** | Healthy | 1 |
| **Squash** | Powdery Mildew | 1 |
| **Strawberry** | Leaf Scorch, Healthy | 2 |
| **Tomato** | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy | 10 |

### Complete Class List (38 classes):

<details>
<summary>Click to expand full class list</summary>

1. Apple___Apple_scab
2. Apple___Black_rot
3. Apple___Cedar_apple_rust
4. Apple___healthy
5. Blueberry___healthy
6. Cherry_(including_sour)___Powdery_mildew
7. Cherry_(including_sour)___healthy
8. Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot
9. Corn_(maize)___Common_rust_
10. Corn_(maize)___Northern_Leaf_Blight
11. Corn_(maize)___healthy
12. Grape___Black_rot
13. Grape___Esca_(Black_Measles)
14. Grape___Leaf_blight_(Isariopsis_Leaf_Spot)
15. Grape___healthy
16. Orange___Haunglongbing_(Citrus_greening)
17. Peach___Bacterial_spot
18. Peach___healthy
19. Pepper,_bell___Bacterial_spot
20. Pepper,_bell___healthy
21. Potato___Early_blight
22. Potato___Late_blight
23. Potato___healthy
24. Raspberry___healthy
25. Soybean___healthy
26. Squash___Powdery_mildew
27. Strawberry___Leaf_scorch
28. Strawberry___healthy
29. Tomato___Bacterial_spot
30. Tomato___Early_blight
31. Tomato___Late_blight
32. Tomato___Leaf_Mold
33. Tomato___Septoria_leaf_spot
34. Tomato___Spider_mites Two-spotted_spider_mite
35. Tomato___Target_Spot
36. Tomato___Tomato_Yellow_Leaf_Curl_Virus
37. Tomato___Tomato_mosaic_virus
38. Tomato___healthy

</details>

## ⚙️ Model Compression

### Compression Techniques Used

The project implements **TensorFlow Lite Dynamic Range Quantization** for model optimization:

#### 1. Dynamic Range Quantization
- **Method**: Post-training quantization
- **Weight Precision**: 8-bit integers
- **Activation Precision**: Float32 (during inference)
- **Size Reduction**: 91.7% (547 MB → 45.6 MB)
- **Performance Impact**: Minimal accuracy loss

#### 2. Benefits of Compression
- **Faster Inference**: Reduced memory bandwidth requirements
- **Lower Memory Usage**: Suitable for resource-constrained environments
- **Cloud Deployment Ready**: Optimized for Streamlit Cloud deployment
- **Mobile Compatibility**: Can be deployed on mobile devices

#### 3. Compression Implementation

```python
# Model compression pipeline
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Model size comparison
original_size = 547 MB
compressed_size = 45.6 MB
compression_ratio = 91.7%
```

#### 4. Deployment Strategy
- **Primary**: Compressed TFLite model for production
- **Fallback**: Original H5 model for development
- **Auto-detection**: System automatically selects best available model
- **Memory Management**: Efficient loading with caching mechanisms

## 🚀 Installation

### Prerequisites
- Python 3.10+
- pip package manager
- 4GB+ RAM (recommended)
- Internet connection (for initial model download)

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd plant-disease-prediction-app
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv disease_detection
disease_detection\Scripts\activate

# Linux/Mac
python -m venv disease_detection
source disease_detection/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Model (if needed)
If the trained model is not included, download it from:
- **Original Model**: [Google Drive Link](https://drive.google.com/file/d/1rKh-IElSdHTqax7XdfSdZTn-r8T_qWPf/view)
- Place in `trained_model/` directory as `plant_disease_prediction_model.h5`

### Step 5: Run Application
```bash
# For production deployment (recommended)
streamlit run app.py

# For compressed model development
streamlit run main_compressed.py

# For original model development
streamlit run main.py
```

## 🚀 Streamlit Cloud Deployment

This application is optimized for Streamlit Cloud deployment with the following features:

### Deployment-Ready Features
- **Optimized Model Loading**: Automatic model detection and fallback
- **Memory Management**: Efficient caching and resource management
- **Error Handling**: Comprehensive error handling for cloud environment
- **Mobile-Friendly**: Responsive design for all devices
- **Fast Startup**: Optimized dependencies and lazy loading

### Quick Deploy to Streamlit Cloud

1. **Fork this repository** on GitHub
2. **Go to** [share.streamlit.io](https://share.streamlit.io)
3. **Connect your GitHub** account
4. **Select this repository** and set main file to `app.py`
5. **Click Deploy!** 

### Model Hosting Options

Due to file size limitations, choose one of these model hosting strategies:

#### Option 1: GitHub Releases (Recommended)
- Upload compressed model to GitHub Releases
- Automatic download in the app
- Version controlled and accessible

#### Option 2: External Hosting
- Host models on cloud storage (AWS S3, Google Drive)
- Update download URLs in the app
- Reliable and scalable

#### Option 3: Include in Repository
- For models under 100MB
- Direct inclusion in the repository
- Fastest loading but limited by file size

For detailed deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md).

## 💻 Usage

### Basic Usage

1. **Launch Application**
   ```bash
   streamlit run main_compressed.py
   ```

2. **Upload Image**
   - Click on "Choose an image..." button
   - Select a plant image (JPG, PNG, JPEG)
   - Wait for processing

3. **View Results**
   - Disease classification with confidence score
   - Detailed disease information
   - Treatment recommendations
   - Prevention guidelines

### Advanced Features

#### Model Selection
The application automatically detects and uses the best available model:
1. Compressed TFLite model (preferred)
2. Original H5 model (fallback)

#### Treatment Information
- **Organic Treatments**: Environmentally friendly options
- **Chemical Treatments**: Professional-grade solutions
- **Cultural Practices**: Traditional farming methods
- **Prevention Tips**: Proactive disease management
- **Emergency Actions**: Critical response measures

#### Confidence Interpretation
- **🥇 Gold (90-100%)**: High confidence, reliable prediction
- **🥈 Silver (80-89%)**: Good confidence, likely accurate
- **🥉 Bronze (70-79%)**: Moderate confidence, verify if possible
- **📊 Standard (<70%)**: Low confidence, consider retaking image

## 📁 Project Structure

```
plant-disease-prediction-app/
├── 📄 app.py                           # 🚀 Main deployment file (optimized)
├── 📄 main.py                          # Main application (original model)
├── 📄 main_compressed.py               # Compressed model application
├── 📄 requirements.txt                 # Python dependencies (deployment-ready)
├── 📄 packages.txt                     # System dependencies for Streamlit Cloud
├── 📄 class_indices.json              # Disease class mappings
├── 📄 disease_info_database.json      # Comprehensive disease information
├── 📄 Dockerfile                      # Docker configuration
├── 📄 README.md                       # Project documentation
├── 📄 DEPLOYMENT.md                   # 🚀 Streamlit Cloud deployment guide
├── 📄 .gitignore                      # Git ignore rules
│
├── 📁 .streamlit/
│   ├── 📄 config.toml                 # Streamlit configuration
│   └── 📄 secrets.toml               # Secrets template
│
├── 📁 trained_model/
│   ├── 📄 plant_disease_prediction_model.h5  # Original CNN model
│   └── 📄 model.txt                          # Model download instructions
│
├── 📁 compressed_models/
│   └── 📄 dynamic_quantized_model.tflite     # Compressed TFLite model
│
├── 📁 model_training_notebook/
│   └── 📄 Plant_Disease_Prediction_CNN_Image_Classifier.ipynb
│
└── 📁 disease_detection/               # Virtual environment
    ├── 📁 Scripts/
    ├── 📁 Lib/
    └── 📄 pyvenv.cfg
```

## 📚 Libraries Used

### Core Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| **TensorFlow** | 2.15.0 | Deep learning framework for model inference |
| **Streamlit** | 1.30.0 | Web application framework for UI |
| **NumPy** | 1.26.3 | Numerical computing for image processing |

### Additional Libraries (Auto-installed)

| Library | Purpose |
|---------|---------|
| **Pillow (PIL)** | Image loading and preprocessing |
| **Matplotlib** | Image visualization and plotting |
| **JSON** | Configuration and data management |
| **OS** | File system operations |
| **Base64** | Image encoding for web display |

### Development Libraries

| Library | Purpose |
|---------|---------|
| **Kaggle API** | Dataset downloading |
| **Zipfile** | Data extraction |
| **Random** | Reproducible random operations |

## 🔬 Technical Details

### Image Preprocessing Pipeline

```python
def preprocess_image(image):
    # 1. Resize to model input size
    image = image.resize((224, 224))
    
    # 2. Convert to numpy array
    img_array = np.array(image)
    
    # 3. Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # 4. Normalize pixel values
    img_array = img_array.astype('float32') / 255.0
    
    return img_array
```

### Model Loading Strategy

```python
class ModelManager:
    def find_best_model(self):
        # Priority order:
        # 1. TensorFlow Lite models
        # 2. Optimized Keras models
        # 3. Original model
        
        model_search_order = [
            "dynamic_quantized_model.tflite",
            "quantized_model.tflite", 
            "optimized_model.h5",
            "plant_disease_prediction_model.h5"
        ]
```

### Memory Optimization

- **Lazy Loading**: Models loaded only when needed
- **Caching**: Streamlit cache decorators for efficient reuse
- **Memory Management**: Automatic cleanup of unused resources
- **Error Handling**: Graceful fallback mechanisms

### Security Features

- **Input Validation**: File type and size restrictions
- **Error Handling**: Comprehensive exception management
- **Safe File Processing**: Secure image handling
- **Resource Limits**: Memory and processing constraints

## 📈 Performance

### Model Metrics

| Metric | Original Model | Compressed Model |
|--------|----------------|------------------|
| **File Size** | 547 MB | 45.6 MB |
| **Loading Time** | ~15-20 seconds | ~3-5 seconds |
| **Inference Time** | ~2-3 seconds | ~1-2 seconds |
| **Memory Usage** | ~2GB | ~500MB |
| **Accuracy** | Baseline | ~99% of original |

### System Requirements

#### Minimum Requirements
- **RAM**: 2GB
- **Storage**: 1GB free space
- **CPU**: Dual-core processor
- **Internet**: For initial setup

#### Recommended Requirements
- **RAM**: 4GB+
- **Storage**: 2GB+ free space
- **CPU**: Quad-core processor
- **Internet**: Stable connection

### Deployment Options

1. **Streamlit Cloud**: Optimized deployment with `app.py` (recommended)
2. **Local Development**: Full model with all features using `main.py`
3. **Docker**: Containerized deployment
4. **Mobile**: TensorFlow Lite for mobile apps

### Cloud Deployment Features
- **Auto Model Detection**: Automatically finds and loads the best available model
- **Memory Optimization**: Efficient resource usage for cloud constraints
- **Error Recovery**: Graceful handling of deployment issues
- **Progressive Loading**: Fast initial load with lazy model loading
- **Mobile Responsive**: Works perfectly on all device sizes

## 🤝 Contributing

### Development Setup

1. **Fork the repository**
2. **Create feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Install development dependencies**
   ```bash
   pip install -r requirements-dev.txt
   ```
4. **Make changes and test**
5. **Submit pull request**

### Contribution Guidelines

- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation
- Ensure backward compatibility
- Test with both model formats

### Areas for Contribution

- [ ] Additional crop diseases
- [ ] Model accuracy improvements
- [ ] Mobile app development
- [ ] API development
- [ ] Performance optimizations
- [ ] Multilingual support
- [ ] Real-time video analysis

---

## 📞 Support

For questions, issues, or contributions:

- **Documentation**: Check this README and code comments
- **Issues**: Use GitHub Issues for bug reports
- **Discussions**: Use GitHub Discussions for questions
- **Email**: [Your contact information]

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **PlantVillage Dataset**: For providing comprehensive plant disease data
- **TensorFlow Team**: For the excellent deep learning framework
- **Streamlit Team**: For the intuitive web app framework
- **Kaggle Community**: For dataset hosting and community support
- **Open Source Contributors**: For inspiration and code references

---

<div align="center">

**Made with ❤️ for sustainable agriculture and plant health**

[⬆ Back to Top](#-plant-disease-detection-system)

</div>
