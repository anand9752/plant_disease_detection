# Model Deployment with Git LFS

## Overview
This project now uses **Git LFS (Large File Storage)** to handle the optimized Keras model (182MB) for deployment.

## Current Model Configuration

### Primary Model
- **File**: `compressed_models/optimized_model.h5`
- **Size**: 182MB (191MB on disk)
- **Type**: Optimized Keras model
- **Storage**: Git LFS
- **Priority**: 1st choice for deployment

### Model Loading Priority Order
1. ✅ **Optimized Keras** (`compressed_models/optimized_model.h5`) - Primary choice
2. 🔄 **Pruned Keras** (`compressed_models/pruned_model.h5`) - Fallback
3. 🔄 **TensorFlow Lite Dynamic** (`compressed_models/dynamic_quantized_model.tflite`) - Fallback
4. 🔄 **TensorFlow Lite INT8** (`compressed_models/quantized_model.tflite`) - Fallback
5. 🔄 **Original Keras** (`trained_model/plant_disease_prediction_model.h5`) - Last resort

## Git LFS Configuration

### Files Tracked by LFS
```
*.h5 filter=lfs diff=lfs merge=lfs -text
*.tflite filter=lfs diff=lfs merge=lfs -text
*.pkl filter=lfs diff=lfs merge=lfs -text
*.joblib filter=lfs diff=lfs merge=lfs -text
```

### Benefits
- ✅ Handles large model files (>100MB)
- ✅ Compatible with Streamlit Cloud
- ✅ Maintains Git repository performance
- ✅ Automatic download during deployment

## Streamlit Cloud Deployment

### What Happens During Deployment
1. **Git Clone**: Repository is cloned normally
2. **LFS Download**: Git LFS automatically downloads the 182MB model
3. **Model Loading**: App prioritizes `optimized_model.h5`
4. **Validation**: Model is tested before use
5. **Ready**: App serves predictions using the optimized model

### Expected Deployment Message
```
✅ Found model: Optimized Keras (Primary) (182.0MB)
🔄 Loading model: Optimized Keras (Primary)
✅ Model validation passed
🤖 Using Optimized Keras (Primary) model
```

## Local Development

### Clone Repository with LFS
```bash
git clone https://github.com/anand9752/plant_disease_detection.git
cd plant_disease_detection
git lfs pull  # Download LFS files
```

### Commands
```bash
# Check LFS files
git lfs ls-files

# Check LFS status
git lfs status

# Track new large files
git lfs track "*.h5"
```

## Troubleshooting

### If Model Fails to Load
- App automatically falls back to alternative models
- Random prediction mode as last resort
- Clear error messages in Streamlit interface

### LFS Issues
- Ensure Git LFS is installed: `git lfs install`
- Check file tracking: `git lfs ls-files`
- Verify .gitattributes configuration

---
*Last updated: September 20, 2025*