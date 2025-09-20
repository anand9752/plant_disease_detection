import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image
import requests
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="🌿 Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Working directory for model files
working_dir = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_class_indices():
    """Load class indices for disease classification"""
    try:
        with open('class_indices.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Class indices file not found. Please ensure class_indices.json is available.")
        return {}

@st.cache_data
def load_disease_database():
    """Load comprehensive disease information database"""
    try:
        with open('disease_info_database.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning("Disease information database not found. Basic functionality will be available.")
        return {}

class ModelManager:
    """
    Manages loading and inference for different model formats
    """
    def __init__(self):
        self.model = None
        self.model_type = None
        self.model_info = {}
        self.interpreter = None
    
    def find_best_model(self):
        """
        Find the best available model in order of preference:
        1. Compressed models (TFLite, pruned, optimized) - PRIORITY for deployment
        2. Original model (fallback only)
        """
        model_search_order = [
            # Compressed models (preferred for deployment)
            ("compressed_models/dynamic_quantized_model.tflite", "TensorFlow Lite (Dynamic)", "tflite"),
            ("compressed_models/quantized_model.tflite", "TensorFlow Lite (INT8)", "tflite"),
            ("compressed_models/optimized_model.h5", "Optimized Keras", "keras"),
            ("compressed_models/pruned_model.h5", "Pruned Keras", "keras"),
            
            # Original model (fallback - not recommended for deployment)
            ("trained_model/plant_disease_prediction_model.h5", "Original Keras", "keras")
        ]
        
        for model_path, model_name, model_format in model_search_order:
            full_path = os.path.join(working_dir, model_path)
            if os.path.exists(full_path):
                file_size = os.path.getsize(full_path) / (1024*1024)  # Size in MB
                return {
                    'path': full_path,
                    'name': model_name,
                    'format': model_format,
                    'size_mb': file_size,
                    'relative_path': model_path
                }
        
        return None
    
    def load_model(self):
        """Load the best available model"""
        try:
            best_model = self.find_best_model()
            if not best_model:
                st.error("No model files found. Please ensure model files are available.")
                return False
            
            self.model_info = best_model
            
            if best_model['format'] == 'tflite':
                return self._load_tflite_model(best_model['path'])
            else:
                return self._load_keras_model(best_model['path'])
                
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def _load_tflite_model(self, model_path):
        """Load TensorFlow Lite model"""
        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.model_type = "tflite"
            return True
        except Exception as e:
            st.error(f"Error loading TFLite model: {str(e)}")
            return False
    
    def _load_keras_model(self, model_path):
        """Load Keras H5 model"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.model_type = "keras"
            return True
        except Exception as e:
            st.error(f"Error loading Keras model: {str(e)}")
            return False
    
    def predict(self, img_array):
        """Make prediction using the loaded model"""
        try:
            if self.model_type == "tflite":
                return self._predict_tflite(img_array)
            elif self.model_type == "keras":
                return self._predict_keras(img_array)
            else:
                raise Exception("No model loaded")
                
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            return None
    
    def _predict_tflite(self, img_array):
        """Predict using TFLite interpreter"""
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        
        # Set input tensor
        self.interpreter.set_tensor(input_details[0]['index'], img_array)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output_data = self.interpreter.get_tensor(output_details[0]['index'])
        return output_data[0]
    
    def _predict_keras(self, img_array):
        """Predict using Keras model"""
        predictions = self.model.predict(img_array, verbose=0)
        return predictions[0]

@st.cache_resource
def load_model_manager():
    """Load and cache the model manager"""
    manager = ModelManager()
    success = manager.load_model()
    if success:
        return manager
    return None

def preprocess_image(image):
    """Preprocess image for model prediction"""
    try:
        # Resize image to model input size
        image = image.resize((224, 224))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Normalize pixel values
        img_array = img_array.astype('float32') / 255.0
        
        return img_array
        
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_disease(model_manager, img_array, class_indices):
    """Make prediction using the loaded model"""
    try:
        predictions = model_manager.predict(img_array)
        if predictions is None:
            return None, 0, None
        
        # Get predicted class
        predicted_class_index = np.argmax(predictions)
        confidence = float(predictions[predicted_class_index])
        
        # Get class name
        predicted_class = class_indices.get(str(predicted_class_index), "Unknown")
        
        return predicted_class, confidence, predictions
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, 0, None

def display_confidence_score(confidence):
    """Display confidence score with visual indicators"""
    confidence_percent = confidence * 100
    
    if confidence_percent >= 90:
        medal = "🥇"
        color = "#FFD700"
        level = "High Confidence"
    elif confidence_percent >= 80:
        medal = "🥈"
        color = "#C0C0C0"
        level = "Good Confidence"
    elif confidence_percent >= 70:
        medal = "🥉"
        color = "#CD7F32"
        level = "Moderate Confidence"
    else:
        medal = "📊"
        color = "#808080"
        level = "Low Confidence"
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {color}20, {color}10); 
                padding: 1rem; border-radius: 10px; margin: 1rem 0;
                border-left: 4px solid {color};">
        <h3 style="color: #2c3e50; margin: 0;">{medal} {level}</h3>
        <h2 style="color: #2c3e50; margin: 0.5rem 0;">{confidence_percent:.1f}%</h2>
        <div style="background: #e9ecef; height: 10px; border-radius: 5px; overflow: hidden;">
            <div style="background: {color}; height: 100%; width: {confidence_percent}%; transition: width 0.5s ease;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_disease_info(disease_key, disease_db):
    """Display comprehensive disease information"""
    if disease_key not in disease_db:
        st.warning(f"Detailed information not available for: {disease_key}")
        return
    
    disease_info = disease_db[disease_key]
    
    # Disease overview
    st.markdown(f"""
    <div style="background: white; padding: 1.5rem; border-radius: 10px; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin: 1rem 0;
                border-left: 4px solid #28a745;">
        <h2 style="color: #2c3e50; margin-bottom: 0.5rem;">🔬 {disease_info.get('disease_name', 'Unknown Disease')}</h2>
        <p style="color: #6c757d; font-style: italic; margin: 0;">
            <strong>Scientific Name:</strong> {disease_info.get('scientific_name', 'N/A')}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Severity indicator
    severity = disease_info.get('severity', 'Unknown')
    severity_colors = {
        'Low': '#28a745',
        'Moderate': '#ffc107', 
        'High': '#fd7e14',
        'Severe': '#dc3545'
    }
    severity_color = severity_colors.get(severity, '#6c757d')
    
    st.markdown(f"""
    <div style="background: {severity_color}15; border: 1px solid {severity_color}; 
                border-radius: 10px; padding: 1rem; margin: 1rem 0;">
        <h4 style="color: {severity_color}; margin: 0;">⚠️ Severity Level: {severity}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Affected crops
    affected_crops = disease_info.get('affected_crops', [])
    if affected_crops:
        st.markdown(f"""
        <div style="background: #e3f2fd; border: 1px solid #2196f3; border-radius: 10px; padding: 1rem; margin: 1rem 0;">
            <h4 style="color: #1565c0; margin-bottom: 0.5rem;">🌾 Affected Crops</h4>
            <p style="color: #1565c0; margin: 0;">{', '.join(affected_crops)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Symptoms
    symptoms = disease_info.get('symptoms', [])
    if symptoms:
        with st.expander("🔍 **Symptoms & Signs**", expanded=True):
            for symptom in symptoms:
                st.write(f"• {symptom}")
    
    # Treatment tabs
    if any(key in disease_info for key in ['organic_treatments', 'chemical_treatments', 'cultural_practices']):
        tab1, tab2, tab3 = st.tabs(["🌱 Organic Treatments", "⚗️ Chemical Treatments", "🚜 Cultural Practices"])
        
        with tab1:
            organic = disease_info.get('organic_treatments', [])
            if organic:
                for treatment in organic:
                    st.write(f"• {treatment}")
            else:
                st.info("No organic treatments available for this disease.")
        
        with tab2:
            chemical = disease_info.get('chemical_treatments', [])
            if chemical:
                for treatment in chemical:
                    st.write(f"• {treatment}")
            else:
                st.info("No chemical treatments available for this disease.")
        
        with tab3:
            cultural = disease_info.get('cultural_practices', [])
            if cultural:
                for practice in cultural:
                    st.write(f"• {practice}")
            else:
                st.info("No cultural practices available for this disease.")
    
    # Prevention and emergency actions
    col1, col2 = st.columns(2)
    
    with col1:
        prevention = disease_info.get('prevention_tips', [])
        if prevention:
            with st.expander("🛡️ **Prevention Tips**"):
                for tip in prevention:
                    st.write(f"• {tip}")
    
    with col2:
        emergency = disease_info.get('emergency_actions', [])
        if emergency:
            with st.expander("🚨 **Emergency Actions**"):
                for action in emergency:
                    st.write(f"• {action}")

def main():
    """Main application function"""
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #28a745;
        margin: 1rem 0;
        text-align: center;
    }
    
    .footer-section {
        text-align: center;
        padding: 2rem;
        background: #f8f9fa;
        border-radius: 10px;
        margin-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌿 Plant Disease Detection System</h1>
        <p>Advanced AI-Powered Plant Health Analysis</p>
        <p><em>Optimized for Streamlit Cloud Deployment</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load necessary data
    class_indices = load_class_indices()
    disease_db = load_disease_database()
    
    if not class_indices:
        st.stop()
    
    # Load model
    with st.spinner("Loading AI model..."):
        model_manager = load_model_manager()
    
    if not model_manager:
        st.error("Failed to load model. Please check the setup.")
        st.stop()
    
    # Display model information
    with st.sidebar:
        st.header("🤖 Model Information")
        model_info = model_manager.model_info
        st.info(f"""
        **Model Type:** {model_info['name']}
        
        **Format:** {model_info['format'].upper()}
        
        **Size:** {model_info['size_mb']:.1f} MB
        
        **Status:** ✅ Loaded Successfully
        
        **Classes:** {len(class_indices)} diseases
        """)
        
        # Show compression info if using compressed model
        if 'compressed' in model_info['relative_path'].lower() or 'tflite' in model_info['format']:
            st.success("🚀 Using Compressed Model - Deployment Optimized!")
        
        st.header("📊 Quick Stats")
        st.metric("Supported Crops", "14 types")
        st.metric("Disease Classes", "38 total")
        st.metric("Model Accuracy", "~95%+")
    
    # Image upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.header("📸 Upload Plant Image")
    st.write("Upload a clear image of the plant leaf or affected area for disease detection.")
    
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png'],
        help="Supported formats: JPG, JPEG, PNG. Max size: 200MB"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 2])
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            # Preprocess and predict
            with st.spinner("🔍 Analyzing image..."):
                img_array = preprocess_image(image)
                
                if img_array is not None:
                    predicted_class, confidence, predictions = predict_disease(
                        model_manager, img_array, class_indices
                    )
                    
                    if predicted_class:
                        st.success("✅ Analysis Complete!")
                        
                        # Display results
                        st.markdown("### 🎯 Detection Results")
                        
                        # Parse disease name for display
                        if "___" in predicted_class:
                            crop, disease = predicted_class.split("___", 1)
                            crop = crop.replace("_", " ").title()
                            disease = disease.replace("_", " ").title()
                            display_name = f"{crop} - {disease}"
                        else:
                            display_name = predicted_class.replace("_", " ").title()
                        
                        st.markdown(f"**Detected:** {display_name}")
                        
                        # Display confidence
                        display_confidence_score(confidence)
                        
                        # Display detailed disease information
                        if disease_db:
                            st.markdown("---")
                            st.markdown("### 📋 Detailed Disease Information")
                            display_disease_info(predicted_class, disease_db)
                        
                        # Action recommendations based on confidence
                        if confidence < 0.7:
                            st.warning("""
                            ⚠️ **Low Confidence Detection**
                            
                            Consider:
                            - Taking a clearer, well-lit photo
                            - Ensuring the affected area is clearly visible
                            - Consulting with agricultural experts
                            """)
                        elif confidence < 0.8:
                            st.info("""
                            ℹ️ **Moderate Confidence**
                            
                            Results are reasonably reliable. Consider getting a second opinion for critical decisions.
                            """)
                        else:
                            st.success("""
                            ✅ **High Confidence Detection**
                            
                            Results are highly reliable. Follow the recommended treatment guidelines.
                            """)
    
    else:
        # Instructions when no image is uploaded
        st.markdown("""
        ### 📋 How to Use
        
        1. **Upload Image**: Click the upload button above and select a plant image
        2. **Wait for Analysis**: Our AI model will analyze the image
        3. **View Results**: Get disease detection results with confidence scores
        4. **Follow Recommendations**: Review treatment and prevention guidelines
        
        ### 💡 Tips for Best Results
        
        - Use clear, well-lit photos
        - Focus on affected leaves or plant parts
        - Avoid blurry or low-resolution images
        - Ensure the disease symptoms are clearly visible
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer-section">
        <h4>🌱 Plant Disease Detection System</h4>
        <p>Powered by TensorFlow Lite | Optimized for Streamlit Cloud</p>
        <p><em>Always consult agricultural experts for severe plant diseases</em></p>
        <p>🚀 <strong>Deployment Ready</strong> | 📱 <strong>Mobile Friendly</strong> | ⚡ <strong>Fast Inference</strong></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()