import os
import json
import gc
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="🌱 Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Custom header styling */
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #2E8B57, #90EE90);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Upload section styling */
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #28a745;
        margin: 1rem 0;
        text-align: center;
        color: #2c3e50;
    }
    
    .upload-section h3, .upload-section p {
        color: #2c3e50 !important;
    }
    
    /* Result cards */
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #28a745;
        color: #2c3e50;
    }
    
    .result-card h1, .result-card h2, .result-card h3, .result-card p {
        color: #2c3e50 !important;
    }
    
    .confidence-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    /* Severity indicators */
    .severity-healthy {
        background: linear-gradient(135deg, #4CAF50, #8BC34A);
        color: white;
        padding: 0.8rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
    }
    
    .severity-mild {
        background: linear-gradient(135deg, #2196F3, #03DAC6);
        color: white;
        padding: 0.8rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
    }
    
    .severity-moderate {
        background: linear-gradient(135deg, #FF9800, #FFC107);
        color: white;
        padding: 0.8rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
    }
    
    .severity-severe {
        background: linear-gradient(135deg, #F44336, #E91E63);
        color: white;
        padding: 0.8rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
    }
    
    /* Treatment tabs styling */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1rem;
        font-weight: bold;
        color: #2c3e50 !important;
    }
    
    /* Ensure all text in tabs is visible */
    .stTabs [data-baseweb="tab-panel"] {
        color: #2c3e50 !important;
    }
    
    .stTabs [data-baseweb="tab-panel"] p, 
    .stTabs [data-baseweb="tab-panel"] h1, 
    .stTabs [data-baseweb="tab-panel"] h2, 
    .stTabs [data-baseweb="tab-panel"] h3, 
    .stTabs [data-baseweb="tab-panel"] h4, 
    .stTabs [data-baseweb="tab-panel"] li {
        color: #2c3e50 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #28a745, #20c997);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
    
    /* Image container */
    .image-container {
        text-align: center;
        padding: 1rem;
        background: white;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        color: #2c3e50;
    }
    
    .image-container h3, .image-container p, .image-container strong {
        color: #2c3e50 !important;
    }
    
    /* Info boxes */
    .info-box {
        background: #e8f5e8;
        border: 1px solid #28a745;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: #2c3e50;
    }
    
    .info-box h4, .info-box p, .info-box strong, .info-box ul, .info-box li {
        color: #2c3e50 !important;
    }
    
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
    }
    
    .warning-box h4, .warning-box p, .warning-box strong, .warning-box ul, .warning-box li {
        color: #856404 !important;
    }
    
    .error-box {
        background: #f8d7da;
        border: 1px solid #dc3545;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: #721c24;
    }
    
    .error-box h4, .error-box p, .error-box strong, .error-box ul, .error-box li {
        color: #721c24 !important;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        .main-header p {
            font-size: 1rem;
        }
        .upload-section {
            padding: 1rem;
        }
    }
    
    /* Global text color fixes for white backgrounds */
    .stMarkdown, .stText {
        color: #2c3e50;
    }
    
    /* Ensure all expandable sections have proper text color */
    .streamlit-expanderHeader {
        color: #2c3e50 !important;
    }
    
    .streamlit-expanderContent {
        color: #2c3e50 !important;
    }
    
    /* Fix text color in all containers */
    div[data-testid="stMarkdownContainer"] {
        color: #2c3e50;
    }
    
    /* Footer text color */
    .footer-section {
        color: #2c3e50 !important;
    }
    
    .footer-section h4, .footer-section p, .footer-section em {
        color: #2c3e50 !important;
    }
    
    /* Loading animation */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid #3498db;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)


working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"

# Function to load model with memory management
@st.cache_resource
def load_model_with_cache():
    """Load the model with caching and memory management"""
    try:
        # Configure TensorFlow for better memory management
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                st.warning(f"GPU configuration warning: {e}")
        
        # Force garbage collection before loading
        gc.collect()
        
        # Load model with explicit memory management
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Please check if the model file exists and you have sufficient memory.")
        return None

# Load the pre-trained model using cached function
model = load_model_with_cache()

# loading the class names
try:
    with open(f"{working_dir}/class_indices.json", 'r') as f:
        class_indices = json.load(f)
except FileNotFoundError:
    st.error("class_indices.json file not found!")
    st.stop()
except json.JSONDecodeError:
    st.error("Error reading class_indices.json file!")
    st.stop()

# loading the disease information database
try:
    with open(f"{working_dir}/disease_info_database.json", 'r') as f:
        disease_database = json.load(f)
except FileNotFoundError:
    st.error("disease_info_database.json file not found!")
    st.stop()
except json.JSONDecodeError:
    st.error("Error reading disease_info_database.json file!")
    st.stop()


# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array


# Function to Predict the Class of an Image with confidence scores
def predict_image_class(model, image_path, class_indices):
    if model is None:
        return None, "Model not loaded. Please check system memory."
    
    try:
        preprocessed_img = load_and_preprocess_image(image_path)
        predictions = model.predict(preprocessed_img)
        
        # Get top 3 predictions with confidence scores
        top_indices = np.argsort(predictions[0])[::-1][:3]
        results = []
        
        for idx in top_indices:
            confidence = predictions[0][idx] * 100
            disease_name = class_indices[str(idx)]
            results.append((disease_name, confidence))
        
        # Return the top prediction and all results
        return results[0][0], results
        
    except Exception as e:
        return None, f"Prediction error: {str(e)}"

# Function to get disease information
def get_disease_info(disease_name, disease_database):
    """Get detailed disease information from database"""
    if disease_name in disease_database:
        return disease_database[disease_name]
    else:
        return None


# Streamlit App
# Custom header
st.markdown("""
<div class="main-header">
    <h1>🌱 Plant Disease Classifier</h1>
    <p>Advanced AI-powered plant health analysis and treatment recommendations</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for additional information
with st.sidebar:
    st.markdown("### 📋 How to Use")
    st.markdown("""
    1. **Upload Image**: Choose a clear photo of your plant
    2. **Analyze**: Click the 'Classify' button
    3. **Review Results**: Get detailed disease information
    4. **Take Action**: Follow treatment recommendations
    """)
    
    st.markdown("### 📊 Model Information")
    st.info("""
    **Model Type**: Convolutional Neural Network (CNN)
    
    **Supported Plants**: 
    - Apple, Cherry, Corn, Grape
    - Orange, Peach, Pepper, Potato
    - Raspberry, Soybean, Squash
    - Strawberry, Tomato, Blueberry
    
    **Disease Classes**: 38 conditions
    """)
    
    st.markdown("### 🏥 Emergency Contacts")
    st.warning("""
    For severe plant diseases, consider contacting:
    - Local Agricultural Extension Office
    - Plant Disease Diagnostic Laboratory
    - Professional Agronomist
    """)

# Check if model loaded successfully
if model is None:
    st.markdown("""
    <div class="error-box">
        <h3>⚠️ Model Loading Error</h3>
        <p>The AI model failed to load. This might be due to insufficient memory or a corrupted model file.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 💡 Troubleshooting Steps:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🔧 Memory Issues**
        - Close other applications
        - Restart the application
        - Ensure 2GB+ free RAM
        """)
    
    with col2:
        st.markdown("""
        **📁 File Issues**
        - Check model file exists
        - Verify file permissions
        - Re-download if corrupted
        """)
    
    with col3:
        st.markdown("""
        **🔄 System Issues**
        - Restart computer
        - Update dependencies
        - Check disk space
        """)
    
    st.stop()

# Main upload section
st.markdown("""
<div class="upload-section">
    <h3>📸 Upload Plant Image for Analysis</h3>
    <p>Supported formats: JPG, JPEG, PNG | Max size: 200MB</p>
</div>
""", unsafe_allow_html=True)

uploaded_image = st.file_uploader("", type=["jpg", "jpeg", "png"], help="Upload a clear image of your plant for disease analysis")

if uploaded_image is not None:
    # Create responsive layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        image = Image.open(uploaded_image)
        
        # Display original image with better sizing
        st.image(image, caption="📷 Uploaded Image", use_column_width=True)
        
        # Image info
        st.markdown(f"""
        <div class="info-box">
            <strong>📋 Image Details:</strong><br>
            • Size: {image.size[0]} × {image.size[1]} pixels<br>
            • Format: {image.format}<br>
            • Mode: {image.mode}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Analysis section
    analysis_col1, analysis_col2, analysis_col3 = st.columns([1, 2, 1])
    
    with analysis_col2:
        if st.button('🔍 Analyze Plant Health', key="analyze_btn"):
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text('🤖 Initializing AI model...')
            progress_bar.progress(25)
            
            status_text.text('📊 Processing image...')
            progress_bar.progress(50)
            
            status_text.text('🧠 Analyzing plant health...')
            progress_bar.progress(75)
            
            # Actual prediction
            top_prediction, all_predictions = predict_image_class(model, uploaded_image, class_indices)
            
            progress_bar.progress(100)
            status_text.text('✅ Analysis complete!')
            
            # Clear progress indicators
            import time
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            if top_prediction is None or "error" in str(all_predictions).lower():
                st.markdown(f"""
                <div class="error-box">
                    <h4>❌ Analysis Error</h4>
                    <p>{str(all_predictions)}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Success message with primary prediction
                st.markdown(f"""
                <div class="result-card">
                    <h2 style="color: #28a745; text-align: center;">🎯 Analysis Results</h2>
                    <h3 style="text-align: center; color: #2c3e50;">Primary Diagnosis: <span style="color: #e74c3c;">{top_prediction}</span></h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence scores with enhanced styling
                st.markdown("### 📊 Confidence Scores")
                
                conf_col1, conf_col2, conf_col3 = st.columns(3)
                
                for i, (disease, confidence) in enumerate(all_predictions):
                    if i == 0:
                        with conf_col1:
                            st.markdown(f"""
                            <div class="confidence-card" style="background: linear-gradient(135deg, #FFD700, #FFA500);">
                                <h4>🥇 Primary</h4>
                                <p style="font-size: 0.9rem; margin-bottom: 0.5rem;">{disease}</p>
                                <h3>{confidence:.1f}%</h3>
                            </div>
                            """, unsafe_allow_html=True)
                    elif i == 1:
                        with conf_col2:
                            st.markdown(f"""
                            <div class="confidence-card" style="background: linear-gradient(135deg, #C0C0C0, #808080);">
                                <h4>🥈 Secondary</h4>
                                <p style="font-size: 0.9rem; margin-bottom: 0.5rem;">{disease}</p>
                                <h3>{confidence:.1f}%</h3>
                            </div>
                            """, unsafe_allow_html=True)
                    elif i == 2:
                        with conf_col3:
                            st.markdown(f"""
                            <div class="confidence-card" style="background: linear-gradient(135deg, #CD7F32, #8B4513);">
                                <h4>🥉 Tertiary</h4>
                                <p style="font-size: 0.9rem; margin-bottom: 0.5rem;">{disease}</p>
                                <h3>{confidence:.1f}%</h3>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Get detailed disease information
                disease_info = get_disease_info(top_prediction, disease_database)
                
                if disease_info:
                    st.markdown("---")
                    
                    # Disease header with enhanced styling
                    st.markdown(f"""
                    <div class="result-card">
                        <h1 style="text-align: center; color: #2c3e50;">🔬 {disease_info['disease_name']}</h1>
                        <p style="text-align: center; font-style: italic; color: #7f8c8d;">Scientific Name: {disease_info['scientific_name']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Overview section with better layout
                    overview_col1, overview_col2 = st.columns([2, 1])
                    
                    with overview_col1:
                        st.markdown("### 📋 Disease Overview")
                        st.markdown(f"""
                        <div class="info-box">
                            <strong>🌾 Affected Crops:</strong> {', '.join(disease_info['affected_crops'])}<br>
                            <strong>🔬 Scientific Classification:</strong> {disease_info['scientific_name']}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with overview_col2:
                        st.markdown("### 🚨 Severity Level")
                        severity = disease_info['severity']
                        if severity == "None":
                            st.markdown('<div class="severity-healthy">✅ Healthy Plant</div>', unsafe_allow_html=True)
                        elif severity == "Mild":
                            st.markdown('<div class="severity-mild">ℹ️ Mild Severity</div>', unsafe_allow_html=True)
                        elif severity == "Moderate":
                            st.markdown('<div class="severity-moderate">⚠️ Moderate Severity</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="severity-severe">🚨 Severe Disease</div>', unsafe_allow_html=True)
                    
                    # Symptoms section with expandable format
                    if disease_info['symptoms']:
                        with st.expander("� Symptoms & Signs", expanded=True):
                            symptoms_col1, symptoms_col2 = st.columns(2)
                            
                            mid_point = len(disease_info['symptoms']) // 2
                            
                            with symptoms_col1:
                                for symptom in disease_info['symptoms'][:mid_point]:
                                    st.markdown(f"• {symptom}")
                            
                            with symptoms_col2:
                                for symptom in disease_info['symptoms'][mid_point:]:
                                    st.markdown(f"• {symptom}")
                    
                    # Treatment sections with enhanced tabs
                    if disease_info['severity'] != "None":
                        st.markdown("### 💊 Treatment & Management Options")
                        
                        tab1, tab2, tab3, tab4 = st.tabs([
                            "🌿 Organic Solutions", 
                            "⚗️ Chemical Treatments", 
                            "🌱 Cultural Practices", 
                            "🚨 Emergency Actions"
                        ])
                        
                        with tab1:
                            st.markdown("#### 🌿 Natural & Organic Treatments")
                            for i, treatment in enumerate(disease_info['organic_treatments'], 1):
                                st.markdown(f"**{i}.** {treatment}")
                        
                        with tab2:
                            st.markdown("#### ⚗️ Chemical Treatment Options")
                            st.warning("⚠️ Always follow label instructions and safety guidelines")
                            for i, treatment in enumerate(disease_info['chemical_treatments'], 1):
                                st.markdown(f"**{i}.** {treatment}")
                        
                        with tab3:
                            st.markdown("#### 🌱 Cultural & Preventive Practices")
                            for i, practice in enumerate(disease_info['cultural_practices'], 1):
                                st.markdown(f"**{i}.** {practice}")
                        
                        with tab4:
                            st.markdown("#### 🚨 Immediate Emergency Actions")
                            st.error("Take these steps immediately for severe cases:")
                            for i, action in enumerate(disease_info['emergency_actions'], 1):
                                st.markdown(f"**{i}.** {action}")
                    
                    # Prevention tips with better styling
                    if disease_info['prevention_tips']:
                        with st.expander("🛡️ Prevention & Future Care Tips", expanded=False):
                            prev_col1, prev_col2 = st.columns(2)
                            
                            mid_point = len(disease_info['prevention_tips']) // 2
                            
                            with prev_col1:
                                for tip in disease_info['prevention_tips'][:mid_point]:
                                    st.markdown(f"• {tip}")
                            
                            with prev_col2:
                                for tip in disease_info['prevention_tips'][mid_point:]:
                                    st.markdown(f"• {tip}")
                    
                    # Action plan with severity-based recommendations
                    st.markdown("---")
                    st.markdown("### 💡 Recommended Action Plan")
                    
                    if disease_info['severity'] == "None":
                        st.markdown("""
                        <div class="info-box">
                            <h4>🎉 Excellent News!</h4>
                            <p>Your plant appears to be in excellent health! Continue with your current care routine and monitor regularly for any changes.</p>
                            <strong>Next Steps:</strong>
                            <ul>
                                <li>Continue current watering and fertilization schedule</li>
                                <li>Regular monitoring for early disease detection</li>
                                <li>Maintain good garden hygiene</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    elif disease_info['severity'] == "Mild":
                        st.markdown("""
                        <div class="info-box">
                            <h4>📋 Mild Condition Detected</h4>
                            <p>Early intervention can prevent this condition from worsening.</p>
                            <strong>Action Timeline:</strong>
                            <ul>
                                <li><strong>Today:</strong> Start with organic treatments</li>
                                <li><strong>This Week:</strong> Implement cultural practices</li>
                                <li><strong>Monitor:</strong> Check progress in 7-10 days</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    elif disease_info['severity'] == "Moderate":
                        st.markdown("""
                        <div class="warning-box">
                            <h4>⚡ Moderate Condition - Action Required</h4>
                            <p>This condition requires prompt treatment to prevent spread and damage.</p>
                            <strong>Action Timeline:</strong>
                            <ul>
                                <li><strong>Immediately:</strong> Remove affected plant parts</li>
                                <li><strong>Today:</strong> Apply recommended treatments</li>
                                <li><strong>This Week:</strong> Monitor and repeat treatments</li>
                                <li><strong>Follow-up:</strong> Reassess in 5-7 days</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="error-box">
                            <h4>🚨 Severe Condition - Urgent Action Required</h4>
                            <p>This is a serious plant disease that requires immediate intervention.</p>
                            <strong>Emergency Protocol:</strong>
                            <ul>
                                <li><strong>RIGHT NOW:</strong> Isolate affected plants</li>
                                <li><strong>TODAY:</strong> Implement emergency treatments</li>
                                <li><strong>ASAP:</strong> Consult with local agricultural expert</li>
                                <li><strong>Monitor:</strong> Daily assessment required</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                
                else:
                    st.markdown("""
                    <div class="warning-box">
                        <h4>⚠️ Information Not Available</h4>
                        <p>Detailed disease information is not available for this condition. Consider consulting with a local agricultural expert.</p>
                    </div>
                    """, unsafe_allow_html=True)

# Footer with additional information
st.markdown("---")
st.markdown("""
<div class="footer-section" style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px; margin-top: 2rem;">
    <h4>🌱 Plant Disease Classifier</h4>
    <p>Powered by Advanced Deep Learning | For Educational and Research Purposes</p>
    <p><em>Always consult with professional agronomists for severe plant diseases</em></p>
</div>
""", unsafe_allow_html=True)
