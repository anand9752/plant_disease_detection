"""
Compressed Model Testing Script
==============================

This script tests the compressed models to ensure they work correctly
and maintain acceptable accuracy levels.

Author: AI Assistant
Date: September 2025
"""

import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
import time

class CompressedModelTester:
    def __init__(self, compressed_models_dir="compressed_models"):
        """
        Initialize the model tester
        
        Args:
            compressed_models_dir (str): Directory containing compressed models
        """
        self.compressed_models_dir = compressed_models_dir
        self.class_indices = None
        self.test_image = None
        self.load_class_indices()
        self.create_test_image()
    
    def load_class_indices(self):
        """Load class indices for prediction mapping"""
        try:
            with open("class_indices.json", 'r') as f:
                self.class_indices = json.load(f)
            print(f"✅ Loaded {len(self.class_indices)} disease classes")
        except FileNotFoundError:
            print("⚠️  class_indices.json not found. Creating dummy classes...")
            # Create dummy classes for testing
            self.class_indices = {str(i): f"Class_{i}" for i in range(38)}
    
    def create_test_image(self):
        """Create a synthetic test image"""
        # Create a realistic plant-like image
        test_image = np.random.uniform(0.0, 1.0, (224, 224, 3)).astype(np.float32)
        
        # Add some plant-like patterns (green tones)
        test_image[:, :, 1] *= 1.2  # Enhance green channel
        test_image = np.clip(test_image, 0.0, 1.0)
        
        self.test_image = np.expand_dims(test_image, axis=0)
        print("✅ Test image created (224x224x3)")
    
    def test_original_model(self, model_path):
        """Test the original Keras model"""
        print("\n🔍 Testing Original Keras Model...")
        
        try:
            start_time = time.time()
            model = tf.keras.models.load_model(model_path)
            load_time = time.time() - start_time
            
            start_time = time.time()
            predictions = model.predict(self.test_image, verbose=0)
            inference_time = time.time() - start_time
            
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class] * 100
            class_name = self.class_indices.get(str(predicted_class), "Unknown")
            
            print(f"✅ Original Model Results:")
            print(f"   📊 Prediction: {class_name}")
            print(f"   🎯 Confidence: {confidence:.2f}%")
            print(f"   ⏱️  Load Time: {load_time:.2f}s")
            print(f"   ⚡ Inference Time: {inference_time:.4f}s")
            
            return {
                'predictions': predictions,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'class_name': class_name,
                'load_time': load_time,
                'inference_time': inference_time
            }
            
        except Exception as e:
            print(f"❌ Original model test failed: {str(e)}")
            return None
    
    def test_tflite_model(self, model_path, model_name):
        """Test TensorFlow Lite models"""
        print(f"\n🔍 Testing {model_name}...")
        
        try:
            start_time = time.time()
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            load_time = time.time() - start_time
            
            # Get input and output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Prepare input data
            input_data = self.test_image.astype(input_details[0]['dtype'])
            
            # Handle quantized models
            if input_details[0]['dtype'] == np.int8:
                # Quantize input to int8
                input_scale, input_zero_point = input_details[0]['quantization']
                input_data = (input_data / input_scale + input_zero_point).astype(np.int8)
            
            start_time = time.time()
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            inference_time = time.time() - start_time
            
            # Handle quantized output
            if output_details[0]['dtype'] == np.int8:
                output_scale, output_zero_point = output_details[0]['quantization']
                output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
            
            # Apply softmax to get probabilities
            predictions = tf.nn.softmax(output_data[0]).numpy()
            
            predicted_class = np.argmax(predictions)
            confidence = predictions[predicted_class] * 100
            class_name = self.class_indices.get(str(predicted_class), "Unknown")
            
            print(f"✅ {model_name} Results:")
            print(f"   📊 Prediction: {class_name}")
            print(f"   🎯 Confidence: {confidence:.2f}%")
            print(f"   ⏱️  Load Time: {load_time:.2f}s")
            print(f"   ⚡ Inference Time: {inference_time:.4f}s")
            print(f"   🔧 Input Type: {input_details[0]['dtype']}")
            print(f"   🔧 Output Type: {output_details[0]['dtype']}")
            
            return {
                'predictions': predictions,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'class_name': class_name,
                'load_time': load_time,
                'inference_time': inference_time,
                'input_dtype': str(input_details[0]['dtype']),
                'output_dtype': str(output_details[0]['dtype'])
            }
            
        except Exception as e:
            print(f"❌ {model_name} test failed: {str(e)}")
            return None
    
    def test_keras_model(self, model_path, model_name):
        """Test compressed Keras models"""
        print(f"\n🔍 Testing {model_name}...")
        
        try:
            start_time = time.time()
            model = tf.keras.models.load_model(model_path)
            load_time = time.time() - start_time
            
            start_time = time.time()
            predictions = model.predict(self.test_image, verbose=0)
            inference_time = time.time() - start_time
            
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class] * 100
            class_name = self.class_indices.get(str(predicted_class), "Unknown")
            
            print(f"✅ {model_name} Results:")
            print(f"   📊 Prediction: {class_name}")
            print(f"   🎯 Confidence: {confidence:.2f}%")
            print(f"   ⏱️  Load Time: {load_time:.2f}s")
            print(f"   ⚡ Inference Time: {inference_time:.4f}s")
            
            return {
                'predictions': predictions[0],
                'predicted_class': predicted_class,
                'confidence': confidence,
                'class_name': class_name,
                'load_time': load_time,
                'inference_time': inference_time
            }
            
        except Exception as e:
            print(f"❌ {model_name} test failed: {str(e)}")
            return None
    
    def compare_predictions(self, original_result, compressed_result, model_name):
        """Compare original vs compressed model predictions"""
        if original_result is None or compressed_result is None:
            return
        
        print(f"\n📊 Comparison: Original vs {model_name}")
        print("-" * 50)
        
        # Prediction accuracy
        same_prediction = original_result['predicted_class'] == compressed_result['predicted_class']
        confidence_diff = abs(original_result['confidence'] - compressed_result['confidence'])
        
        print(f"🎯 Same Prediction: {'✅ Yes' if same_prediction else '❌ No'}")
        print(f"📈 Confidence Difference: {confidence_diff:.2f}%")
        
        # Performance comparison
        speed_improvement = (original_result['inference_time'] / compressed_result['inference_time']) - 1
        load_improvement = (original_result['load_time'] / compressed_result['load_time']) - 1
        
        print(f"⚡ Inference Speed: {speed_improvement*100:+.1f}%")
        print(f"⏱️  Load Speed: {load_improvement*100:+.1f}%")
        
        # Overall assessment
        if same_prediction and confidence_diff < 10:
            print("🏆 Status: ✅ EXCELLENT - Same prediction, similar confidence")
        elif same_prediction and confidence_diff < 20:
            print("🏆 Status: ✅ GOOD - Same prediction, acceptable confidence difference")
        elif confidence_diff < 30:
            print("🏆 Status: ⚠️  FAIR - Different prediction but close confidence")
        else:
            print("🏆 Status: ❌ POOR - Significant difference in predictions")
    
    def test_all_models(self):
        """Test all available compressed models"""
        print("🧪 COMPREHENSIVE MODEL TESTING")
        print("=" * 60)
        
        results = {}
        
        # Test original model if available
        original_path = os.path.join("trained_model", "plant_disease_prediction_model.h5")
        if os.path.exists(original_path):
            results['original'] = self.test_original_model(original_path)
        else:
            print("⚠️  Original model not found, skipping comparison")
            results['original'] = None
        
        # Test compressed models
        if not os.path.exists(self.compressed_models_dir):
            print(f"❌ Compressed models directory not found: {self.compressed_models_dir}")
            print("💡 Run model_compression.py first to create compressed models")
            return
        
        # Test TensorFlow Lite models
        tflite_models = [
            ("quantized_model.tflite", "INT8 Quantized Model"),
            ("dynamic_quantized_model.tflite", "Dynamic Range Quantized Model")
        ]
        
        for model_file, model_name in tflite_models:
            model_path = os.path.join(self.compressed_models_dir, model_file)
            if os.path.exists(model_path):
                results[model_file] = self.test_tflite_model(model_path, model_name)
                if results['original']:
                    self.compare_predictions(results['original'], results[model_file], model_name)
        
        # Test Keras models
        keras_models = [
            ("pruned_model.h5", "Pruned Model"),
            ("optimized_model.h5", "Optimized Model")
        ]
        
        for model_file, model_name in keras_models:
            model_path = os.path.join(self.compressed_models_dir, model_file)
            if os.path.exists(model_path):
                results[model_file] = self.test_keras_model(model_path, model_name)
                if results['original']:
                    self.compare_predictions(results['original'], results[model_file], model_name)
        
        # Generate final report
        self.generate_test_report(results)
        
        return results
    
    def generate_test_report(self, results):
        """Generate a comprehensive test report"""
        print("\n" + "=" * 60)
        print("📋 FINAL TEST REPORT")
        print("=" * 60)
        
        successful_tests = {k: v for k, v in results.items() if v is not None}
        
        if not successful_tests:
            print("❌ No models tested successfully")
            return
        
        print(f"✅ Successfully tested {len(successful_tests)} models")
        print()
        
        # Performance summary
        print("⚡ PERFORMANCE SUMMARY:")
        print("-" * 40)
        
        for model_name, result in successful_tests.items():
            if model_name == 'original':
                continue
            
            print(f"🔸 {model_name}:")
            print(f"   Load Time: {result['load_time']:.2f}s")
            print(f"   Inference: {result['inference_time']:.4f}s")
            print(f"   Prediction: {result['class_name']}")
            print(f"   Confidence: {result['confidence']:.1f}%")
            print()
        
        # Recommendations
        print("💡 DEPLOYMENT RECOMMENDATIONS:")
        print("-" * 40)
        
        # Find fastest model
        inference_times = {k: v['inference_time'] for k, v in successful_tests.items() if k != 'original'}
        if inference_times:
            fastest_model = min(inference_times.keys(), key=lambda k: inference_times[k])
            print(f"🚀 Fastest Model: {fastest_model}")
            print(f"   Inference Time: {inference_times[fastest_model]:.4f}s")
        
        # Check file sizes
        print(f"\n📁 Model Files in {self.compressed_models_dir}:")
        for file in os.listdir(self.compressed_models_dir):
            if file.endswith(('.h5', '.tflite')):
                path = os.path.join(self.compressed_models_dir, file)
                size_mb = os.path.getsize(path) / (1024 * 1024)
                status = "✅ Streamlit Ready" if size_mb < 100 else "⚠️  Large"
                print(f"   {file}: {size_mb:.2f} MB - {status}")
        
        print(f"\n📄 Detailed compression report available in: {self.compressed_models_dir}/compression_report.txt")

def main():
    """
    Main testing function
    """
    print("🧪 Compressed Model Testing Tool")
    print("=" * 50)
    
    tester = CompressedModelTester()
    results = tester.test_all_models()
    
    print("\n🎉 Testing completed!")
    print("💡 Use the best performing compressed model for deployment")

if __name__ == "__main__":
    main()