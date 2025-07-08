# test_basic.py
"""
Basic test to check if ML models can be loaded
Run this first to verify everything is working
"""

def test_imports():
    """Test if all required packages are installed"""
    print("🔧 Testing Package Imports...")
    print("-" * 40)
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        import torchvision
        print(f"✅ TorchVision: {torchvision.__version__}")
        
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
        
        from PIL import Image
        print("✅ PIL/Pillow: Available")
        
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
        
        import pandas as pd
        print(f"✅ Pandas: {pd.__version__}")
        
        import fastapi
        print("✅ FastAPI: Available")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_device():
    """Test device availability"""
    print("\n💻 Testing Device Availability...")
    print("-" * 40)
    
    import torch
    
    print(f"CPU Available: ✅")
    
    if torch.cuda.is_available():
        print(f"GPU Available: ✅")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        device = "cuda"
    else:
        print(f"GPU Available: ❌ (Will use CPU)")
        device = "cpu"
    
    print(f"Selected Device: {device}")
    return device

def test_yolo_loading():
    """Test YOLOv8 model loading"""
    print("\n🎯 Testing YOLOv8 Loading...")
    print("-" * 40)
    
    try:
        from ultralytics import YOLO
        print("📦 Downloading YOLOv8 model (first time only)...")
        
        model = YOLO('yolov8n.pt')  # This will download the model
        print("✅ YOLOv8 model loaded successfully!")
        
        # Test model info
        print(f"Model classes: {len(model.names)} classes")
        print(f"Sample classes: {list(model.names.values())[:5]}...")
        
        return True, model
        
    except Exception as e:
        print(f"❌ YOLOv8 loading error: {e}")
        return False, None

def test_midas_loading():
    """Test MiDaS depth estimation model loading"""
    print("\n🏔️  Testing MiDaS Loading...")
    print("-" * 40)
    
    try:
        from transformers import pipeline
        print("📦 Loading MiDaS depth estimation model...")
        
        depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large")
        print("✅ MiDaS model loaded successfully!")
        
        return True, depth_estimator
        
    except Exception as e:
        print(f"❌ MiDaS loading error: {e}")
        return False, None

def create_test_image():
    """Create a simple test image"""
    print("\n🖼️  Creating Test Image...")
    print("-" * 40)
    
    try:
        import numpy as np
        import cv2
        
        # Create a 640x480 test image
        height, width = 480, 640
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create a simple pattern
        image[:, :] = [100, 150, 200]  # Light blue background
        
        # Add some rectangles to simulate land features
        cv2.rectangle(image, (100, 100), (300, 250), (0, 255, 0), -1)  # Green field
        cv2.rectangle(image, (350, 150), (500, 300), (139, 69, 19), -1)  # Brown soil
        cv2.rectangle(image, (50, 350), (200, 450), (0, 0, 255), -1)    # Red building
        
        # Save test image
        cv2.imwrite("test_image.jpg", image)
        print("✅ Test image created: test_image.jpg")
        
        return True, "test_image.jpg"
        
    except Exception as e:
        print(f"❌ Error creating test image: {e}")
        return False, None

def test_yolo_inference(model, image_path):
    """Test YOLOv8 inference on test image"""
    print("\n🔍 Testing YOLOv8 Inference...")
    print("-" * 40)
    
    try:
        import time
        from PIL import Image
        
        # Load image
        image = Image.open(image_path)
        print(f"Image loaded: {image.size}")
        
        # Run inference
        print("Running YOLO detection...")
        start_time = time.time()
        
        results = model(image, conf=0.5, verbose=False)
        
        inference_time = time.time() - start_time
        print(f"⏱️  Inference time: {inference_time:.2f} seconds")
        
        # Process results
        detections = 0
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                detections = len(result.boxes)
                print(f"🎯 Detections found: {detections}")
                
                for i, box in enumerate(result.boxes[:3]):  # Show first 3
                    bbox = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = model.names.get(class_id, f"class_{class_id}")
                    
                    print(f"   {i+1}. {class_name}: {confidence:.2f} confidence")
            else:
                print("ℹ️  No objects detected")
        
        print("✅ YOLO inference completed")
        return True
        
    except Exception as e:
        print(f"❌ YOLO inference error: {e}")
        return False

def test_midas_inference(depth_estimator, image_path):
    """Test MiDaS depth estimation"""
    print("\n📊 Testing MiDaS Depth Estimation...")
    print("-" * 40)
    
    try:
        import time
        from PIL import Image
        import numpy as np
        
        # Load image
        image = Image.open(image_path)
        
        # Run depth estimation
        print("Running depth estimation...")
        start_time = time.time()
        
        depth_result = depth_estimator(image)
        depth_map = np.array(depth_result['depth'])
        
        inference_time = time.time() - start_time
        print(f"⏱️  Inference time: {inference_time:.2f} seconds")
        
        # Analyze depth map
        print(f"📏 Depth map shape: {depth_map.shape}")
        print(f"📊 Depth range: {depth_map.min():.3f} - {depth_map.max():.3f}")
        print(f"📈 Average depth: {depth_map.mean():.3f}")
        
        print("✅ Depth estimation completed")
        return True
        
    except Exception as e:
        print(f"❌ Depth estimation error: {e}")
        return False

def main():
    """Run all basic tests"""
    print("🧪 DROT ML Model - Basic Test")
    print("=" * 50)
    print("Testing ML models locally without API server")
    print()
    
    # Test 1: Package imports
    if not test_imports():
        print("\n❌ Package import failed. Please check your installation.")
        return
    
    # Test 2: Device availability
    device = test_device()
    
    # Test 3: Model loading
    yolo_success, yolo_model = test_yolo_loading()
    midas_success, midas_model = test_midas_loading()
    
    if not yolo_success or not midas_success:
        print("\n❌ Model loading failed. Please check your internet connection.")
        return
    
    # Test 4: Create test image
    image_success, image_path = create_test_image()
    if not image_success:
        return
    
    # Test 5: Test inference
    yolo_inference = test_yolo_inference(yolo_model, image_path)
    midas_inference = test_midas_inference(midas_model, image_path)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    all_tests = [
        ("Package Imports", True),
        ("YOLOv8 Loading", yolo_success),
        ("MiDaS Loading", midas_success),
        ("Test Image Creation", image_success),
        ("YOLOv8 Inference", yolo_inference),
        ("MiDaS Inference", midas_inference)
    ]
    
    for test_name, result in all_tests:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<30} {status}")
    
    if all(result for _, result in all_tests):
        print(f"\n🎉 ALL TESTS PASSED!")
        print("Your ML models are working correctly!")
        print("\n📋 Next Steps:")
        print("1. Test with real drone images")
        print("2. Create the full ML service")
        print("3. Integrate with Node.js backend")
    else:
        print(f"\n❌ SOME TESTS FAILED")
        print("Please fix the issues before proceeding.")

if __name__ == "__main__":
    main()