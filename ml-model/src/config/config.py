# src/config/config.py

import os
from typing import Dict, Any

class ModelConfig:
    """Configuration for ML models and processing"""
    
    # Model settings
    YOLO_MODEL_VERSION = "yolov8n.pt"  # Start with nano for speed
    MIDAS_MODEL_VERSION = "Intel/dpt-large"
    DEVICE = "cuda" if os.getenv("USE_GPU", "false").lower() == "true" else "cpu"
    
    # Image processing settings
    MAX_IMAGE_SIZE = 1280  # Max size for YOLO processing
    DEPTH_IMAGE_SIZE = 512  # Size for depth estimation
    SUPPORTED_FORMATS = [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]
    
    # Detection settings
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.45
    
    # Measurement settings
    PIXEL_TO_METER_RATIO = 1.0  # Will be calculated from drone data
    DEFAULT_DRONE_HEIGHT = 50.0  # meters
    
    # API settings
    API_HOST = "0.0.0.0"
    API_PORT = 8080
    MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB
    
    # Processing settings
    BATCH_SIZE = 4
    MAX_CONCURRENT_REQUESTS = 10

class LandAnalysisConfig:
    """Configuration specific to land analysis"""
    
    # Land feature classes that YOLO should detect
    LAND_FEATURES = {
        'field': 0,
        'building': 1, 
        'road': 2,
        'vegetation': 3,
        'water': 4,
        'bare_land': 5
    }
    
    # Minimum area threshold (in square meters)
    MIN_AREA_THRESHOLD = 10.0
    
    # Depth analysis settings
    DEPTH_SMOOTHING_KERNEL = 5
    DEPTH_OUTLIER_THRESHOLD = 3.0  # Standard deviations
    
    # Report generation settings
    REPORT_TEMPLATE = "land_analysis_template.html"
    OUTPUT_FORMATS = ["json", "pdf", "html"]

# Global config instance
config = ModelConfig()
land_config = LandAnalysisConfig()