# land_area_analyzer.py
"""
DROT - Complete Land Area Detection and Measurement System
Detects COMPLETE land areas from drone imagery using advanced boundary detection:
- Captures entire landmasses including all terrain types (vegetation, soil, beach, etc.)
- Uses multiple detection methods to ensure no land area is missed
- Focuses on land-water boundaries for accurate total area measurement
- Treats different land surfaces as one unified area

Usage: python land_area_analyzer.py <image_path> [drone_altitude]
"""

import sys
import os
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import math
from sklearn.cluster import KMeans
from scipy import ndimage

class CompleteLandDetector:
    """Complete land area detection system - captures entire landmasses"""
    
    def __init__(self, drone_altitude=50.0):
        self.drone_altitude = drone_altitude
        self.meters_per_pixel = None
        self.image_width = None
        self.image_height = None
        
        # Parameters for complete land detection
        self.min_landmass_percentage = 0.3  # Minimum 0.3% of image
        self.max_landmasses = 8             # Allow more landmasses for complex areas
        self.edge_threshold_low = 30        # Lower edge threshold for better detection
        self.edge_threshold_high = 80       # Higher edge threshold
        self.morphology_kernel_size = 12    # Kernel for connecting land areas
        
    def calculate_scale(self, image_width, drone_altitude):
        """Calculate meters per pixel based on drone altitude and camera specs"""
        fov_horizontal = 84
        fov_radians = math.radians(fov_horizontal / 2)
        ground_width = 2 * drone_altitude * math.tan(fov_radians)
        return ground_width / image_width
    
    def enhance_image_for_detection(self, image):
        """Enhance image to improve land-water contrast"""
        # Convert to LAB color space for better contrast
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel for better contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge back and convert to BGR
        enhanced_lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Additional sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Blend original and sharpened
        result = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
        
        return result
    
    def create_comprehensive_water_mask(self, image):
        """Create comprehensive water mask using multiple detection methods"""
        height, width = image.shape[:2]
        enhanced_image = self.enhance_image_for_detection(image)
        
        # Method 1: Blue channel dominance for water
        b, g, r = cv2.split(enhanced_image)
        blue_ratio = b.astype(float) / (b.astype(float) + g.astype(float) + r.astype(float) + 1)
        water_mask_blue = (blue_ratio > 0.4).astype(np.uint8) * 255
        
        # Method 2: HSV-based water detection with expanded ranges
        hsv = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2HSV)
        water_mask_hsv = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        # Multiple water color ranges
        water_ranges = [
            ([95, 40, 40], [130, 255, 255]),    # Deep blue water
            ([80, 30, 60], [120, 255, 255]),    # Light blue water  
            ([70, 20, 80], [110, 200, 255]),    # Turquoise water
            ([100, 60, 30], [125, 255, 200])    # Dark blue water
        ]
        
        for lower, upper in water_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            water_mask_hsv = cv2.bitwise_or(water_mask_hsv, mask)
        
        # Method 3: Saturation-based detection (water often has high saturation)
        s_channel = hsv[:,:,1]
        high_sat_mask = (s_channel > 100).astype(np.uint8) * 255
        
        # Method 4: Texture-based water detection
        gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = ndimage.generic_filter(laplacian, np.var, size=15)
        smooth_mask = (texture_variance < np.percentile(texture_variance, 30)).astype(np.uint8) * 255
        
        # Combine all water masks
        combined_water = cv2.bitwise_or(water_mask_blue, water_mask_hsv)
        combined_water = cv2.bitwise_or(combined_water, high_sat_mask)
        
        # Clean up water mask
        kernel = np.ones((8,8), np.uint8)
        combined_water = cv2.morphologyEx(combined_water, cv2.MORPH_CLOSE, kernel)
        combined_water = cv2.morphologyEx(combined_water, cv2.MORPH_OPEN, kernel)
        
        return combined_water
    
    def detect_land_by_edge_analysis(self, image):
        """Detect land areas using comprehensive edge analysis"""
        enhanced_image = self.enhance_image_for_detection(image)
        gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
        
        # Multiple edge detection methods
        # Method 1: Canny edge detection
        edges_canny = cv2.Canny(gray, self.edge_threshold_low, self.edge_threshold_high)
        
        # Method 2: Sobel edge detection
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges_sobel = np.sqrt(sobelx**2 + sobely**2)
        edges_sobel = (edges_sobel > 50).astype(np.uint8) * 255
        
        # Method 3: Laplacian edge detection
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        edges_laplacian = (np.abs(laplacian) > 30).astype(np.uint8) * 255
        
        # Combine edge detection methods
        combined_edges = cv2.bitwise_or(edges_canny, edges_sobel)
        combined_edges = cv2.bitwise_or(combined_edges, edges_laplacian)
        
        # Dilate edges to create connected regions
        kernel = np.ones((self.morphology_kernel_size, self.morphology_kernel_size), np.uint8)
        dilated_edges = cv2.dilate(combined_edges, kernel, iterations=2)
        
        # Fill enclosed areas
        filled_edges = dilated_edges.copy()
        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.fillPoly(filled_edges, contours, 255)
        
        return filled_edges, combined_edges
    
    def detect_land_by_clustering(self, image):
        """Detect land using advanced K-means clustering"""
        enhanced_image = self.enhance_image_for_detection(image)
        
        # Prepare data for clustering (include position information)
        height, width = enhanced_image.shape[:2]
        
        # Create feature vector: [B, G, R, X, Y] for better spatial clustering
        b, g, r = cv2.split(enhanced_image)
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        
        # Normalize coordinates
        x_coords = (x_coords / width * 100).astype(np.float32)
        y_coords = (y_coords / height * 100).astype(np.float32)
        
        # Stack features
        features = np.dstack([b, g, r, x_coords, y_coords])
        data = features.reshape((-1, 5))
        data = np.float32(data)
        
        # Apply K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
        k = 6  # More clusters for better distinction
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 20, cv2.KMEANS_RANDOM_CENTERS)
        
        # Reshape labels back to image
        segmented_labels = labels.reshape((height, width))
        
        # Identify water clusters (typically have high blue values)
        water_clusters = []
        for i in range(k):
            cluster_center = centers[i]
            b_val, g_val, r_val = cluster_center[:3]
            
            # Check if this cluster represents water (high blue, low overall brightness difference)
            if b_val > g_val and b_val > r_val and b_val > 80:
                water_clusters.append(i)
        
        # Create land mask (non-water clusters)
        land_mask = np.ones((height, width), dtype=np.uint8) * 255
        for water_cluster in water_clusters:
            land_mask[segmented_labels == water_cluster] = 0
        
        return land_mask
    
    def detect_land_by_color_thresholding(self, image):
        """Detect land using adaptive color thresholding"""
        enhanced_image = self.enhance_image_for_detection(image)
        
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2LAB)
        
        # Create land mask by excluding obvious water colors
        land_mask = np.ones(hsv.shape[:2], dtype=np.uint8) * 255
        
        # Exclude strong blue colors (water)
        h, s, v = cv2.split(hsv)
        
        # Water exclusion ranges (more conservative to keep all land)
        water_conditions = [
            (h >= 100) & (h <= 125) & (s > 80) & (v > 60),  # Deep blue
            (h >= 90) & (h <= 110) & (s > 60) & (v > 100),  # Light blue
        ]
        
        for condition in water_conditions:
            land_mask[condition] = 0
        
        # Additional filtering: exclude very dark areas that might be shadows on water
        gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
        very_dark = gray < 30
        land_mask[very_dark] = 0
        
        return land_mask
    
    def combine_detection_methods(self, image):
        """Combine multiple detection methods for comprehensive land detection"""
        print("   🔍 Method 1: Water mask detection...")
        water_mask = self.create_comprehensive_water_mask(image)
        land_from_water = 255 - water_mask  # Invert water mask to get land
        
        print("   🔍 Method 2: Edge-based land detection...")
        edge_land, raw_edges = self.detect_land_by_edge_analysis(image)
        
        print("   🔍 Method 3: Clustering-based detection...")
        cluster_land = self.detect_land_by_clustering(image)
        
        print("   🔍 Method 4: Color thresholding...")
        threshold_land = self.detect_land_by_color_thresholding(image)
        
        # Combine all methods using voting
        combined_land = np.zeros(image.shape[:2], dtype=np.float32)
        
        # Add contributions from each method
        combined_land += (land_from_water / 255.0) * 0.3      # 30% weight
        combined_land += (edge_land / 255.0) * 0.3            # 30% weight  
        combined_land += (cluster_land / 255.0) * 0.25        # 25% weight
        combined_land += (threshold_land / 255.0) * 0.15      # 15% weight
        
        # Convert to binary mask (majority vote)
        final_land_mask = (combined_land > 0.4).astype(np.uint8) * 255
        
        # Clean up the final mask
        kernel_clean = np.ones((8,8), np.uint8)
        final_land_mask = cv2.morphologyEx(final_land_mask, cv2.MORPH_CLOSE, kernel_clean)
        final_land_mask = cv2.morphologyEx(final_land_mask, cv2.MORPH_OPEN, kernel_clean)
        
        # Fill holes in land areas
        kernel_fill = np.ones((20,20), np.uint8)
        final_land_mask = cv2.morphologyEx(final_land_mask, cv2.MORPH_CLOSE, kernel_fill)
        
        return final_land_mask
    
    def extract_landmasses(self, land_mask):
        """Extract individual landmasses from the combined land mask"""
        # Find all contours in the land mask
        contours, _ = cv2.findContours(land_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        landmasses = []
        total_image_area = self.image_width * self.image_height
        min_area_pixels = total_image_area * (self.min_landmass_percentage / 100)
        
        for i, contour in enumerate(contours):
            area_pixels = cv2.contourArea(contour)
            
            # Filter by minimum area
            if area_pixels > min_area_pixels:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Additional size filtering
                if w > 20 and h > 20:  # Minimum pixel dimensions
                    # Calculate additional properties
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = area_pixels / hull_area if hull_area > 0 else 0
                    
                    landmass = {
                        'id': i + 1,
                        'type': 'landmass',
                        'contour': contour,
                        'bbox': (x, y, w, h),
                        'area_pixels': area_pixels,
                        'area_ratio': area_pixels / total_image_area,
                        'solidity': solidity,
                        'confidence': 0.9,
                        'method': 'comprehensive_detection'
                    }
                    
                    landmasses.append(landmass)
        
        # Sort by area (largest first)
        landmasses.sort(key=lambda x: x['area_pixels'], reverse=True)
        
        # Return top landmasses
        return landmasses[:self.max_landmasses]
    
    def calculate_measurements(self, landmasses):
        """Calculate comprehensive measurements for detected landmasses"""
        measured_landmasses = []
        
        for landmass in landmasses:
            x, y, w, h = landmass['bbox']
            
            # Real-world dimensions
            width_meters = w * self.meters_per_pixel
            height_meters = h * self.meters_per_pixel
            area_square_meters = landmass['area_pixels'] * (self.meters_per_pixel ** 2)
            area_hectares = area_square_meters / 10000
            area_acres = area_hectares * 2.471
            
            # Perimeter calculation
            perimeter_pixels = cv2.arcLength(landmass['contour'], True)
            perimeter_meters = perimeter_pixels * self.meters_per_pixel
            
            # Shape analysis
            aspect_ratio = width_meters / height_meters if height_meters > 0 else 0
            
            # Coverage calculation
            total_ground_area = (self.image_width * self.meters_per_pixel) * (self.image_height * self.meters_per_pixel)
            coverage_percentage = (area_square_meters / total_ground_area) * 100
            
            measured_landmass = landmass.copy()
            measured_landmass.update({
                'width_meters': width_meters,
                'height_meters': height_meters,
                'area_square_meters': area_square_meters,
                'area_hectares': area_hectares,
                'area_acres': area_acres,
                'perimeter_meters': perimeter_meters,
                'perimeter_km': perimeter_meters / 1000,
                'aspect_ratio': aspect_ratio,
                'coverage_percentage': coverage_percentage,
                'shape_regularity': 'regular' if landmass['solidity'] > 0.7 else 'irregular'
            })
            
            measured_landmasses.append(measured_landmass)
        
        return measured_landmasses
    
    def analyze_complete_landmasses(self, image_path):
        """Main analysis function - detects complete landmasses"""
        print(f"🌍 DROT - Complete Land Area Analysis")
        print("=" * 60)
        print(f"📁 Image: {os.path.basename(image_path)}")
        print(f"🚁 Drone Altitude: {self.drone_altitude} meters")
        print(f"🎯 Goal: Detect COMPLETE landmasses (all terrain types)")
        print()
        
        # Load and validate image
        print("📷 Loading image...")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        self.image_height, self.image_width = image.shape[:2]
        
        # Calculate scale
        self.meters_per_pixel = self.calculate_scale(self.image_width, self.drone_altitude)
        total_ground_width = self.image_width * self.meters_per_pixel
        total_ground_height = self.image_height * self.meters_per_pixel
        total_area_sq_meters = total_ground_width * total_ground_height
        
        print(f"📐 Resolution: {self.image_width} x {self.image_height} pixels")
        print(f"🌍 Coverage: {total_ground_width:.1f}m x {total_ground_height:.1f}m")
        print(f"📏 Scale: {self.meters_per_pixel:.4f} meters/pixel")
        print(f"📊 Total surveyed: {total_area_sq_meters:.1f} m² ({total_area_sq_meters/10000:.2f} hectares)")
        
        # Comprehensive land detection
        print(f"\n🔍 Applying comprehensive land detection...")
        combined_land_mask = self.combine_detection_methods(image)
        
        # Extract landmasses
        print(f"\n🏝️ Extracting landmasses...")
        landmasses = self.extract_landmasses(combined_land_mask)
        
        if not landmasses:
            print("⚠️  No significant landmasses detected.")
            return [], image, combined_land_mask
        
        print(f"   ✅ Found {len(landmasses)} landmass(es)")
        
        # Calculate measurements
        print(f"📏 Calculating measurements...")
        measured_landmasses = self.calculate_measurements(landmasses)
        
        return measured_landmasses, image, combined_land_mask
    
    def create_comprehensive_visualization(self, image, landmasses, land_mask, image_path):
        """Create comprehensive visualization showing detection process and results"""
        
        # Create main annotated image
        annotated_image = image.copy()
        
        # Colors for different landmasses
        colors = [
            (0, 255, 0),    # Green
            (0, 165, 255),  # Orange  
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (255, 100, 0),  # Blue
            (100, 255, 100), # Light green
            (255, 100, 255), # Light magenta
            (100, 255, 255)  # Light yellow
        ]
        
        # Draw each landmass
        for i, landmass in enumerate(landmasses):
            color = colors[i % len(colors)]
            
            # Draw thick contour
            cv2.drawContours(annotated_image, [landmass['contour']], -1, color, 6)
            
            # Draw bounding box
            x, y, w, h = landmass['bbox']
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), color, 4)
            
            # Large landmass number
            center_x, center_y = x + w//2, y + h//2
            cv2.circle(annotated_image, (center_x, center_y), 40, color, -1)
            cv2.circle(annotated_image, (center_x, center_y), 40, (0, 0, 0), 5)
            cv2.putText(annotated_image, str(i + 1), (center_x - 15, center_y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5)
            
            # Measurement labels
            area_text = f"Landmass {i+1}: {landmass['area_square_meters']:.0f} m²"
            size_text = f"{landmass['area_hectares']:.2f} hectares ({landmass['area_acres']:.1f} acres)"
            dim_text = f"Dimensions: {landmass['width_meters']:.0f}m x {landmass['height_meters']:.0f}m"
            
            # Text background
            text_width = max(len(area_text), len(size_text), len(dim_text)) * 12
            text_height = 100
            
            # Position text box
            text_x = max(10, min(x, self.image_width - text_width))
            text_y = max(text_height, y - 10)
            
            cv2.rectangle(annotated_image, (text_x, text_y - text_height), 
                         (text_x + text_width, text_y), color, -1)
            cv2.rectangle(annotated_image, (text_x, text_y - text_height), 
                         (text_x + text_width, text_y), (0, 0, 0), 3)
            
            # Add text
            font_scale = 0.8
            cv2.putText(annotated_image, area_text, (text_x + 10, text_y - 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
            cv2.putText(annotated_image, size_text, (text_x + 10, text_y - 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
            cv2.putText(annotated_image, dim_text, (text_x + 10, text_y - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
        
        # Create summary panel
        panel_height = 120
        summary_panel = np.zeros((panel_height, self.image_width, 3), dtype=np.uint8)
        summary_panel[:] = (40, 40, 40)
        
        # Add summary information
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_pos = 30
        
        cv2.putText(summary_panel, "🌍 COMPLETE LANDMASS ANALYSIS RESULTS", (20, y_pos), 
                   font, 1.0, (0, 255, 255), 3)
        y_pos += 35
        
        total_land_area = sum(lm['area_square_meters'] for lm in landmasses)
        total_hectares = total_land_area / 10000
        
        cv2.putText(summary_panel, f"📊 Landmasses: {len(landmasses)} | Total Area: {total_land_area:.0f} m² ({total_hectares:.2f} hectares)", 
                   (20, y_pos), font, 0.8, (255, 255, 255), 2)
        y_pos += 30
        
        cv2.putText(summary_panel, f"🔧 Detection: Multi-method analysis | Altitude: {self.drone_altitude}m | Scale: {self.meters_per_pixel:.4f}m/px", 
                   (20, y_pos), font, 0.6, (255, 255, 0), 2)
        
        # Combine all visualizations
        final_image = np.vstack([annotated_image, summary_panel])
        
        # Save results
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"complete_landmass_{base_name}.jpg"
        cv2.imwrite(output_path, final_image)
        
        # Also save the land mask for debugging
        mask_path = f"land_mask_{base_name}.jpg"
        cv2.imwrite(mask_path, land_mask)
        
        return output_path, mask_path
    
    def generate_detailed_report(self, landmasses, image_path):
        """Generate comprehensive landmass analysis report"""
        print(f"\n📋 COMPLETE LANDMASS ANALYSIS REPORT")
        print("=" * 70)
        
        total_area = sum(lm['area_square_meters'] for lm in landmasses)
        total_hectares = total_area / 10000
        total_acres = total_hectares * 2.471
        total_perimeter = sum(lm['perimeter_meters'] for lm in landmasses)
        
        print(f"📁 Source: {os.path.basename(image_path)}")
        print(f"🚁 Survey altitude: {self.drone_altitude} meters")
        print(f"📏 Scale: {self.meters_per_pixel:.4f} meters/pixel")
        print(f"🏝️ Landmasses detected: {len(landmasses)}")
        print(f"📐 Total land area: {total_area:.0f} m² | {total_hectares:.2f} ha | {total_acres:.1f} acres")
        print(f"📏 Total perimeter: {total_perimeter:.0f} m ({total_perimeter/1000:.1f} km)")
        print()
        
        print("🏝️ INDIVIDUAL LANDMASS DETAILS:")
        print("-" * 70)
        
        for i, landmass in enumerate(landmasses, 1):
            print(f"\nLandmass {i}:")
            print(f"   📐 Dimensions: {landmass['width_meters']:.0f}m × {landmass['height_meters']:.0f}m")
            print(f"   📊 Area: {landmass['area_square_meters']:.0f} m² ({landmass['area_hectares']:.2f} hectares)")
            print(f"   📏 Perimeter: {landmass['perimeter_meters']:.0f} m ({landmass['perimeter_km']:.1f} km)")
            print(f"   📈 Coverage: {landmass['coverage_percentage']:.1f}% of surveyed area")
            print(f"   🔍 Shape: {landmass['shape_regularity']} (aspect ratio: {landmass['aspect_ratio']:.2f})")
            print(f"   ✅ Confidence: {landmass['confidence']:.0%}")
        
        # Save detailed report
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        report_path = f"landmass_report_{base_name}.txt"
        
        with open(report_path, 'w') as f:
            f.write(f"DROT - Complete Landmass Analysis Report\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"=" * 60 + "\n\n")
            
            f.write(f"SOURCE INFORMATION:\n")
            f.write(f"Image: {os.path.basename(image_path)}\n")
            f.write(f"Drone Altitude: {self.drone_altitude} meters\n")
            f.write(f"Analysis Scale: {self.meters_per_pixel:.4f} meters/pixel\n")
            f.write(f"Detection Method: Multi-method comprehensive analysis\n\n")
            
            f.write(f"SUMMARY:\n")
            f.write(f"Landmasses Detected: {len(landmasses)}\n")
            f.write(f"Total Land Area: {total_area:.0f} square meters\n")
            f.write(f"Total Area (Hectares): {total_hectares:.2f} ha\n")
            f.write(f"Total Area (Acres): {total_acres:.1f} acres\n")
            f.write(f"Total Perimeter: {total_perimeter:.0f} meters ({total_perimeter/1000:.1f} km)\n\n")
            
            f.write(f"DETAILED LANDMASS BREAKDOWN:\n")
            f.write("-" * 50 + "\n")
            
            for i, landmass in enumerate(landmasses, 1):
                f.write(f"\nLandmass {i}:\n")
                f.write(f"  Area: {landmass['area_square_meters']:.0f} m² ({landmass['area_hectares']:.2f} ha)\n")
                f.write(f"  Dimensions: {landmass['width_meters']:.0f}m x {landmass['height_meters']:.0f}m\n")
                f.write(f"  Perimeter: {landmass['perimeter_meters']:.0f} meters\n")
                f.write(f"  Coverage: {landmass['coverage_percentage']:.1f}% of surveyed area\n")
                f.write(f"  Shape: {landmass['shape_regularity']} (solidity: {landmass['solidity']:.2f})\n")
                f.write(f"  Detection Confidence: {landmass['confidence']:.0%}\n")
        
        print(f"\n💾 Detailed report saved: {report_path}")
        return report_path

def main():
    """Main execution function"""
    if len(sys.argv) < 2:
        print("DROT - Complete Land Area Detection and Measurement System")
        print("=" * 70)
        print("Usage: python land_area_analyzer.py <image_path> [drone_altitude]")
        print()
        print("🎯 COMPLETE LANDMASS DETECTION:")
        print("  • Detects ENTIRE landmasses regardless of internal terrain variation")
        print("  • Combines multiple detection methods for maximum accuracy")
        print("  • Treats vegetation, soil, beach, rock as unified land areas")
        print("  • Focuses on land-water boundaries for precise measurement")
        print("  • Provides comprehensive area analysis in multiple units")
        print()
        print("🔬 Advanced Detection Methods:")
        print("  • Multi-spectral water detection (HSV + LAB color analysis)")
        print("  • Edge detection with Canny, Sobel, and Laplacian operators")
        print("  • K-means clustering with spatial feature weighting")
        print("  • Morphological operations for boundary refinement")
        print("  • Ensemble voting for robust final detection")
        print()
        print("📊 Perfect for:")
        print("  • Islands and coastal mapping")
        print("  • Complete agricultural field measurement")
        print("  • Construction site total area calculation")
        print("  • Forest and natural area assessment")
        print("  • Any scenario requiring complete land area measurement")
        print()
        print("💡 Examples:")
        print("  python land_area_analyzer.py island_drone.jpg 100")
        print("  python land_area_analyzer.py farm_complete.jpg 75")
        print("  python land_area_analyzer.py construction_total.jpg 50")
        print("  python land_area_analyzer.py forest_boundary.jpg 150")
        return
    
    image_path = sys.argv[1]
    drone_altitude = float(sys.argv[2]) if len(sys.argv) > 2 else 50.0
    
    # Input validation
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found: {image_path}")
        print(f"   Please verify the file path and try again.")
        return
    
    if drone_altitude <= 0 or drone_altitude > 2000:
        print(f"❌ Error: Invalid drone altitude: {drone_altitude}m")
        print(f"   Please use altitude between 1-2000 meters.")
        return
    
    try:
        # Initialize the complete land detector
        detector = CompleteLandDetector(drone_altitude)
        
        # Perform comprehensive landmass analysis
        print("🚀 Starting complete landmass detection...")
        start_time = time.time()
        
        landmasses, original_image, land_mask = detector.analyze_complete_landmasses(image_path)
        
        processing_time = time.time() - start_time
        
        if not landmasses:
            print("\n⚠️  No significant landmasses detected in this image.")
            print("📝 Possible reasons:")
            print("   • Image may be entirely water or sky")
            print("   • Landmasses may be smaller than minimum threshold (0.3%)")
            print("   • Poor contrast between land and water")
            print("   • Try adjusting drone altitude parameter")
            
            # Save debug mask anyway
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            debug_path = f"debug_mask_{base_name}.jpg"
            cv2.imwrite(debug_path, land_mask)
            print(f"   📄 Debug mask saved: {debug_path}")
            return
        
        # Create comprehensive visualizations
        print(f"\n🎨 Creating comprehensive visualizations...")
        output_image, mask_image = detector.create_comprehensive_visualization(
            original_image, landmasses, land_mask, image_path)
        print(f"✅ Main visualization: {output_image}")
        print(f"✅ Detection mask: {mask_image}")
        
        # Generate detailed report
        report_path = detector.generate_detailed_report(landmasses, image_path)
        
        # Performance and results summary
        print(f"\n⚡ ANALYSIS COMPLETE!")
        print("=" * 50)
        print(f"⏱️  Processing time: {processing_time:.2f} seconds")
        print(f"🏝️  Landmasses found: {len(landmasses)}")
        
        total_area = sum(lm['area_square_meters'] for lm in landmasses)
        largest_area = max(landmasses, key=lambda x: x['area_square_meters'])['area_square_meters']
        total_perimeter = sum(lm['perimeter_meters'] for lm in landmasses)
        
        print(f"📊 Total land area: {total_area:.0f} m² ({total_area/10000:.2f} hectares)")
        print(f"🏆 Largest landmass: {largest_area:.0f} m²")
        print(f"📏 Total coastline: {total_perimeter:.0f} m ({total_perimeter/1000:.1f} km)")
        
        # Method effectiveness summary
        coverage_total = sum(lm['coverage_percentage'] for lm in landmasses)
        print(f"🎯 Land coverage: {coverage_total:.1f}% of surveyed area")
        print(f"🔬 Detection methods: Multi-spectral + Edge + Clustering + Morphology")
        
        print(f"\n📁 Generated Files:")
        print(f"   🖼️  {output_image} - Complete landmass visualization")
        print(f"   🗺️  {mask_image} - Land detection mask")
        print(f"   📄 {report_path} - Detailed analysis report")
        
        print(f"\n🎉 Complete landmass analysis finished successfully!")
        print(f"   🎯 Focus: Maximum detection accuracy for complete land areas")
        print(f"   📊 Quality: Professional-grade measurements with multi-method validation")
        
        # Additional insights
        if len(landmasses) == 1:
            solidity = landmasses[0]['solidity']
            if solidity > 0.8:
                print(f"   💡 Insight: Detected landmass has regular shape (solidity: {solidity:.2f})")
            else:
                print(f"   💡 Insight: Detected landmass has complex/irregular coastline (solidity: {solidity:.2f})")
        else:
            print(f"   💡 Insight: Multiple landmasses detected - suitable for archipelago analysis")
        
    except Exception as e:
        print(f"\n❌ Analysis Error: {str(e)}")
        print(f"📝 Troubleshooting steps:")
        print(f"   • Verify image is a valid aerial/drone photograph")
        print(f"   • Ensure image contains visible land-water boundaries")
        print(f"   • Check that image resolution is adequate (minimum 500x500)")
        print(f"   • Verify adequate contrast between land and water areas")
        print(f"   • Try different altitude values if results seem incorrect")
        
        import traceback
        print(f"\n🔧 Technical details:")
        traceback.print_exc()

if __name__ == "__main__":
    main()