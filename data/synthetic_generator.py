"""
Synthetic Data Generator for RareCan-AI
Generates synthetic histopathology-like images for few-shot learning experiments.
This is a placeholder for real data preprocessing when using public datasets.
"""

import numpy as np
import os
from PIL import Image, ImageDraw, ImageFilter
import json
from typing import Dict, List, Tuple
import random


class SyntheticHistopathologyGenerator:
    """Generates synthetic histopathology images for testing purposes."""
    
    def __init__(self, output_dir: str = "data/sample", seed: int = 42):
        """
        Args:
            output_dir: Directory to save generated images
            seed: Random seed for reproducibility
        """
        self.output_dir = output_dir
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Rare cancer types (simplified for demonstration)
        self.cancer_types = {
            "pancreatic_neuroendocrine": {
                "color_range": [(180, 120, 100), (220, 180, 160)],
                "pattern": "nested"
            },
            "sarcoma_subtype_a": {
                "color_range": [(150, 100, 80), (200, 150, 120)],
                "pattern": "spindle"
            },
            "medullary_thyroid": {
                "color_range": [(160, 140, 120), (210, 190, 170)],
                "pattern": "solid"
            },
            "gastrointestinal_stromal": {
                "color_range": [(170, 130, 110), (230, 200, 180)],
                "pattern": "mixed"
            },
            "chondrosarcoma": {
                "color_range": [(140, 110, 90), (190, 160, 140)],
                "pattern": "lobular"
            }
        }
    
    def generate_image(self, cancer_type: str, image_id: str, 
                      width: int = 224, height: int = 224) -> Image.Image:
        """
        Generate a synthetic histopathology-like image.
        
        Args:
            cancer_type: Type of rare cancer
            image_id: Unique identifier for the image
            width: Image width
            height: Image height
            
        Returns:
            PIL Image object
        """
        if cancer_type not in self.cancer_types:
            raise ValueError(f"Unknown cancer type: {cancer_type}")
        
        config = self.cancer_types[cancer_type]
        color_low, color_high = config["color_range"]
        pattern = config["pattern"]
        
        # Create base image with H&E-like staining
        img = Image.new('RGB', (width, height), color=(240, 230, 220))
        draw = ImageDraw.Draw(img)
        
        # Generate pattern based on cancer type
        if pattern == "nested":
            self._draw_nested_pattern(draw, width, height, color_low, color_high)
        elif pattern == "spindle":
            self._draw_spindle_pattern(draw, width, height, color_low, color_high)
        elif pattern == "solid":
            self._draw_solid_pattern(draw, width, height, color_low, color_high)
        elif pattern == "mixed":
            self._draw_mixed_pattern(draw, width, height, color_low, color_high)
        elif pattern == "lobular":
            self._draw_lobular_pattern(draw, width, height, color_low, color_high)
        
        # Add noise and texture
        img = self._add_texture(img)
        
        return img
    
    def _draw_nested_pattern(self, draw, width, height, color_low, color_high):
        """Draw nested pattern typical of neuroendocrine tumors."""
        for i in range(15):
            center_x = random.randint(20, width - 20)
            center_y = random.randint(20, height - 20)
            radius = random.randint(10, 30)
            
            color = tuple(random.randint(c, d) for c, d in zip(color_low, color_high))
            draw.ellipse([center_x - radius, center_y - radius,
                         center_x + radius, center_y + radius],
                        fill=color, outline=None)
    
    def _draw_spindle_pattern(self, draw, width, height, color_low, color_high):
        """Draw spindle pattern typical of sarcomas."""
        for i in range(20):
            x1 = random.randint(0, width)
            y1 = random.randint(0, height)
            x2 = x1 + random.randint(-40, 40)
            y2 = y1 + random.randint(-20, 20)
            
            color = tuple(random.randint(c, d) for c, d in zip(color_low, color_high))
            draw.line([x1, y1, x2, y2], fill=color, width=random.randint(2, 5))
    
    def _draw_solid_pattern(self, draw, width, height, color_low, color_high):
        """Draw solid pattern."""
        for i in range(12):
            x = random.randint(0, width - 50)
            y = random.randint(0, height - 50)
            w = random.randint(30, 60)
            h = random.randint(30, 60)
            
            color = tuple(random.randint(c, d) for c, d in zip(color_low, color_high))
            draw.rectangle([x, y, x + w, y + h], fill=color, outline=None)
    
    def _draw_mixed_pattern(self, draw, width, height, color_low, color_high):
        """Draw mixed pattern."""
        self._draw_nested_pattern(draw, width, height, color_low, color_high)
        self._draw_spindle_pattern(draw, width, height, color_low, color_high)
    
    def _draw_lobular_pattern(self, draw, width, height, color_low, color_high):
        """Draw lobular pattern."""
        for i in range(10):
            center_x = random.randint(30, width - 30)
            center_y = random.randint(30, height - 30)
            radius = random.randint(20, 50)
            
            color = tuple(random.randint(c, d) for c, d in zip(color_low, color_high))
            # Draw multiple overlapping circles
            for offset in range(0, radius, 5):
                draw.ellipse([center_x - radius + offset, center_y - radius + offset,
                             center_x + radius - offset, center_y + radius - offset],
                            fill=None, outline=color, width=2)
    
    def _add_texture(self, img: Image.Image) -> Image.Image:
        """Add texture and noise to make image more realistic."""
        # Convert to numpy array
        img_array = np.array(img).astype(np.float32)
        
        # Add Gaussian noise
        noise = np.random.normal(0, 5, img_array.shape).astype(np.float32)
        img_array = np.clip(img_array + noise, 0, 255)
        
        # Apply slight blur
        img = Image.fromarray(img_array.astype(np.uint8))
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        return img
    
    def generate_clinical_data(self, image_id: str, cancer_type: str) -> Dict:
        """Generate synthetic clinical data for an image."""
        return {
            "image_id": image_id,
            "cancer_type": cancer_type,
            "age": random.randint(30, 80),
            "gender": random.choice(["M", "F"]),
            "tumor_stage": random.choice(["I", "II", "III", "IV"]),
            "genetic_mutation": random.choice(["BRAF", "KRAS", "TP53", "None"]),
            "tumor_size_mm": round(random.uniform(10, 100), 1)
        }
    
    def generate_dataset(self, samples_per_class: int = 50) -> Dict[str, List[Dict]]:
        """
        Generate a synthetic dataset with specified samples per class.
        
        Args:
            samples_per_class: Number of samples to generate for each cancer type
            
        Returns:
            Dictionary mapping cancer types to lists of metadata
        """
        dataset_metadata = {}
        
        for cancer_type in self.cancer_types.keys():
            metadata_list = []
            
            for i in range(samples_per_class):
                image_id = f"{cancer_type}_{i:04d}"
                
                # Generate image
                img = self.generate_image(cancer_type, image_id)
                img_path = os.path.join(self.output_dir, f"{image_id}.png")
                img.save(img_path)
                
                # Generate clinical data
                clinical_data = self.generate_clinical_data(image_id, cancer_type)
                clinical_data["image_path"] = img_path
                metadata_list.append(clinical_data)
            
            dataset_metadata[cancer_type] = metadata_list
        
        # Save metadata
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(dataset_metadata, f, indent=2)
        
        print(f"Generated {samples_per_class} samples per class for {len(self.cancer_types)} classes")
        print(f"Total images: {samples_per_class * len(self.cancer_types)}")
        print(f"Metadata saved to: {metadata_path}")
        
        return dataset_metadata


if __name__ == "__main__":
    generator = SyntheticHistopathologyGenerator()
    generator.generate_dataset(samples_per_class=50)

