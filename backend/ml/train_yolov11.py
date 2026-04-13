"""
YOLOv11 Training Script for Floor Plan Element Detection
"""

from ultralytics import YOLO
import os
import yaml
from pathlib import Path
from loguru import logger

def train_yolo_v11():
    # 1. Configuration
    dataset_config = {
        'path': '../datasets/floorplan_dataset', # Path to dataset root
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {
            0: 'wall',
            1: 'door',
            2: 'window',
            3: 'stair',
            4: 'room',
            5: 'fixture'
        }
    }
    
    # Save dataset.yaml
    config_path = 'floorplan_dataset.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(dataset_config, f)
    
    logger.info(f"Created dataset config at {config_path}")
    
    # 2. Load Model
    # Note: YOLOv11 weights might need to be downloaded or specified
    # Assuming 'yolo11n.pt' is the base model
    model = YOLO('yolo11n.pt') 
    
    # 3. Training
    logger.info("Starting YOLOv11 training...")
    results = model.train(
        data=config_path,
        epochs=100,
        imgsz=640,
        batch=16,
        name='yolov11_floorplan_v1',
        device=0, # Use GPU 0, or 'cpu'
        patience=20,
        save=True,
        cache=True
    )
    
    logger.info("Training complete!")
    
    # 4. Export weights
    export_path = model.export(format='pt')
    logger.info(f"Exported model to {export_path}")
    
    # Move to weights directory
    target_weights = 'weights/yolov11_floorplan.pt'
    Path('weights').mkdir(exist_ok=True)
    os.rename(export_path, target_weights)
    logger.info(f"Final weights saved to {target_weights}")

if __name__ == "__main__":
    train_yolo_v11()
