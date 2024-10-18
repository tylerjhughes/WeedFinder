import torch
from ultralytics import YOLO
import numpy as np
from preprocess_batch import Preprocess_batch

def train_model():


    model = YOLO('yolov11n') 

    dataset = np.load('data/dataset_preprocessed.yaml')

    model.train(
        data=dataset,  # Path to dataset YAML
        epochs=50,  
        imgsz=640,  
        batch=16,  
        device='cuda'
    )

    model.export(format='onnx')

if __name__ == '__main__':
    train_model()
