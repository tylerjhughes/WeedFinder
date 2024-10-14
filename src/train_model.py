import torch
from ultralytics import YOLO
import numpy as np
from preprocess_batch import Preprocess_batch

def train_model():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = YOLO('yolov11n') 
    model.to(device)

    dataset = np.load('data/custom_dataset.npz')
    dataset_preprocessed = Preprocess_batch(dataset)

    model.train(
        data=dataset_preprocessed,  # Path to dataset YAML
        epochs=50,  
        imgsz=640,  
        batch=16,  
        device=device
    )

    model.export(format='onnx')

if __name__ == '__main__':
    train_model()
