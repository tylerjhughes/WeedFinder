import torch
from ultralytics import YOLO
import numpy as np
from skimage.transform import resize

def preprocess(self, image_batch):
        # square image
        min_dim = min(image_batch.shape[2], image_batch.shape[3])
        centre_max = max(image_batch.shape[2], image_batch.shape[3]) // 2 
        image_batch = image_batch[:, :, centre_max - min_dim:centre_max + min_dim, :]

        # downsample image
        image_batch = resize(image_batch, (16, 3, 640, 640), anti_aliasing=True, preserve_range=True)

        # mean normalisation on batch with shape (16, 3, 640, 640)
        mean = np.mean(image_batch, axis=(2, 3), keepdims=True)
        std = np.std(image_batch, axis=(2, 3), keepdims=True)

        return (image_batch - mean) / std
def train_model():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = YOLO('yolov11n') 
    model.to(device)

    dataset = np.load('data/custom_dataset.npz')
    dataset_preprocessed = preprocess(dataset)

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
