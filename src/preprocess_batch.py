import numpy as np
from skimage.transform import resize
def Preprocess_batch(image_batch):
        ''' Resizes and mean normalises an image_batch for the YOLO model'''
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

if __name__ == '__main__':
    preprocess()