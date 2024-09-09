import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import jaccard_score
import numpy as np
from tqdm import tqdm
from model import Model
import os
from PIL import Image
import cv2
from scipy.fftpack import dct

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.models import Model

def apply_srm_filters(img):
    # Define SRM filters
    srm_filters = np.array([
        [[ 0, 0, 0, 0, 0], [ 0, -1, 2, -1, 0], [ 0, 2, -4, 2, 0], [ 0, -1, 2, -1, 0], [ 0, 0, 0, 0, 0]],  
        [[-1, 2, -2, 2, -1], [ 2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [ 2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]],  
        [[ 0, 0, 0, 0, 0], [ 0, 0, 1, 0, 0], [ 0, 1, -2, 1, 0], [ 0, 0, 1, 0, 0], [ 0, 0, 0, 0, 0]]  
    ], dtype=np.float32)

    srm_filters[0] /= 4.0
    srm_filters[1] /= 12.0
    srm_filters[2] /= 2.0

    srm_filters = np.reshape(srm_filters, (5, 5, 1, 3))
    srm_filters = np.repeat(srm_filters, 3, axis=2)

    # Create model with SRM filters
    input_image = Input(shape=(None, None, 3)) 
    x = Conv2D(filters=3, kernel_size=(5, 5), padding='same', use_bias=False, kernel_initializer=tf.constant_initializer(srm_filters))(input_image)
    model = Model(inputs=input_image, outputs=x)

    # Load image using OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR (OpenCV) to RGB (matplotlib)
    img_resized = cv2.resize(img, (512, 512))  # Resize to 512x512 if needed
    img_array = np.expand_dims(img_resized, axis=0)

    # Get SRM filter output
    srm_output = model.predict(img_array)

    return srm_output[0, :, :, 0]


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, threshold=10):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_names = os.listdir(image_dir)
        self.threshold = threshold

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        mask_path = os.path.join(self.mask_dir, self.image_names[idx])  # Assuming masks have the same names as images

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale

        # Apply preprocessing
        preprocessed_images = preprocessing(image, threshold=self.threshold)

        # If transformations are provided, apply them to each preprocessed image and mask
        if self.transform:
            preprocessed_images = [self.transform(Image.fromarray(img)) for img in preprocessed_images]
            mask = self.transform(mask)

        return preprocessed_images, mask

def high_pass_filter(dct_coeffs, threshold=10):
    """
    Applies a high-pass filter to the DCT coefficients.
    
    Parameters:
    dct_coeffs (numpy.ndarray): DCT coefficients of the image.
    threshold (int): Threshold for high-pass filtering.
    
    Returns:
    numpy.ndarray: High-pass filtered DCT coefficients.
    """
    # Zero out low-frequency components below the threshold
    h, w = dct_coeffs.shape
    dct_coeffs[:threshold, :threshold] = 0
    return dct_coeffs

def laplacian_filter(image):
    """
    Applies a Laplacian filter to the input grayscale image.
    
    Parameters:
    image (numpy.ndarray): Input image in grayscale.
    
    Returns:
    numpy.ndarray: Image after applying the Laplacian filter.
    """
    # Apply the Laplacian filter
    laplacian_filtered = cv2.Laplacian(image, cv2.CV_64F)
    return laplacian_filtered

def preprocessing(image, threshold=10):
    img = cv2.resize(image, (224, 224))

    # SRM filter
    srm_output = apply_srm_filters(img)
    # DCT converting to YCrCb
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(img_ycrcb)
    dct_filtered = dct(dct(Y.T, norm='ortho').T, norm='ortho')
    dct_filtered = high_pass_filter(dct_filtered, threshold=threshold)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = laplacian_filter(img)
    # print(img.shape, dct_filtered.shape, laplacian.shape)
    
    return [srm_output, dct_filtered, laplacian]


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for preprocessed_images, masks in tqdm(dataloader):
        # Unpack preprocessed images
        original, dct_filtered, laplacian = preprocessed_images
        original = original.squeeze(1).float().to(device).requires_grad_(True)
        dct_filtered = dct_filtered.squeeze(1).float().to(device).requires_grad_(True)
        laplacian = laplacian.squeeze(1).to(device).float().requires_grad_(True)
        combined_input = torch.stack([original, dct_filtered, laplacian], dim=1).to(device)
        
        # Normalize the input
        # combined_input = combined_input / 255.0
        
        # print(combined_input.shape, "combined_input")
        masks = masks.float().to(device)
        # masks = masks/255.0
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(combined_input)
        
        # Calculate loss
        loss = criterion(outputs, masks)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Accumulate the loss
        running_loss += loss.item() * combined_input.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    iou_scores = []
    
    with torch.no_grad():
        for preprocessed_images, masks in tqdm(dataloader):
            original, dct_filtered, laplacian = preprocessed_images
            original = original.squeeze(1).float().to(device).requires_grad_(True)
            dct_filtered = dct_filtered.squeeze(1).float().to(device).requires_grad_(True)
            laplacian = laplacian.squeeze(1).to(device).float().requires_grad_(True)
            combined_input = torch.stack([original, dct_filtered, laplacian], dim=1).to(device)
            
            # Normalize the input
            # combined_input = combined_input / 255.0
            
            # print(combined_input.shape, "combined_input")
            masks = masks.float().to(device)
            
            outputs = model(combined_input)
            
            # Calculate loss
            loss = criterion(outputs, masks)
            running_loss += loss.item() * combined_input.size(0)
            
            # Calculate IoU for binary segmentation
            preds = (outputs > 0.5).float()
            binary_masks = (masks > 0.5).float()
            iou = jaccard_score(binary_masks.cpu().numpy().flatten(), preds.cpu().numpy().flatten(), average='binary')
            iou_scores.append(iou)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    mean_iou = np.mean(iou_scores)
    return epoch_loss, mean_iou

