from main import train, validate, SegmentationDataset
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

if __name__ == '__main__':
    num_epochs = 10
    # Paths to your image and mask directories
    image_dir = 'Dataset/val/img'
    mask_dir = 'Dataset/val/mask'

    # Create Dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = SegmentationDataset(image_dir, mask_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    # Instantiate the model, criterion, and optimizer
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(num_classes=1).to(device)

    criterion = nn.BCEWithLogitsLoss()  # Suitable for binary segmentation
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for param in model.parameters():
        param.requires_grad = True
            
    for epoch in range(num_epochs):
        train_loss = train(model, dataloader, criterion, optimizer, device)
        val_loss, val_iou = validate(model, dataloader, criterion, device)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f} | Validation IoU: {val_iou:.4f}')
        
    # Save the model
    torch.save(model.state_dict(), 'models/model3.pth')

