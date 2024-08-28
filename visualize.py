import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from model import Model
import cv2
from scipy.fftpack import dct

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

    # DCT
    dct_filtered = dct(dct(img.T, norm='ortho').T, norm='ortho')
    dct_filtered = high_pass_filter(dct_filtered, threshold=threshold)

    laplacian = laplacian_filter(img)
    # print(img.shape, dct_filtered.shape, laplacian.shape)
    
    return [img, dct_filtered, laplacian]

def visualize_prediction(model, image_path, mask_path, device, threshold=10):
    """
    Visualize the original image, ground truth mask, and predicted mask.

    Parameters:
    model (nn.Module): The trained model.
    image_path (str): Path to the test image.
    mask_path (str): Path to the ground truth mask.
    device (torch.device): The device to run the model on.
    threshold (int): Threshold for high-pass filtering in preprocessing.
    """
    # Load and preprocess the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = Image.open(mask_path).convert("L")
    preprocessed_images = preprocessing(image, threshold=threshold)
    original, dct_filtered, laplacian = preprocessed_images
    combined_input = torch.stack([
        torch.tensor(original).float(),
        torch.tensor(dct_filtered).float(),
        torch.tensor(laplacian).float()
    ], dim=0).unsqueeze(0).to(device)

    # Predict the mask
    with torch.no_grad():
        output = model(combined_input)
        # output = torch.sigmoid(output)
        pred_mask = (output).float().cpu().numpy().squeeze()

    # Plot the original image, ground truth mask, and predicted mask
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    axs[1].imshow(mask, cmap='gray')
    axs[1].set_title('Ground Truth Mask')
    axs[1].axis('off')
    
    axs[2].imshow(pred_mask, cmap='gray')
    axs[2].set_title('Predicted Mask')
    axs[2].axis('off')
    
    plt.show()

# Example Usage:
device = "mps"
model = Model(num_classes=1)  # Replace with your model class
model_path = 'models/model_9000IM_50EP.pth'  # Path to the trained model

# Load the trained model
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
# Visualize a prediction
test_image_path = '/Users/hiteshgupta/Documents/ML-CV/Image-Segement-Forgery/Dataset/train/img/img/0_000000009069.tif'
test_mask_path = '/Users/hiteshgupta/Documents/ML-CV/Image-Segement-Forgery/Dataset/train/donor_mask/0_000000009069.tif'
visualize_prediction(model, test_image_path, test_mask_path, device, threshold=10)
