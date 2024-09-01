import cv2
import numpy as np
from scipy.fftpack import dct, idct

def high_pass_filter(dct_coeffs, threshold=1):
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
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))

    # DCT
    dct_filtered = dct(dct(img.T, norm='ortho').T, norm='ortho')
    dct_filtered = high_pass_filter(dct_filtered, threshold = threshold)

    laplacian = laplacian_filter(img)
    
    
    return img, dct_filtered, laplacian

def dct_YCbCr(image, threshold=10):
    """
    Applies a high-pass filter to the DCT coefficients.
    
    Parameters:
    dct_coeffs (numpy.ndarray): DCT coefficients of the image.
    threshold (int): Threshold for high-pass filtering.
    
    Returns:
    numpy.ndarray: High-pass filtered DCT coefficients.
    """
    # Zero out low-frequency components below the threshold
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(img)
    dct_Y = dct(dct(Y.T, norm='ortho').T, norm='ortho')
    dct_Y = high_pass_filter(dct_Y, threshold = threshold)
    dct_Cr = dct(dct(Cr.T, norm='ortho').T, norm='ortho')
    # dct_Cr = high_pass_filter(dct_Cr, threshold = threshold)
    dct_Cb = dct(dct(Cb.T, norm='ortho').T, norm='ortho')
    dct_Cb = high_pass_filter(dct_Cb, threshold = threshold)
    return dct_Y, img, dct_Cr, dct_Cb
image_path = 'test.jpg'
dct_Y, img, dct_Cr, dct_Cb = dct_YCbCr(image_path)
# img, dct_filtered, laplacian = preprocessing(image_path)
# print(dct_filtered.shape)
# print(laplacian.shape)

# Show the images
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(dct_Y, cmap='gray')
plt.title('DCT Filtered Image')
plt.axis('off')
# plt.subplot(1, 2, 2)
# plt.imshow(laplacian, cmap='grey')
# plt.title('Laplacian Filtered Image')
# plt.axis('off')
plt.show()