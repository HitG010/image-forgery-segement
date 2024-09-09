import cv2
import numpy as np
from scipy.fftpack import dct, idct
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, Input, Reshape
from tensorflow.keras.models import Model
import cv2

def apply_srm_filters(image_path):
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

    # Input layer for the image
    input_image = Input(shape=(None, None, 3)) 

    # SRM convolutional layer
    srm_layer = Conv2D(filters=3, kernel_size=(5, 5), padding='same', use_bias=False, kernel_initializer=tf.constant_initializer(srm_filters))(input_image)

    # Additional Conv2D layer to reduce the SRM output to 1 channel
    output_layer = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation=None)(srm_layer)

    # Reshape the output to (1, 512, 512)
    reshaped_output = Reshape((512, 512,))(output_layer)

    # Build the model
    model = Model(inputs=input_image, outputs=reshaped_output)

    # Load image using OpenCV
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR (OpenCV) to RGB
    img_resized = cv2.resize(img, (512, 512))  # Resize to 512x512 if needed
    img_array = np.expand_dims(img_resized, axis=0)

    # Get the final output (1 * 512 * 512)
    final_output = model.predict(img_array)

    return final_output

# # Example usage
# image_path = 'test.jpg'  # Specify the image path
# output = apply_srm_filters(image_path)

# # Output shape of the result (1 * 512 * 512)
# # print("Output shape:", output.shape)



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
# image_path = 'test.jpg'
# dct_Y, img, dct_Cr, dct_Cb = dct_YCbCr(image_path, threshold=50)
# # img, dct_filtered, laplacian = preprocessing(image_path)
# # print(dct_filtered.shape)
# # print(laplacian.shape)

# # Show the images
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(dct_Y, cmap='gray')
# plt.title('DCT Filtered Image')
# plt.axis('off')
# # plt.subplot(1, 2, 2)
# # plt.imshow(laplacian, cmap='grey')
# # plt.title('Laplacian Filtered Image')
# # plt.axis('off')
# plt.show()