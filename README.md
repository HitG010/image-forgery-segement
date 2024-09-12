# Image_Forgery_EfiicientNet

The EfficientNet-based segmentation model is designed to perform binary image segmentation. It utilizes a pre-trained EfficientNet-B0 model as the encoder, which is well-known for its efficient feature extraction capabilities. EfficientNet-B0 is trained on the ImageNet dataset and provides a robust feature extractor that can handle complex image data while maintaining computational efficiency. The encoder extracts features from the input image and passes them on to the decoder.

The decoder in this model is a CNN-based architecture that progressively upsamples the feature map obtained from the encoder to match the original input image dimensions. This is achieved through several transposed convolution layers, which allow the decoder to increase the resolution of the encoded features. The final layer produces a single-channel output, representing the binary segmentation mask for the input image. A sigmoid activation function is applied to the output to ensure the values are scaled between 0 and 1, suitable for binary segmentation tasks.

This segmentation model takes input images of size (3, 224, 224) and outputs a binary mask of size (1, 224, 224), where each pixel value represents whether that region of the image belongs to a particular class or not. The architecture is suitable for tasks where object boundaries need to be identified or specific regions in an image need to be classified.

To use this model, you can initialize it with PyTorch and pass an image through it for inference. It can be integrated into a larger image segmentation pipeline for training, where a loss function like binary cross-entropy can be used. During inference, you can pass images into the model to generate the binary segmentation mask. The use of EfficientNet makes the model both accurate and efficient, which is essential for tasks that require real-time or large-scale image processing.

Dataset used : https://www.kaggle.com/datasets/defactodataset/defactosplicing

