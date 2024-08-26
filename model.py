import torch
import torch.nn as nn
from swin_functions_and_classes import SwinTransformer
from preprocess import preprocessing
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F


class SegmetationHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegmetationHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        return self.conv(x)
    
# Define the FPN decoder
class FPNDecoder(nn.Module):
    def __init__(self, feature_channels, out_channels):
        super(FPNDecoder, self).__init__()

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels in feature_channels
        ])

        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in feature_channels
        ])

    def forward(self, features):
        reshaped_features = []
        for i, f in enumerate(features):
            B, N, C = f.shape
            H = W = int(N ** 0.5)  # Assuming the number of patches forms a square grid
            reshaped_f = f.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
            reshaped_features.append(reshaped_f)
            
        lateral_features = [lateral_conv(f) for lateral_conv, f in zip(self.lateral_convs, reshaped_features)]

        # Upsample and add the feature maps, ensuring they have the same size
        for i in range(len(lateral_features) - 1, 0, -1):
            # Get the target size from the previous feature map
            target_size = lateral_features[i - 1].shape[2:]
            # Upsample to the target size
            upsampled = F.interpolate(lateral_features[i], size=target_size, mode='nearest')
            lateral_features[i - 1] += upsampled

        pyramid_features = [smooth_conv(f) for smooth_conv, f in zip(self.smooth_convs, lateral_features)]

        return pyramid_features

class Model(nn.Module):
    def __init__(self, num_classes = 1, threshold = 0.5):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.threshold = threshold
        self.device = 'mps'
        self.Encoder = SwinTransformer(num_classes=num_classes)
        # self.Encoder.to(self.device)
        self.FPNDecoder = FPNDecoder([192, 384, 768, 768], 256)
        # self.FPNDecoder.to(self.device)
        self.segmentation_head = SegmetationHead(256, num_classes)

        
    def forward(self, x):
        features = self.Encoder(x)    
        # print(features[0].shape, features[1].shape, features[2].shape, features[3].shape)
        pyramid_features = self.FPNDecoder(features)
        # Initialize combined feature map with the highest resolution feature map
        combined_feature_map = pyramid_features[0]
        target_size = combined_feature_map.shape[2:]  # Get target size from the first feature map
        
        # Upsample and add the other feature maps to the combined map
        for i in range(1, len(pyramid_features)):
            upsampled = F.interpolate(pyramid_features[i], size=target_size, mode='nearest')
            combined_feature_map += upsampled
        
        segmentation_map = self.segmentation_head(combined_feature_map)
        
        final_segmentation_map = F.interpolate(segmentation_map, size=(224, 224), mode='bilinear', align_corners=False)
        
        # final_segmentation_map = torch.sigmoid(final_segmentation_map)
        # binary_segmentation_map = (final_segmentation_map > self.threshold).float()
        
        return final_segmentation_map
    
    


# # take a random picture
# input_picture = 'test.jpg'
# input_tensor = preprocessing(input_picture)
# #  convert the input tensor to a tensor
# input_tensor = torch.tensor(input_tensor).unsqueeze(0).float()
# print(input_tensor.shape)

# model = Model()
# # model = model.to('mps')

# # Perform a forward pass
# output = model.forward(input_tensor)
# print(output.shape)

# # Visualize the output
# import matplotlib.pyplot as plt
# plt.imshow(output[0, 0].detach().numpy(), cmap='gray')
# plt.axis('off')
# plt.show()

    

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS
):

  # Use ImageFolder to create dataset(s)
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  # Get class names
  class_names = train_data.classes

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names


# Create image size
IMG_SIZE = 224

# # Create transform pipeline manually
# manual_transforms = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
# ])           
# print(f"Manually created transforms: {manual_transforms}")

