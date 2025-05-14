import torch
import torch.nn as nn
import random
import kornia
from kornia.geometry.transform import Resize, Hflip


class VolumeTransformEquivarianceLoss(nn.Module):
    """Transformation Equivariance (TE) Loss for Volume outputs.

    Similar to TfEquivarianceLoss but modified to handle volume outputs with shape (B, T, H, W),
    where T is the temporal dimension.

    Input:
        transform_type (str): type of transformation.
            Implemented types: ['rotation'].
            Default: 'rotation'
        consistency_type (str): difference function in the volume space.
            Implemented types : ['mse', 'l1loss']
            Default: 'mse'
        batch_size (int): expected size of the batch. Default: 128.
        max_angle (int): maximum angle of rotation for rotation transformation.
            Default: 45.
        input_hw (tuple of int): image size (height, width) of input images.
            Default: (224, 224).
        output_hw (tuple of int): size (height, width) of output heatmaps/volumes.
            Default: (14, 14).
    """
    def __init__(self,
                 transform_type='rotation',
                 consistency_type='mse',
                 batch_size=128,
                 max_angle=45,
                 input_hw=(224, 224),
                 output_hw=(14, 14)):
        super(VolumeTransformEquivarianceLoss, self).__init__()
        self.transform_type = transform_type
        self.batch_size = batch_size
        self.max_angle = max_angle
        self.input_hw = input_hw
        self.output_hw = output_hw

        if consistency_type == 'mse':
            self.consistency = nn.MSELoss()
        elif consistency_type == 'l1loss':
            self.consistency = nn.L1Loss()
        else:
            raise ValueError(f'Incorrect consistency_type {consistency_type}')

        # Transformation matrices for input images
        self.tf_matrices_input = None
        # Transformation matrices for output volumes
        self.tf_matrices_output = None
        self.scale = 1.0
        self.hflip = False
        
        # Store the rotation angle and flip state to ensure consistent transformations
        self.angle = None
        self.is_flipped = False

    def set_tf_matrices(self):
        """Set transformation matrices for both input images and output volumes"""
        if self.transform_type == 'rotation':
            # Randomly decide scale for this batch
            if random.random() > 0.5:
                self.scale = 1.0
            else:
                self.scale = 0.5
                
            # Randomly decide if horizontal flip should be applied
            self.is_flipped = random.random() > 0.5
            if self.is_flipped:
                self.hflip = Hflip()
            else:
                self.hflip = False
            
            # Generate random angles (same for both input and output)
            self.angle = torch.tensor(
                [random.randint(-self.max_angle, self.max_angle)
                 for _ in range(self.batch_size)],
                dtype=torch.float32
            )
            
            # Create transformation matrices for input images
            self.tf_matrices_input = self._get_rotation_matrices(self.input_hw)
            
            # Create transformation matrices for output volumes
            self.tf_matrices_output = self._get_rotation_matrices(self.output_hw)

    def _get_rotation_matrices(self, hw):
        """Get transformation matrices for specific height and width
        
        Input:
            hw (tuple): height and width of the image/volume slice
            
        Output:
            tf_matrices (float torch.Tensor): tensor of shape (batch_size, 2, 3)
        """
        # Define the rotation center at the middle of the image/volume
        center = torch.ones(self.batch_size, 2)
        center[..., 0] = hw[1] / 2  # x
        center[..., 1] = hw[0] / 2  # y
        
        # Use the previously generated angle and scale
        scale_tensor = torch.full((self.batch_size, 2), self.scale, dtype=torch.float32)
        
        # Get rotation matrices
        tf_matrices = kornia.geometry.transform.get_rotation_matrix2d(
            center, self.angle, scale_tensor
        )
        
        return tf_matrices

    def transform_image(self, x):
        """Transform input image with input transformation matrices
        
        Input:
            x (float torch.Tensor): input data of shape (batch_size, ch, h, w)

        Output:
            tf_x (float torch.Tensor): transformed data of shape (batch_size, ch, h, w)
        """
        # Check input size
        if x.shape[2:] != self.input_hw:
            x = Resize(self.input_hw)(x)

        # Move matrices to the same device as input
        self.tf_matrices_input = self.tf_matrices_input.to(x.device)
        
        # Apply affine transformation
        dsize = [int(item * self.scale) for item in x.size()[-2:]]
        
        tf_x = kornia.geometry.transform.warp_affine(
            x.float(),
            self.tf_matrices_input,
            dsize=dsize
        )
        
        # Apply horizontal flip if enabled
        if self.hflip:
            tf_x = self.hflip(tf_x)
            
        return tf_x

    def transform_volume(self, volume):
        """Transform volume with output transformation matrices
        
        Input:
            volume (float torch.Tensor): volume data of shape (batch_size, time, height, width)
                                        or (batch_size, channels, time, height, width)

        Output:
            tf_volume (float torch.Tensor): transformed volume with same shape as input
        """
        original_shape = volume.shape
        device = volume.device
        
        # Move matrices to the same device as input
        self.tf_matrices_output = self.tf_matrices_output.to(device)
        
        # Handle 4D input (B, T, H, W)
        if len(original_shape) == 4:
            B, T, H, W = original_shape
            
            # Ensure spatial dimensions match what we expect
            if (H, W) != self.output_hw:
                print(f"Warning: volume spatial dimensions {(H, W)} don't match expected output_hw {self.output_hw}")
            
            # Reshape to (B*T, 1, H, W) to apply spatial transformations to each time slice
            volume_reshaped = volume.view(B*T, 1, H, W)
            
            # Repeat transformation matrices for each time slice
            repeated_matrices = self.tf_matrices_output.repeat_interleave(T, dim=0)
            
            # Apply transformation
            dsize = [int(item * self.scale) for item in [H, W]]
            
            transformed = kornia.geometry.transform.warp_affine(
                volume_reshaped.float(),
                repeated_matrices,
                dsize=dsize
            )
            
            # Apply horizontal flip if enabled
            if self.hflip:
                transformed = self.hflip(transformed)
            
            # Reshape back to original format
            transformed = transformed.view(B, T, dsize[0], dsize[1])
            
        # Handle 5D input (B, C, T, H, W)
        elif len(original_shape) == 5:
            B, C, T, H, W = original_shape
            
            # Ensure spatial dimensions match what we expect
            if (H, W) != self.output_hw:
                print(f"Warning: volume spatial dimensions {(H, W)} don't match expected output_hw {self.output_hw}")
            
            # Reshape to (B*C*T, 1, H, W) to apply spatial transformations to each channel and time slice
            volume_reshaped = volume.view(B*C*T, 1, H, W)
            
            # Repeat transformation matrices for each channel and time slice
            repeated_matrices = self.tf_matrices_output.repeat_interleave(C*T, dim=0)
            
            # Apply transformation
            dsize = [int(item * self.scale) for item in [H, W]]
            
            transformed = kornia.geometry.transform.warp_affine(
                volume_reshaped.float(),
                repeated_matrices,
                dsize=dsize
            )
            
            # Apply horizontal flip if enabled
            if self.hflip:
                transformed = self.hflip(transformed)
            
            # Reshape back to original format
            transformed = transformed.view(B, C, T, dsize[0], dsize[1])
        
        else:
            raise ValueError(f"Unsupported volume shape: {original_shape}. Expected 4D (B,T,H,W) or 5D (B,C,T,H,W)")
            
        return transformed

    def forward(self, tfx_volume, ftx_volume):
        """Compare the transformed volume from the model with 
        the transformed output volume from the original image.
        
        Input:
            tfx_volume: volume from the model with transformed input image (B,T,H,W)
            ftx_volume: transformed volume from the model with original input image (B,T,H,W)
        """
        # Make sure both volumes have the same spatial dimensions
        if ftx_volume.shape[-2:] != tfx_volume.shape[-2:]:
            # Resize spatial dimensions to match
            resized_shape = tfx_volume.shape
            
            if len(ftx_volume.shape) == 4:  # (B,T,H,W)
                B, T, _, _ = ftx_volume.shape
                ftx_volume = ftx_volume.view(B*T, 1, ftx_volume.shape[-2], ftx_volume.shape[-1])
                ftx_volume = kornia.geometry.transform.resize(ftx_volume, tfx_volume.shape[-2:])
                ftx_volume = ftx_volume.view(B, T, tfx_volume.shape[-2], tfx_volume.shape[-1])
                
            elif len(ftx_volume.shape) == 5:  # (B,C,T,H,W)
                B, C, T, _, _ = ftx_volume.shape
                ftx_volume = ftx_volume.view(B*C*T, 1, ftx_volume.shape[-2], ftx_volume.shape[-1])
                ftx_volume = kornia.geometry.transform.resize(ftx_volume, tfx_volume.shape[-2:])
                ftx_volume = ftx_volume.view(B, C, T, tfx_volume.shape[-2], tfx_volume.shape[-1])
        
        # Calculate consistency loss
        loss = self.consistency(tfx_volume, ftx_volume)
        return loss