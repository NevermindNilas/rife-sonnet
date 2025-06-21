"""
RIFE 4.25 TensorRT-Optimized Implementation
Enhanced version incorporating TensorRT optimization techniques and advanced PyTorch optimizations.

Key improvements from the provided implementation:
- Pre-computed warping grids for efficiency
- Optimized padding and grid calculations
- Simplified flow handling
- Enhanced memory layout
- TensorRT-friendly operations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate
import math
from typing import List, Tuple, Optional, Union, Dict
import numpy as np

# Import optimized warp function
try:
    from warplayer_v2 import warp
except ImportError:
    print("Warning: warplayer_v2 not found, using optimized grid_sample")
    warp = None


def conv_optimized(
    in_planes: int,
    out_planes: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    dilation: int = 1,
    tensorrt_friendly: bool = True,
) -> nn.Sequential:
    """TensorRT-optimized convolution block"""
    conv_layer = nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=True,
    )

    # Initialize for stable training and inference
    nn.init.kaiming_normal_(
        conv_layer.weight, mode="fan_out", nonlinearity="leaky_relu"
    )
    nn.init.zeros_(conv_layer.bias)

    # Use inplace=False for TensorRT compatibility if needed
    inplace = not tensorrt_friendly
    return nn.Sequential(conv_layer, nn.LeakyReLU(0.2, inplace=inplace))


class TensorRTOptimizedHead(nn.Module):
    """TensorRT-optimized feature encoding head"""

    def __init__(self, tensorrt_friendly: bool = True):
        super(TensorRTOptimizedHead, self).__init__()
        self.tensorrt_friendly = tensorrt_friendly

        # Use optimized initialization
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, inplace=not tensorrt_friendly)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights optimally"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor, feat: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        x0 = self.cnn0(x)
        x = self.relu(x0)
        x1 = self.cnn1(x)
        x = self.relu(x1)
        x2 = self.cnn2(x)
        x = self.relu(x2)
        x3 = self.cnn3(x)

        if feat:
            return [x0, x1, x2, x3]
        return x3


class TensorRTOptimizedResConv(nn.Module):
    """TensorRT-optimized residual convolution"""

    def __init__(self, c: int, tensorrt_friendly: bool = True):
        super(TensorRTOptimizedResConv, self).__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, padding=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, inplace=not tensorrt_friendly)

        # Optimal initialization
        nn.init.kaiming_normal_(
            self.conv.weight, mode="fan_out", nonlinearity="leaky_relu"
        )
        nn.init.zeros_(self.conv.bias)
        nn.init.constant_(self.beta, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) * self.beta + x)


class TensorRTOptimizedIFBlock(nn.Module):
    """TensorRT-optimized Intermediate Flow Block"""

    def __init__(self, in_planes: int, c: int = 64, tensorrt_friendly: bool = True):
        super(TensorRTOptimizedIFBlock, self).__init__()
        self.in_planes = in_planes
        self.c = c
        self.tensorrt_friendly = tensorrt_friendly

        # Optimized convolution blocks
        self.conv0 = nn.Sequential(
            conv_optimized(
                in_planes, c // 2, 3, 2, 1, tensorrt_friendly=tensorrt_friendly
            ),
            conv_optimized(c // 2, c, 3, 2, 1, tensorrt_friendly=tensorrt_friendly),
        )

        # Residual blocks
        self.convblock = nn.Sequential(
            *[
                TensorRTOptimizedResConv(c, tensorrt_friendly=tensorrt_friendly)
                for _ in range(8)
            ]
        )

        # Optimized upsampling - TensorRT-friendly
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4 * 13, 4, 2, 1), nn.PixelShuffle(2)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights optimally"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor, scale: float = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Optimized interpolation - avoid when scale=1 for TensorRT
        if scale != 1:
            x = interpolate(
                x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
            )

        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)

        # Optimized final interpolation
        if scale != 1:
            tmp = interpolate(
                tmp, scale_factor=scale, mode="bilinear", align_corners=False
            )

        flow = tmp[:, :4]
        mask = tmp[:, 4:5]
        feat_out = tmp[:, 5:]

        # Apply scale to flow after interpolation for better TensorRT optimization
        if scale != 1:
            flow = flow * scale

        return flow, mask, feat_out


class TensorRTOptimizedIFNet(nn.Module):
    """
    TensorRT-Optimized RIFE 4.25 with advanced performance improvements

    Key optimizations:
    - Pre-computed warping grids
    - Optimized padding calculations
    - TensorRT-friendly operations
    - Enhanced memory layout
    - Simplified flow handling
    """

    def __init__(
        self,
        scale: float = 1.0,
        ensemble: bool = False,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
        width: int = 1920,
        height: int = 1080,
        tensorrt_friendly: bool = True,
        half_precision: bool = False,
    ):
        super(TensorRTOptimizedIFNet, self).__init__()

        # Configuration
        self.scale = scale
        self.ensemble = ensemble
        self.dtype = dtype
        self.device_name = device
        self.width = width
        self.height = height
        self.tensorrt_friendly = tensorrt_friendly
        self.half_precision = half_precision

        # Network components with TensorRT optimization
        self.block0 = TensorRTOptimizedIFBlock(
            7 + 8, c=192, tensorrt_friendly=tensorrt_friendly
        )
        self.block1 = TensorRTOptimizedIFBlock(
            8 + 4 + 8 + 8, c=128, tensorrt_friendly=tensorrt_friendly
        )
        self.block2 = TensorRTOptimizedIFBlock(
            8 + 4 + 8 + 8, c=96, tensorrt_friendly=tensorrt_friendly
        )
        self.block3 = TensorRTOptimizedIFBlock(
            8 + 4 + 8 + 8, c=64, tensorrt_friendly=tensorrt_friendly
        )
        self.block4 = TensorRTOptimizedIFBlock(
            8 + 4 + 8 + 8, c=32, tensorrt_friendly=tensorrt_friendly
        )
        self.encode = TensorRTOptimizedHead(tensorrt_friendly=tensorrt_friendly)

        # Optimized scale list
        self.scaleList = [16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale]
        self.blocks = [self.block0, self.block1, self.block2, self.block3, self.block4]

        # Pre-compute padding and grid parameters for efficiency
        self._initialize_grid_parameters()

        # Enable optimizations based on configuration
        if not tensorrt_friendly:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def _initialize_grid_parameters(self):
        """Pre-compute grid parameters for efficient warping"""
        # Optimized padding calculation
        tmp = max(64, int(64 / self.scale))
        self.pw = math.ceil(self.width / tmp) * tmp
        self.ph = math.ceil(self.height / tmp) * tmp
        self.padding = (0, self.pw - self.width, 0, self.ph - self.height)

        # Pre-compute warping grid parameters
        hMul = 2 / (self.pw - 1) if self.pw > 1 else 2.0
        vMul = 2 / (self.ph - 1) if self.ph > 1 else 2.0

        # Register as buffers for TensorRT compatibility
        self.register_buffer(
            "tenFlow", torch.tensor([hMul, vMul], dtype=self.dtype).reshape(1, 2, 1, 1)
        )

        # Pre-compute base warping grid
        self.register_buffer(
            "backWarp",
            torch.cat(
                [
                    (torch.arange(self.pw, dtype=self.dtype) * hMul - 1)
                    .reshape(1, 1, 1, -1)
                    .expand(-1, -1, self.ph, -1),
                    (torch.arange(self.ph, dtype=self.dtype) * vMul - 1)
                    .reshape(1, 1, -1, 1)
                    .expand(-1, -1, -1, self.pw),
                ],
                dim=1,
            ),
        )

    def _warp_images_optimized(
        self, imgs: torch.Tensor, flows: torch.Tensor, fs: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Optimized image warping using pre-computed grids"""
        # Reshape for efficient processing
        imgs2 = imgs.view(2, 3, self.ph, self.pw)

        if fs is not None:
            fs2 = fs.view(2, 4, self.ph, self.pw)
            input_tensor = torch.cat((imgs2, fs2), 1)
        else:
            input_tensor = imgs2

        # Compute warping grid efficiently
        flow_grid = self.backWarp + flows.reshape(2, 2, self.ph, self.pw) * self.tenFlow

        # Optimized grid sampling
        warped = F.grid_sample(
            input_tensor,
            flow_grid.permute(0, 2, 3, 1),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        if fs is not None:
            wimg, wf = torch.split(warped, [3, 4], dim=1)
            wimg = wimg.reshape(1, 6, self.ph, self.pw)
            wf = wf.reshape(1, 8, self.ph, self.pw)
            return wimg, wf
        else:
            return warped

    def forward(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        timeStep: Union[float, torch.Tensor],
        f0: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        TensorRT-optimized forward pass

        Args:
            img0: First input frame [B, 3, H, W]
            img1: Second input frame [B, 3, H, W]
            timeStep: Interpolation time (0.0 to 1.0)
            f0: Pre-computed features for img0 (optional, for efficiency)

        Returns:
            Interpolated frame or (interpolated_frame, f1) tuple
        """
        # Handle padding efficiently
        if self.padding != (0, 0, 0, 0):
            img0 = F.pad(img0, self.padding)
            img1 = F.pad(img1, self.padding)

        # Prepare inputs
        imgs = torch.cat([img0, img1], dim=1)

        # Compute features efficiently
        if f0 is None:
            f0 = self.encode(img0[:, :3])
        f1 = self.encode(img1[:, :3])
        fs = torch.cat([f0, f1], dim=1)

        # Prepare timestep
        if not torch.is_tensor(timeStep):
            timeStep = torch.full(
                (1, 1, self.ph, self.pw), timeStep, dtype=self.dtype, device=img0.device
            )
        elif timeStep.dim() == 0:
            timeStep = timeStep.view(1, 1, 1, 1).expand(1, 1, self.ph, self.pw)

        # Multi-scale processing
        flows = None
        wimg = None
        wf = None

        for block, scale in zip(self.blocks, self.scaleList):
            if flows is None:
                # Initial flow estimation
                temp = torch.cat((imgs, fs, timeStep), 1)
                flows, mask, feat = block(temp, scale=scale)
            else:
                # Hierarchical refinement with optimized warping
                temp = torch.cat(
                    [
                        wimg,
                        wf,
                        timeStep,
                        mask,
                        feat,
                        (flows * (1 / scale) if scale != 1 else flows),
                    ],
                    1,
                )
                fds, mask, feat = block(temp, scale=scale)
                flows = flows + fds

            # Efficient warping for non-final scales
            if scale != 1:
                wimg, wf = self._warp_images_optimized(imgs, flows, fs)

        # Final warping and blending
        warpedImgs = self._warp_images_optimized(imgs, flows)
        mask = torch.sigmoid(mask)
        warpedImg0, warpedImg1 = torch.split(warpedImgs, [1, 1])

        # Final result
        result = warpedImg0 * mask + warpedImg1 * (1 - mask)

        # Remove padding
        if self.padding != (0, 0, 0, 0):
            result = result[:, :, : self.height, : self.width]

        return result, f1

    def __call__(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        timestep: Union[float, torch.Tensor] = 0.5,
    ) -> torch.Tensor:
        """Simplified call interface"""
        result, _ = self.forward(img0, img1, timestep)
        return result

    def get_optimization_info(self) -> Dict[str, any]:
        """Get information about applied optimizations"""
        return {
            "dtype": self.dtype,
            "half_precision": self.half_precision,
            "tensorrt_friendly": self.tensorrt_friendly,
            "device": self.device_name,
            "resolution": f"{self.width}x{self.height}",
            "padded_resolution": f"{self.pw}x{self.ph}",
            "scale_list": self.scaleList,
            "optimizations": [
                "pre_computed_grids",
                "optimized_padding",
                "efficient_warping",
                "tensorrt_compatible",
                "feature_caching_support",
                "optimized_interpolation",
            ],
        }


# Enhanced version that combines PyTorch optimizations with TensorRT techniques
class HybridOptimizedIFNet(TensorRTOptimizedIFNet):
    """
    Hybrid optimization combining TensorRT techniques with PyTorch-specific optimizations
    """

    def __init__(self, *args, **kwargs):
        # Extract PyTorch-specific options
        self.memory_efficient = kwargs.pop("memory_efficient", True)
        self.enable_caching = kwargs.pop("enable_caching", False)

        # Force TensorRT-friendly to False for PyTorch optimizations
        kwargs["tensorrt_friendly"] = False

        super().__init__(*args, **kwargs)

        # PyTorch-specific optimizations
        self._feature_cache = {}
        self._enable_caching = self.enable_caching

        # Enable advanced PyTorch optimizations
        if self.memory_efficient:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def enable_caching(self, enable: bool = True):
        """Enable/disable feature caching"""
        self._enable_caching = enable
        if not enable:
            self._feature_cache.clear()

    def clear_cache(self):
        """Clear cached features"""
        self._feature_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def forward(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        timeStep: Union[float, torch.Tensor],
        f0: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Enhanced forward with PyTorch optimizations"""

        # Feature caching optimization
        if self._enable_caching and f0 is None:
            cache_key = hash(img0.data_ptr())
            if cache_key in self._feature_cache:
                f0 = self._feature_cache[cache_key]
            else:
                f0 = self.encode(img0[:, :3])
                self._feature_cache[cache_key] = f0.detach()

        # Use channels_last memory format for better performance
        if self.memory_efficient:
            img0 = img0.to(memory_format=torch.channels_last)
            img1 = img1.to(memory_format=torch.channels_last)

        # Call parent forward with optimizations
        result, f1 = super().forward(img0, img1, timeStep, f0)

        # Cache f1 for potential future use
        if self._enable_caching:
            cache_key = hash(img1.data_ptr())
            self._feature_cache[cache_key] = f1.detach()

        # Ensure output is in contiguous format
        if hasattr(torch, "contiguous_format"):
            result = result.to(memory_format=torch.contiguous_format)

        return result, f1

    def get_optimization_info(self) -> Dict[str, any]:
        """Get comprehensive optimization information"""
        base_info = super().get_optimization_info()
        base_info.update(
            {
                "memory_efficient": self.memory_efficient,
                "caching_enabled": self._enable_caching,
                "cached_features": len(self._feature_cache),
                "pytorch_optimizations": [
                    "channels_last_memory",
                    "feature_caching",
                    "cudnn_benchmark",
                    "tf32_acceleration",
                    "memory_efficient_operations",
                ],
            }
        )
        return base_info
