"""
RIFE 4.25 Implementation
Based on the original RIFE 4.25 architecture with compatibility for optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union
import numpy as np

# Import warp function - will use our optimized version
try:
    from warplayer_v2 import warp
except ImportError:
    print("Warning: warplayer_v2 not found, using basic warp implementation")

    def warp(x, flow):
        """Basic warp implementation fallback"""
        B, C, H, W = x.size()
        # Create grid
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=0).float().to(x.device)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)

        # Add flow
        vgrid = grid + flow

        # Normalize to [-1, 1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        return F.grid_sample(x, vgrid, align_corners=True)


def conv(
    in_planes: int,
    out_planes: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    dilation: int = 1,
) -> nn.Sequential:
    """Basic convolution block with LeakyReLU activation"""
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        ),
        nn.LeakyReLU(0.2, True),
    )


def conv_bn(
    in_planes: int,
    out_planes: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    dilation: int = 1,
) -> nn.Sequential:
    """Convolution block with BatchNorm and LeakyReLU activation"""
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        ),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, True),
    )


class Head(nn.Module):
    """Feature encoding head for RIFE 4.25"""

    def __init__(self):
        super(Head, self).__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

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


class ResConv(nn.Module):
    """Residual convolution block with learnable scaling"""

    def __init__(self, c: int, dilation: int = 1):
        super(ResConv, self).__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    """Intermediate Flow Block for hierarchical flow estimation"""

    def __init__(self, in_planes: int, c: int = 64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
        )
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4 * 13, 4, 2, 1), nn.PixelShuffle(2)
        )

    def forward(
        self, x: torch.Tensor, flow: Optional[torch.Tensor] = None, scale: float = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = F.interpolate(
            x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
        )
        if flow is not None:
            flow = (
                F.interpolate(
                    flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
                )
                * 1.0
                / scale
            )
            x = torch.cat((x, flow), 1)

        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        tmp = F.interpolate(
            tmp, scale_factor=scale, mode="bilinear", align_corners=False
        )

        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        feat = tmp[:, 5:]

        return flow, mask, feat


class IFNet(nn.Module):
    """RIFE 4.25 Interpolation Network"""

    def __init__(
        self,
        scale: float = 1.0,
        ensemble: bool = False,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
        width: int = 1920,
        height: int = 1080,
    ):
        super(IFNet, self).__init__()

        # Network components
        self.block0 = IFBlock(7 + 8, c=192)
        self.block1 = IFBlock(8 + 4 + 8 + 8, c=128)
        self.block2 = IFBlock(8 + 4 + 8 + 8, c=96)
        self.block3 = IFBlock(8 + 4 + 8 + 8, c=64)
        self.block4 = IFBlock(8 + 4 + 8 + 8, c=32)
        self.encode = Head()

        # Configuration
        self.scale = scale
        self.ensemble = ensemble
        self.dtype = dtype
        self.device_name = device
        self.width = width
        self.height = height

        # Performance settings
        self.scale_list = [8, 4, 2, 1, 1]  # Default scale list for RIFE 4.25

    def forward(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        timestep: Union[float, torch.Tensor] = 0.5,
        scale_list: Optional[List[float]] = None,
        training: bool = False,
        fastmode: bool = True,
        ensemble: bool = False,
    ) -> Union[torch.Tensor, Tuple]:
        """
        Forward pass for RIFE 4.25

        Args:
            img0: First input frame [B, C, H, W]
            img1: Second input frame [B, C, H, W]
            timestep: Interpolation time (0.0 to 1.0)
            scale_list: Multi-scale processing scales
            training: Training mode flag
            fastmode: Fast inference mode
            ensemble: Ensemble mode (not supported in 4.25)

        Returns:
            Interpolated frame or (flow_list, mask, merged) tuple
        """
        if scale_list is None:
            scale_list = self.scale_list

        # Handle different input formats
        if not training and img0.shape[1] == 6:
            # Concatenated input format
            channel = img0.shape[1] // 2
            img1 = img0[:, channel:]
            img0 = img0[:, :channel]

        # Prepare timestep
        if not torch.is_tensor(timestep):
            timestep = (img0[:, :1].clone() * 0 + 1) * timestep
        else:
            timestep = timestep.repeat(1, 1, img0.shape[2], img0.shape[3])

        # Encode features
        f0 = self.encode(img0[:, :3])
        f1 = self.encode(img1[:, :3])

        # Initialize tracking variables
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        mask = None

        # Multi-scale processing
        blocks = [self.block0, self.block1, self.block2, self.block3, self.block4]

        for i in range(5):
            if flow is None:
                # Initial flow estimation
                flow, mask, feat = blocks[i](
                    torch.cat((img0[:, :3], img1[:, :3], f0, f1, timestep), 1),
                    None,
                    scale=scale_list[i],
                )
                if ensemble:
                    print("Warning: ensemble is not supported since RIFEv4.21")
            else:
                # Hierarchical flow refinement
                wf0 = warp(f0, flow[:, :2])
                wf1 = warp(f1, flow[:, 2:4])
                fd, m0, feat = blocks[i](
                    torch.cat(
                        (
                            warped_img0[:, :3],
                            warped_img1[:, :3],
                            wf0,
                            wf1,
                            timestep,
                            mask,
                            feat,
                        ),
                        1,
                    ),
                    flow,
                    scale=scale_list[i],
                )

                if ensemble:
                    print("Warning: ensemble is not supported since RIFEv4.21")
                else:
                    mask = m0

                flow = flow + fd

            # Store intermediate results
            mask_list.append(mask)
            flow_list.append(flow)

            # Warp images with current flow
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged.append((warped_img0, warped_img1))

        # Final blending
        mask = torch.sigmoid(mask)
        merged[4] = warped_img0 * mask + warped_img1 * (1 - mask)

        if not fastmode:
            print("ContextNet is removed in RIFE 4.25")

        # Return format depends on usage
        if training:
            return flow_list, mask_list[4], merged
        else:
            return merged[4]

    def __call__(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        timestep: Union[float, torch.Tensor] = 0.5,
    ) -> torch.Tensor:
        """Simplified call interface for inference"""
        return self.forward(img0, img1, timestep, fastmode=True, training=False)
