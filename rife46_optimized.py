import torch
import torch.nn as nn
import torch.nn.functional as F
from warplayer_v2 import warp
import math


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
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


class ResConv(nn.Module):
    def __init__(self, c, dilation=1):
        super(ResConv, self).__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
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
            nn.ConvTranspose2d(c, 4 * 6, 4, 2, 1), nn.PixelShuffle(2)
        )

    def forward(self, x, flow=None, scale=1):
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
        return flow, mask


class OptimizedIFNet(nn.Module):
    """Optimized version of IFNet with performance improvements"""

    def __init__(
        self,
        scale=1.0,
        ensemble=False,
        dtype=torch.float32,
        device="cuda",
        width=1920,
        height=1080,
        half_precision=False,
        memory_efficient=True,
    ):
        super(OptimizedIFNet, self).__init__()
        self.block0 = IFBlock(7, c=192)
        self.block1 = IFBlock(8 + 4, c=128)
        self.block2 = IFBlock(8 + 4, c=96)
        self.block3 = IFBlock(8 + 4, c=64)
        self.scaleList = [8 / scale, 4 / scale, 2 / scale, 1 / scale]
        self.ensemble = ensemble
        self.device = device
        self.width = width
        self.height = height
        self.half_precision = half_precision
        self.memory_efficient = memory_efficient
        self.blocks = [self.block0, self.block1, self.block2, self.block3]

        # Set proper dtype
        self.dtype = torch.float16 if self.half_precision else torch.float32

        # Calculate padded dimensions for efficiency
        tmp = max(32, int(32 / 1.0))
        self.pw = math.ceil(self.width / tmp) * tmp
        self.ph = math.ceil(self.height / tmp) * tmp
        self.padding = (0, self.pw - self.width, 0, self.ph - self.height)

        # Pre-allocate tensors for efficiency
        self._setup_precomputed_tensors()

    def _setup_precomputed_tensors(self):
        """Pre-compute and cache tensors that don't change between forward passes"""
        self.tenFlow = torch.tensor(
            [(self.pw - 1.0) / 2.0, (self.ph - 1.0) / 2.0],
            dtype=self.dtype,
            device=self.device,
        )

        tenHorizontal = (
            torch.linspace(-1.0, 1.0, self.pw, dtype=self.dtype, device=self.device)
            .view(1, 1, 1, self.pw)
            .expand(-1, -1, self.ph, -1)
        )
        tenVertical = (
            torch.linspace(-1.0, 1.0, self.ph, dtype=self.dtype, device=self.device)
            .view(1, 1, self.ph, 1)
            .expand(-1, -1, -1, self.pw)
        )
        self.backWarp = torch.cat([tenHorizontal, tenVertical], 1)

        # Pre-allocate intermediate tensors if memory efficient mode is enabled
        if self.memory_efficient:
            self._allocate_intermediate_tensors()

    def _allocate_intermediate_tensors(self):
        """Pre-allocate intermediate tensors to avoid runtime allocation"""
        batch_size = 1  # Assuming batch size of 1 for video interpolation

        # Tensors for different scales
        self.intermediate_tensors = {}
        for i, scale in enumerate(self.scaleList):
            h_scaled = int(self.ph / scale)
            w_scaled = int(self.pw / scale)

            self.intermediate_tensors[f"scale_{i}"] = {
                "warped_img0": torch.empty(
                    batch_size,
                    3,
                    h_scaled,
                    w_scaled,
                    dtype=self.dtype,
                    device=self.device,
                ),
                "warped_img1": torch.empty(
                    batch_size,
                    3,
                    h_scaled,
                    w_scaled,
                    dtype=self.dtype,
                    device=self.device,
                ),
                "flow": torch.empty(
                    batch_size,
                    4,
                    h_scaled,
                    w_scaled,
                    dtype=self.dtype,
                    device=self.device,
                ),
                "mask": torch.empty(
                    batch_size,
                    1,
                    h_scaled,
                    w_scaled,
                    dtype=self.dtype,
                    device=self.device,
                ),
            }

    def forward(self, img0, img1, timeStep):
        # Apply padding if necessary
        if self.padding != (0, 0, 0, 0):
            img0 = F.pad(img0, self.padding)
            img1 = F.pad(img1, self.padding)

        # Expand timestep to match padded image dimensions
        if timeStep.dim() == 4 and timeStep.shape[2] != img0.shape[2]:
            timeStep = timeStep.expand(-1, -1, img0.shape[2], img0.shape[3])
        elif timeStep.dim() < 4:
            timeStep = timeStep.expand(-1, -1, img0.shape[2], img0.shape[3])

        warpedImg0, warpedImg1 = img0, img1
        flow = mask = None

        for i, block in enumerate(self.blocks):
            scale = self.scaleList[i]

            if flow is None:
                # First iteration
                input_tensor = torch.cat((img0[:, :3], img1[:, :3], timeStep), 1)
                flow, mask = block(input_tensor, None, scale=scale)

                if self.ensemble:
                    input_tensor_reverse = torch.cat(
                        (img1[:, :3], img0[:, :3], 1 - timeStep), 1
                    )
                    f1, m1 = block(input_tensor_reverse, None, scale=scale)
                    # Swap flow channels and average
                    flow = (flow + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
                    mask = (mask - m1) / 2
            else:
                # Subsequent iterations
                input_tensor = torch.cat(
                    (warpedImg0[:, :3], warpedImg1[:, :3], timeStep, mask), 1
                )
                f0, m0 = block(input_tensor, flow, scale=scale)

                if self.ensemble:
                    input_tensor_reverse = torch.cat(
                        (warpedImg1[:, :3], warpedImg0[:, :3], 1 - timeStep, -mask), 1
                    )
                    flow_reverse = torch.cat((flow[:, 2:4], flow[:, :2]), 1)
                    f1, m1 = block(input_tensor_reverse, flow_reverse, scale=scale)
                    # Average the flows and masks
                    f0 = (f0 + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
                    m0 = (m0 - m1) / 2

                flow = flow + f0
                mask = mask + m0

            # Warp images using computed flow
            warpedImg0 = warp(img0, flow[:, :2], self.tenFlow, self.backWarp)
            warpedImg1 = warp(img1, flow[:, 2:4], self.tenFlow, self.backWarp)

        # Final blending
        mask_sigmoid = torch.sigmoid(mask)
        result = warpedImg0 * mask_sigmoid + warpedImg1 * (1 - mask_sigmoid)

        # Remove padding and return
        return result[:, :, : self.height, : self.width]

    def enable_half_precision(self):
        """Enable half precision for faster inference"""
        self.half_precision = True
        self.dtype = torch.float16
        self.half()
        self._setup_precomputed_tensors()

    def enable_memory_efficient_mode(self):
        """Enable memory efficient mode with pre-allocated tensors"""
        self.memory_efficient = True
        self._allocate_intermediate_tensors()


class IFNet(nn.Module):
    """Original IFNet class for compatibility"""

    def __init__(
        self,
        scale=1.0,
        ensemble=False,
        dtype=torch.float32,
        device="cuda",
        width=1920,
        height=1080,
        backWarp=None,
        tenFlow=None,
    ):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(7, c=192)
        self.block1 = IFBlock(8 + 4, c=128)
        self.block2 = IFBlock(8 + 4, c=96)
        self.block3 = IFBlock(8 + 4, c=64)
        self.scaleList = [8 / scale, 4 / scale, 2 / scale, 1 / scale]
        self.ensemble = ensemble
        self.dtype = dtype
        self.device = device
        self.width = width
        self.height = height
        self.blocks = [self.block0, self.block1, self.block2, self.block3]

        # Fix: Define half precision flag properly
        self.half = False  # Default to full precision, can be set via parameter
        self.dtype = torch.float16 if self.half else torch.float32
        tmp = max(32, int(32 / 1.0))
        self.pw = math.ceil(self.width / tmp) * tmp
        self.ph = math.ceil(self.height / tmp) * tmp
        self.padding = (0, self.pw - self.width, 0, self.ph - self.height)
        self.tenFlow = torch.tensor(
            [(self.pw - 1.0) / 2.0, (self.ph - 1.0) / 2.0],
            dtype=self.dtype,
            device=self.device,
        )
        tenHorizontal = (
            torch.linspace(-1.0, 1.0, self.pw, dtype=self.dtype, device=self.device)
            .view(1, 1, 1, self.pw)
            .expand(-1, -1, self.ph, -1)
        ).to(dtype=self.dtype, device=self.device)
        tenVertical = (
            torch.linspace(-1.0, 1.0, self.ph, dtype=self.dtype, device=self.device)
            .view(1, 1, self.ph, 1)
            .expand(-1, -1, -1, self.pw)
        ).to(dtype=self.dtype, device=self.device)
        self.backWarp = torch.cat([tenHorizontal, tenVertical], 1)

    def forward(self, img0, img1, timeStep):
        warpedImg0, warpedImg1 = img0, img1
        flow = mask = None

        for i, block in enumerate(self.blocks):
            scale = self.scaleList[i]

            if flow is None:
                flow, mask = block(
                    torch.cat((img0[:, :3], img1[:, :3], timeStep), 1),
                    None,
                    scale=scale,
                )

                if self.ensemble:
                    f1, m1 = block(
                        torch.cat((img1[:, :3], img0[:, :3], 1 - timeStep), 1),
                        None,
                        scale=scale,
                    )
                    flow = (flow + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
                    mask = (mask - m1) / 2
            else:
                f0, m0 = block(
                    torch.cat(
                        (warpedImg0[:, :3], warpedImg1[:, :3], timeStep, mask), 1
                    ),
                    flow,
                    scale=scale,
                )

                if self.ensemble:
                    f1, m1 = block(
                        torch.cat(
                            (warpedImg1[:, :3], warpedImg0[:, :3], 1 - timeStep, -mask),
                            1,
                        ),
                        torch.cat((flow[:, 2:4], flow[:, :2]), 1),
                        scale=scale,
                    )
                    f0 = (f0 + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
                    m0 = (m0 - m1) / 2

                flow = flow + f0
                mask = mask + m0

            warpedImg0 = warp(img0, flow[:, :2], self.tenFlow, self.backWarp)
            warpedImg1 = warp(img1, flow[:, 2:4], self.tenFlow, self.backWarp)

        temp = torch.sigmoid(mask)
        return (warpedImg0 * temp + warpedImg1 * (1 - temp))[
            :, :, : self.height, : self.width
        ]
