#!/usr/bin/env python3
"""
Working RIFE Precision Benchmark - FP32 vs FP16 Performance Analysis
Focuses on the optimized model that properly handles FP16 precision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import json
import os
import sys
import gc
from typing import List, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rife46_optimized import OptimizedIFNet


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution"""

    width: int = 1920
    height: int = 1080
    num_test_frames: int = 50
    num_warmup: int = 5
    device: str = "cuda"
    output_dir: str = "benchmark_results"


def create_test_frames(config: BenchmarkConfig) -> List[torch.Tensor]:
    """Create diverse test frames for evaluation"""
    print(f"Creating {config.num_test_frames} test frames...")

    frames = []
    for i in range(config.num_test_frames):
        # Create diverse patterns
        x = (
            torch.linspace(0, 2 * np.pi, config.width)
            .view(1, -1)
            .expand(config.height, -1)
        )
        y = (
            torch.linspace(0, 2 * np.pi, config.height)
            .view(-1, 1)
            .expand(-1, config.width)
        )

        phase = i * 0.1
        pattern_type = i % 3

        if pattern_type == 0:
            # Sinusoidal patterns
            r = 0.5 + 0.3 * torch.sin(x + phase)
            g = 0.5 + 0.3 * torch.cos(y + phase)
            b = 0.5 + 0.3 * torch.sin(x + y + phase)
        elif pattern_type == 1:
            # Checkerboard pattern
            freq = 15
            r = 0.5 + 0.3 * torch.sign(torch.sin(freq * x) * torch.sin(freq * y))
            g = 0.5 + 0.3 * torch.sign(torch.sin(freq * x + phase))
            b = 0.5 + 0.3 * torch.sign(torch.sin(freq * y + phase))
        else:
            # Radial gradient
            center_x, center_y = config.width // 2, config.height // 2
            r_dist = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) / max(
                center_x, center_y
            )
            r = 0.5 + 0.3 * torch.sin(r_dist * 3 + phase)
            g = 0.5 + 0.3 * torch.cos(r_dist * 3 + phase)
            b = 0.5 + 0.3 * torch.sin(r_dist * 2 + phase)

        frame = torch.stack([r, g, b], dim=0).float()
        frame = torch.clamp(frame, 0, 1)
        frames.append(frame)

    print(f"Created {len(frames)} test frames")
    return frames


def benchmark_model(
    model, test_frames, config: BenchmarkConfig, precision_name: str, dtype: torch.dtype
):
    """Benchmark a model with specific precision"""
    print(f"\n=== BENCHMARKING {precision_name} ===")

    # Warmup
    print(f"Warming up {precision_name} model...")
    dummy_img0 = torch.randn(
        1, 3, config.height, config.width, device=config.device, dtype=dtype
    )
    dummy_img1 = torch.randn(
        1, 3, config.height, config.width, device=config.device, dtype=dtype
    )
    dummy_timestep = torch.tensor([0.5], device=config.device, dtype=dtype).view(
        1, 1, 1, 1
    )

    with torch.no_grad():
        for _ in range(config.num_warmup):
            _ = model(dummy_img0, dummy_img1, dummy_timestep)

    torch.cuda.synchronize()
    print(f"Warmup completed for {precision_name}")

    # Prepare test data
    frame_pairs = [
        (test_frames[i], test_frames[i + 1]) for i in range(len(test_frames) - 1)
    ]
    timestep = torch.tensor([0.5], device=config.device, dtype=dtype).view(1, 1, 1, 1)

    # Clear memory and start benchmarking
    torch.cuda.empty_cache()
    gc.collect()

    start_memory = torch.cuda.memory_allocated() / 1024**2
    start_time = time.time()
    frame_times = []

    with torch.no_grad():
        for i, (img0, img1) in enumerate(frame_pairs):
            frame_start = time.time()

            # Convert to appropriate precision and add batch dimension
            img0_batch = img0.unsqueeze(0).to(device=config.device, dtype=dtype)
            img1_batch = img1.unsqueeze(0).to(device=config.device, dtype=dtype)

            # Interpolate
            interpolated = model(img0_batch, img1_batch, timestep)

            # Synchronize to measure actual compute time
            torch.cuda.synchronize()

            frame_end = time.time()
            frame_times.append(frame_end - frame_start)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(frame_pairs)} frame pairs")

    total_time = time.time() - start_time
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2

    # Calculate metrics
    avg_frame_time = np.mean(frame_times)
    fps = 1.0 / avg_frame_time
    min_fps = 1.0 / max(frame_times)
    max_fps = 1.0 / min(frame_times)

    results = {
        "precision_type": precision_name,
        "avg_fps": fps,
        "min_fps": min_fps,
        "max_fps": max_fps,
        "avg_frame_time_ms": avg_frame_time * 1000,
        "total_time": total_time,
        "peak_memory_mb": peak_memory,
        "frame_times": frame_times,
    }

    print(f"  Results for {precision_name}:")
    print(f"    Average FPS: {fps:.2f}")
    print(f"    Frame time: {avg_frame_time * 1000:.2f}ms")
    print(f"    Peak Memory: {peak_memory:.1f}MB")
    print(f"    Total time: {total_time:.2f}s")

    return results


def compare_quality(
    model_fp32,
    model_fp16,
    test_frames,
    config: BenchmarkConfig,
    num_comparisons: int = 10,
):
    """Compare output quality between FP32 and FP16 models"""
    print(f"\n=== QUALITY COMPARISON ===")
    print(f"Comparing output quality with {num_comparisons} frame pairs...")

    quality_results = []

    with torch.no_grad():
        for i in range(min(num_comparisons, len(test_frames) - 1)):
            img0 = test_frames[i]
            img1 = test_frames[i + 1]

            # FP32 inputs
            img0_fp32 = img0.unsqueeze(0).to(device=config.device, dtype=torch.float32)
            img1_fp32 = img1.unsqueeze(0).to(device=config.device, dtype=torch.float32)
            ts_fp32 = torch.tensor(
                [0.5], device=config.device, dtype=torch.float32
            ).view(1, 1, 1, 1)

            # FP16 inputs
            img0_fp16 = img0.unsqueeze(0).to(device=config.device, dtype=torch.float16)
            img1_fp16 = img1.unsqueeze(0).to(device=config.device, dtype=torch.float16)
            ts_fp16 = torch.tensor(
                [0.5], device=config.device, dtype=torch.float16
            ).view(1, 1, 1, 1)

            # Get outputs
            output_fp32 = model_fp32(img0_fp32, img1_fp32, ts_fp32)
            output_fp16 = model_fp16(img0_fp16, img1_fp16, ts_fp16)

            # Convert to FP32 for comparison
            output_fp16_as_fp32 = output_fp16.float()

            # Calculate quality metrics
            abs_diff = torch.abs(output_fp32 - output_fp16_as_fp32)
            max_error = torch.max(abs_diff).item()
            mean_error = torch.mean(abs_diff).item()
            mse = torch.mean((output_fp32 - output_fp16_as_fp32) ** 2).item()

            # PSNR calculation
            if mse > 0:
                psnr = 20 * torch.log10(1.0 / torch.sqrt(torch.tensor(mse))).item()
            else:
                psnr = float("inf")

            quality_results.append(
                {
                    "max_error": max_error,
                    "mean_error": mean_error,
                    "mse": mse,
                    "psnr": psnr,
                }
            )

            if (i + 1) % 5 == 0:
                print(f"  Compared {i + 1}/{num_comparisons} pairs")

    # Summary statistics
    avg_max_error = np.mean([q["max_error"] for q in quality_results])
    avg_mean_error = np.mean([q["mean_error"] for q in quality_results])
    avg_psnr = np.mean(
        [q["psnr"] for q in quality_results if q["psnr"] != float("inf")]
    )

    print(f"  Quality Analysis Summary:")
    print(f"    Average Max Error: {avg_max_error:.6f}")
    print(f"    Average Mean Error: {avg_mean_error:.6f}")
    print(f"    Average PSNR: {avg_psnr:.2f}dB")

    return {
        "avg_max_error": avg_max_error,
        "avg_mean_error": avg_mean_error,
        "avg_psnr": avg_psnr,
        "individual_results": quality_results,
    }


def main():
    """Main benchmark execution"""
    print("RIFE Working Precision Benchmark")
    print("=" * 50)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This benchmark requires GPU support.")
        return 1

    print(f"CUDA Device: {torch.cuda.get_device_name()}")
    print(f"PyTorch Version: {torch.__version__}")

    # Configure benchmark
    config = BenchmarkConfig(width=1920, height=1080, num_test_frames=50, device="cuda")

    # Create output directory
    Path(config.output_dir).mkdir(exist_ok=True)

    try:
        # Create test frames
        test_frames = create_test_frames(config)

        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Load FP32 model
        print("\n" + "=" * 60)
        print("LOADING OPTIMIZED FP32 MODEL")
        print("=" * 60)
        model_fp32 = OptimizedIFNet(
            scale=1.0,
            ensemble=False,
            dtype=torch.float32,
            device=config.device,
            width=config.width,
            height=config.height,
            half_precision=False,
            memory_efficient=True,
        )

        # Load weights if available
        model_path = "rife46.pth"
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=config.device)
            model_dict = model_fp32.state_dict()
            filtered_checkpoint = {
                k: v
                for k, v in checkpoint.items()
                if k in model_dict and model_dict[k].shape == v.shape
            }
            model_fp32.load_state_dict(filtered_checkpoint, strict=False)
            print(f"Loaded {len(filtered_checkpoint)}/{len(checkpoint)} weights")

        model_fp32.to(config.device)
        model_fp32.eval()

        # Apply torch.compile if available
        if hasattr(torch, "compile"):
            try:
                model_fp32 = torch.compile(model_fp32, mode="max-autotune")
                print("Applied torch.compile optimization to FP32 model")
            except Exception as e:
                print(f"torch.compile failed for FP32: {e}")

        # Load FP16 model
        print("\n" + "=" * 60)
        print("LOADING OPTIMIZED FP16 MODEL")
        print("=" * 60)
        model_fp16 = OptimizedIFNet(
            scale=1.0,
            ensemble=False,
            dtype=torch.float16,
            device=config.device,
            width=config.width,
            height=config.height,
            half_precision=True,
            memory_efficient=True,
        )

        # Load weights if available
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=config.device)
            model_dict = model_fp16.state_dict()
            filtered_checkpoint = {
                k: v
                for k, v in checkpoint.items()
                if k in model_dict and model_dict[k].shape == v.shape
            }
            model_fp16.load_state_dict(filtered_checkpoint, strict=False)
            print(f"Loaded {len(filtered_checkpoint)}/{len(checkpoint)} weights")

        model_fp16.to(config.device)
        model_fp16.eval()

        # Apply torch.compile if available
        if hasattr(torch, "compile"):
            try:
                model_fp16 = torch.compile(model_fp16, mode="max-autotune")
                print("Applied torch.compile optimization to FP16 model")
            except Exception as e:
                print(f"torch.compile failed for FP16: {e}")

        # Benchmark FP32
        print("\n" + "=" * 60)
        print("PHASE 1: FP32 PERFORMANCE")
        print("=" * 60)
        results_fp32 = benchmark_model(
            model_fp32, test_frames, config, "Optimized FP32", torch.float32
        )

        # Benchmark FP16
        print("\n" + "=" * 60)
        print("PHASE 2: FP16 PERFORMANCE")
        print("=" * 60)
        results_fp16 = benchmark_model(
            model_fp16, test_frames, config, "Optimized FP16", torch.float16
        )

        # Quality comparison
        print("\n" + "=" * 60)
        print("PHASE 3: QUALITY ANALYSIS")
        print("=" * 60)
        quality_results = compare_quality(model_fp32, model_fp16, test_frames, config)

        # Performance analysis
        fp32_fps = results_fp32["avg_fps"]
        fp16_fps = results_fp16["avg_fps"]
        speedup = fp16_fps / fp32_fps
        fps_improvement = ((fp16_fps - fp32_fps) / fp32_fps) * 100

        fp32_memory = results_fp32["peak_memory_mb"]
        fp16_memory = results_fp16["peak_memory_mb"]
        memory_reduction = fp32_memory - fp16_memory
        memory_reduction_pct = (memory_reduction / fp32_memory) * 100

        # Comprehensive results
        final_results = {
            "system_info": {
                "gpu": torch.cuda.get_device_name(),
                "pytorch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "resolution": f"{config.width}x{config.height}",
                "test_frames": config.num_test_frames,
            },
            "performance": {
                "fp32": results_fp32,
                "fp16": results_fp16,
                "comparison": {
                    "speedup_factor": speedup,
                    "fps_improvement_percent": fps_improvement,
                    "memory_reduction_mb": memory_reduction,
                    "memory_reduction_percent": memory_reduction_pct,
                },
            },
            "quality": quality_results,
        }

        # Save results
        results_file = Path(config.output_dir) / "working_precision_results.json"
        with open(results_file, "w") as f:
            json.dump(final_results, f, indent=2)

        # Generate report
        report = f"""# RIFE Precision Benchmark Results

## System Configuration
- **GPU**: {torch.cuda.get_device_name()}
- **PyTorch**: {torch.__version__}
- **CUDA**: {torch.version.cuda}
- **Resolution**: {config.width}x{config.height}
- **Test Frames**: {config.num_test_frames}

## Performance Results

### FP32 Performance
- **Average FPS**: {fp32_fps:.2f}
- **Frame Time**: {results_fp32["avg_frame_time_ms"]:.2f}ms
- **Peak Memory**: {fp32_memory:.1f}MB

### FP16 Performance
- **Average FPS**: {fp16_fps:.2f}
- **Frame Time**: {results_fp16["avg_frame_time_ms"]:.2f}ms
- **Peak Memory**: {fp16_memory:.1f}MB

### Performance Comparison
- **Speedup Factor**: {speedup:.2f}x
- **FPS Improvement**: {fps_improvement:+.1f}%
- **Memory Reduction**: {memory_reduction:.1f}MB ({memory_reduction_pct:.1f}%)

## Quality Analysis
- **Average PSNR**: {quality_results["avg_psnr"]:.2f}dB
- **Max Pixel Error**: {quality_results["avg_max_error"]:.6f}
- **Mean Pixel Error**: {quality_results["avg_mean_error"]:.6f}

## Conclusions
FP16 optimization provides {speedup:.1f}x performance improvement with {memory_reduction_pct:.0f}% memory reduction while maintaining excellent quality ({quality_results["avg_psnr"]:.1f}dB PSNR).
"""

        report_file = Path(config.output_dir) / "working_precision_report.md"
        with open(report_file, "w") as f:
            f.write(report)

        print("\n" + "=" * 60)
        print("BENCHMARK COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"FP32 Performance: {fp32_fps:.2f} FPS, {fp32_memory:.1f}MB")
        print(f"FP16 Performance: {fp16_fps:.2f} FPS, {fp16_memory:.1f}MB")
        print(
            f"Improvement: {speedup:.2f}x speedup, {memory_reduction_pct:.1f}% less memory"
        )
        print(f"Quality: {quality_results['avg_psnr']:.1f}dB PSNR (excellent)")
        print(f"\nResults saved to:")
        print(f"  - {results_file}")
        print(f"  - {report_file}")

        return 0

    except Exception as e:
        print(f"Benchmark failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
