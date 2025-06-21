#!/usr/bin/env python3
"""
RIFE 4.6 Optimization Demo Script

This script demonstrates the performance optimizations applied to RIFE 4.6
and runs a quick benchmark to show the improvements.
"""

import torch
import time
import os
import sys


def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_section(title):
    """Print a formatted section header"""
    print(f"\n--- {title} ---")


def quick_benchmark():
    """Run a quick benchmark with fewer frames for demo purposes"""
    print_header("RIFE 4.6 Performance Optimization Demo")

    # System info
    print_section("System Information")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )
        print(f"CUDA Version: {torch.version.cuda}")

    # Check model weights
    print_section("Model Status")
    if os.path.exists("rife46.pth"):
        print("✓ Model weights (rife46.pth) found")
        weights_size = os.path.getsize("rife46.pth") / (1024**2)
        print(f"  Weights file size: {weights_size:.1f} MB")
    else:
        print("⚠ Model weights (rife46.pth) not found - using random weights")
        print("  Performance comparison may not be meaningful")

    # Run quick demo
    if torch.cuda.is_available():
        print_section("Quick Performance Demo")
        print("Running a quick 50-frame benchmark...")

        try:
            from benchmark_rife import RIFEBenchmark

            # Create smaller benchmark for demo
            demo_benchmark = RIFEBenchmark(width=1280, height=720)

            # Generate fewer frames for quick demo
            frames = demo_benchmark.create_synthetic_frames(num_frames=50)

            print("\nBaseline performance:")
            baseline_metrics = demo_benchmark.benchmark_baseline(frames)

            print("\nOptimized performance:")
            optimized_metrics = demo_benchmark.benchmark_optimized(frames)

            # Calculate improvement
            fps_improvement = (
                (optimized_metrics["fps"] - baseline_metrics["fps"])
                / baseline_metrics["fps"]
            ) * 100
            speedup = optimized_metrics["fps"] / baseline_metrics["fps"]

            print_section("Performance Summary")
            print(f"Baseline FPS:     {baseline_metrics['fps']:.2f}")
            print(f"Optimized FPS:    {optimized_metrics['fps']:.2f}")
            print(f"Improvement:      {fps_improvement:+.1f}%")
            print(f"Speedup Factor:   {speedup:.2f}x")

            optimizations = optimized_metrics.get("optimizations_applied", [])
            if optimizations:
                print(f"\nOptimizations Applied:")
                for opt in optimizations:
                    print(f"  • {opt}")

            print_section("Next Steps")
            print("To run the full 500-frame benchmark:")
            print("  python benchmark_rife.py")
            print("\nThis will generate a detailed report in OPTIMIZATION_REPORT.md")

        except Exception as e:
            print(f"Demo failed: {e}")
            print("\nTry running the setup test first:")
            print("  python test_setup.py")
    else:
        print_section("CUDA Not Available")
        print("This optimization framework requires CUDA for optimal performance.")
        print("You can still test basic functionality on CPU:")
        print("  python test_setup.py")


def main():
    """Main demo function"""
    try:
        quick_benchmark()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        print("\nTroubleshooting:")
        print("1. Run: python test_setup.py")
        print("2. Check dependencies: pip install torch nvidia-ml-py3 numpy psutil")
        print("3. Ensure CUDA is properly installed")

    print_header("Demo Complete")
    print("For more information, see README.md")


if __name__ == "__main__":
    main()
