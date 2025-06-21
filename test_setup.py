"""
Test script to verify RIFE optimization setup and dependencies
"""

import sys
import importlib


def test_imports():
    """Test all required imports"""
    print("Testing imports...")

    required_modules = ["torch", "numpy", "time", "gc", "os", "warnings", "psutil"]

    optional_modules = ["nvidia_ml_py3"]

    # Test required modules
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module} - {e}")
            return False

    # Test optional modules
    for module in optional_modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module} (optional)")
        except ImportError:
            print(f"⚠ {module} (optional) - not available")

    return True


def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA...")

    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name()
            device_count = torch.cuda.device_count()
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

            print(f"✓ CUDA is available")
            print(f"  Device: {device_name}")
            print(f"  Device count: {device_count}")
            print(f"  Memory: {memory_gb:.1f} GB")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  PyTorch Version: {torch.__version__}")
            return True
        else:
            print("✗ CUDA is not available")
            return False
    except Exception as e:
        print(f"✗ Error testing CUDA: {e}")
        return False


def test_model_files():
    """Test if model files exist"""
    print("\nTesting model files...")

    import os

    files_to_check = [
        "rife46.py",
        "rife46_optimized.py",
        "warplayer_v2.py",
        "benchmark_rife.py",
    ]

    all_exist = True
    for file in files_to_check:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - not found")
            all_exist = False

    # Check for weights file
    if os.path.exists("rife46.pth"):
        print("✓ rife46.pth (model weights)")
    else:
        print("⚠ rife46.pth (model weights) - not found, will use random weights")

    return all_exist


def test_basic_functionality():
    """Test basic model loading and inference"""
    print("\nTesting basic functionality...")

    try:
        import torch
        from rife46 import IFNet

        # Create a simple model
        model = IFNet(width=256, height=256, device="cpu")

        # Test with dummy data
        dummy_img0 = torch.randn(1, 3, 256, 256)
        dummy_img1 = torch.randn(1, 3, 256, 256)
        dummy_timestep = torch.tensor([0.5]).view(1, 1, 1, 1).expand(1, 1, 256, 256)

        with torch.no_grad():
            output = model(dummy_img0, dummy_img1, dummy_timestep)

        print(f"✓ Basic model inference works")
        print(f"  Input shape: {dummy_img0.shape}")
        print(f"  Output shape: {output.shape}")

        return True

    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False


def test_optimized_model():
    """Test optimized model loading"""
    print("\nTesting optimized model...")

    try:
        import torch
        from rife46_optimized import OptimizedIFNet

        # Create optimized model
        model = OptimizedIFNet(
            width=256, height=256, device="cpu", memory_efficient=True
        )

        # Test with dummy data
        dummy_img0 = torch.randn(1, 3, 256, 256)
        dummy_img1 = torch.randn(1, 3, 256, 256)
        dummy_timestep = torch.tensor([0.5]).view(1, 1, 1, 1).expand(1, 1, 256, 256)

        with torch.no_grad():
            output = model(dummy_img0, dummy_img1, dummy_timestep)

        print(f"✓ Optimized model inference works")
        print(f"  Input shape: {dummy_img0.shape}")
        print(f"  Output shape: {output.shape}")

        return True

    except Exception as e:
        print(f"✗ Optimized model test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("RIFE Optimization Setup Test")
    print("=" * 40)

    tests = [
        ("Import Dependencies", test_imports),
        ("CUDA Support", test_cuda),
        ("Model Files", test_model_files),
        ("Basic Functionality", test_basic_functionality),
        ("Optimized Model", test_optimized_model),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))
        result = test_func()
        results.append((test_name, result))

    # Summary
    print("\n" + "=" * 40)
    print("TEST SUMMARY")
    print("=" * 40)

    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nPassed: {passed}/{len(tests)} tests")

    if passed == len(tests):
        print("\n✓ All tests passed! Ready to run benchmark.")
    else:
        print(
            f"\n⚠ {len(tests) - passed} test(s) failed. Please resolve issues before running benchmark."
        )

    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
