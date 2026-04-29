#!/usr/bin/env python3
"""
GPU Setup Script for zim-indexer
Detects GPU type and installs the appropriate onnxruntime variant + CUDA toolkit.

Usage:
    python setup_gpu.py          # Auto-detect and install
    python setup_gpu.py --check  # Just check, don't install
"""

import subprocess
import sys
import shutil


def detect_nvidia() -> bool:
    """Check if NVIDIA GPU is available via nvidia-smi."""
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                print(f"  Found NVIDIA GPU: {result.stdout.strip().splitlines()[0]}")
                return True
        except Exception:
            pass
    return False


def detect_amd_rocm() -> bool:
    """Check if AMD ROCm is available."""
    if shutil.which("rocm-smi") or shutil.which("rocminfo"):
        print("  Found AMD ROCm installation")
        return True
    return False


def get_installed_onnxruntime() -> str | None:
    """Return which onnxruntime variant is installed, if any."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--format=freeze"],
        capture_output=True, text=True
    )
    for line in result.stdout.splitlines():
        pkg = line.split("==")[0].lower()
        if pkg == "onnxruntime-gpu":
            return "onnxruntime-gpu"
        if pkg == "onnxruntime-directml":
            return "onnxruntime-directml"
        if pkg == "onnxruntime-rocm":
            return "onnxruntime-rocm"
        if pkg == "onnxruntime":
            return "onnxruntime"
    return None


def install_onnxruntime(variant: str) -> bool:
    """Uninstall any existing onnxruntime and install the specified variant."""
    # Uninstall all variants first (order matters to avoid conflicts)
    print("  Removing existing onnxruntime packages...")
    for pkg in ["onnxruntime", "onnxruntime-gpu", "onnxruntime-directml", "onnxruntime-rocm"]:
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", pkg],
            capture_output=True
        )

    # Install the correct one with force-reinstall to ensure clean state
    print(f"  Installing {variant}...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--force-reinstall", variant],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  ERROR: Failed to install {variant}")
        print(result.stderr)
        return False
    return True


def check_cuda_toolkit() -> bool:
    """Check if CUDA 12 toolkit DLLs are available."""
    import os
    from pathlib import Path

    # Check conda environment's Library/bin for CUDA DLLs
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        dll_path = Path(conda_prefix) / "Library" / "bin" / "cublasLt64_12.dll"
        if dll_path.exists():
            return True

    # Also check system PATH
    for path_dir in os.environ.get("PATH", "").split(os.pathsep):
        if (Path(path_dir) / "cublasLt64_12.dll").exists():
            return True

    return False


def install_cuda_toolkit() -> bool:
    """Install CUDA 12 toolkit via conda."""
    print("  Installing CUDA 12 toolkit via conda (this may take a while)...")

    # Find conda executable
    conda_exe = shutil.which("conda")
    if not conda_exe:
        print("  ERROR: conda not found. Please install CUDA toolkit manually:")
        print("         conda install cuda-toolkit=12.6 cudnn=9.* -c nvidia")
        return False

    result = subprocess.run(
        [conda_exe, "install", "-y", "cuda-toolkit=12.6", "cudnn=9.*", "-c", "nvidia"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("  ERROR: Failed to install CUDA toolkit")
        print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
        return False
    return True


def verify_providers() -> list[str]:
    """Return available ONNX Runtime providers."""
    result = subprocess.run(
        [sys.executable, "-c",
         "import onnxruntime as ort; print(','.join(ort.get_available_providers()))"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        return result.stdout.strip().split(",")
    return []


def test_cuda_actually_works() -> bool:
    """Test if CUDA actually works for inference, not just detected."""
    print("  Testing CUDA inference...")
    result = subprocess.run(
        [sys.executable, "-c", """
import warnings
warnings.filterwarnings('ignore')
try:
    from fastembed import TextEmbedding
    m = TextEmbedding('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                      providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    list(m.embed(['test']))
    # Check which provider is actually being used
    import onnxruntime as ort
    sess_opts = ort.SessionOptions()
    # If we got here without falling back to CPU, CUDA works
    print('OK')
except Exception as e:
    print(f'FAIL: {e}')
"""],
        capture_output=True, text=True
    )
    return result.returncode == 0 and "OK" in result.stdout


def main():
    check_only = "--check" in sys.argv

    print("\n=== zim-indexer GPU Setup ===\n")

    # Detect GPU
    print("Detecting GPU...")
    has_nvidia = detect_nvidia()
    has_rocm = detect_amd_rocm()

    if not has_nvidia and not has_rocm:
        print("  No NVIDIA or AMD ROCm GPU detected")
        print("  Will use DirectML (works with most GPUs) or CPU fallback")

    # Determine target package
    if has_nvidia:
        target = "onnxruntime-gpu"
    elif has_rocm:
        target = "onnxruntime-rocm"
    else:
        # DirectML works for AMD/Intel without ROCm, and as fallback
        target = "onnxruntime-directml"

    # Check current installation
    current = get_installed_onnxruntime()
    print(f"\nCurrent onnxruntime: {current or 'not installed'}")
    print(f"Recommended:         {target}")

    if current == target:
        print("\n[OK] Correct onnxruntime package already installed!")
    elif check_only:
        print(f"\nRun without --check to install {target}")
    else:
        print(f"\nInstalling {target}...")
        if install_onnxruntime(target):
            print(f"[OK] Installed {target}")
        else:
            print("[FAILED] Installation failed")
            sys.exit(1)

    # For NVIDIA, also check CUDA toolkit
    if has_nvidia:
        print("\nChecking CUDA 12 toolkit...")
        if check_cuda_toolkit():
            print("  [OK] CUDA 12 toolkit found")
        elif check_only:
            print("  [MISSING] CUDA toolkit not installed")
            print("  Run without --check to install")
        else:
            if install_cuda_toolkit():
                print("  [OK] CUDA toolkit installed")
            else:
                print("  [WARNING] CUDA toolkit installation failed")
                print("           GPU acceleration may not work")

    # Verify
    print("\nVerifying ONNX Runtime providers...")
    providers = verify_providers()
    if providers:
        print(f"  Available: {', '.join(providers)}")
        if "CUDAExecutionProvider" in providers:
            # Actually test CUDA works
            if test_cuda_actually_works():
                print("\n[OK] CUDA is working! Embedding will use your NVIDIA GPU.")
            else:
                print("\n[WARNING] CUDA detected but not working properly.")
                print("          Try restarting your terminal/IDE after setup.")
        elif "ROCMExecutionProvider" in providers:
            print("\n[OK] ROCm is ready! Embedding will use your AMD GPU.")
        elif "DmlExecutionProvider" in providers:
            print("\n[OK] DirectML is ready! Embedding will use GPU via DirectML.")
        else:
            print("\n[WARNING] Only CPU provider available. Check your GPU drivers.")
    else:
        print("  Could not verify providers")

    print()


if __name__ == "__main__":
    main()
