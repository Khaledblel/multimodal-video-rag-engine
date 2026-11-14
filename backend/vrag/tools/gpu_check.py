"""Report whether this container can use the GPU: python -m vrag.tools.gpu_check (exits 1 if it cannot)."""

import ctranslate2
import torch


def main() -> int:
    print(f"torch {torch.__version__}, built for CUDA {torch.version.cuda}")
    visible = torch.cuda.is_available()
    if visible:
        gpu = torch.cuda.get_device_properties(0)
        print(f"PASS  GPU visible to PyTorch: {gpu.name}, {gpu.total_memory / 2**30:.1f} GiB")
    else:
        print("FAIL  no GPU visible to PyTorch in this container")
    print(
        f"ctranslate2 {ctranslate2.__version__}: "
        f"{ctranslate2.get_cuda_device_count()} CUDA device(s) for faster-whisper"
    )
    return 0 if visible else 1


if __name__ == "__main__":
    raise SystemExit(main())
