import torch
import logging
from typing import Dict, Any
from llama_index.core import Settings


class CUDAChecker:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def check_cuda_availability(self) -> Dict[str, Any]:
        """Check CUDA availability and version information."""
        cuda_info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": None,
            "device_name": None,
            "device_capability": None,
            "using_cuda": False
        }

        if cuda_info["cuda_available"]:
            try:
                cuda_info["current_device"] = torch.cuda.current_device()
                cuda_info["device_name"] = torch.cuda.get_device_name(cuda_info["current_device"])
                cuda_info["device_capability"] = torch.cuda.get_device_capability(cuda_info["current_device"])
                cuda_info["using_cuda"] = True
            except Exception as e:
                self.logger.warning(f"Error getting CUDA device details: {str(e)}")

        return cuda_info

    def check_embedding_device(self) -> str:
        """Check which device is being used for embeddings."""
        embed_model = Settings.embed_model
        if hasattr(embed_model, 'device'):
            return str(embed_model.device)
        return "Device information not available"

    def print_cuda_status(self, mode: str = None):
        """Print CUDA status information."""
        cuda_info = self.check_cuda_availability()

        print("\n=== CUDA Status ===")
        if cuda_info["cuda_available"]:
            print("✓ CUDA is available")
            print(f"CUDA Version: {cuda_info['cuda_version']}")
            print(f"GPU Devices: {cuda_info['device_count']}")
            if cuda_info["device_name"]:
                print(f"Current Device: {cuda_info['device_name']}")
                print(f"Device Capability: {cuda_info['device_capability']}")
        else:
            print("✗ CUDA is not available - using CPU")

        # Mode-specific information
        if mode:
            print(f"\n=== {mode} Mode CUDA Usage ===")
            if mode == "Embedding":
                embed_device = self.check_embedding_device()
                print(f"Embedding Device: {embed_device}")
            elif mode == "LLM":
                print("LLM Processing: Using Ollama (CPU/GPU depends on Ollama configuration)")
            elif mode == "Topic Modeling":
                device = "cuda" if cuda_info["cuda_available"] else "cpu"
                print(f"Topic Modeling Device: {device}")

        print("=" * 50)