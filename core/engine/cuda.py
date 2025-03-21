"""
CUDA utilities for the document analysis toolkit
"""

from core.engine.logging import debug_print

def check_cuda_status(config):
    """Check for CUDA availability and configuration"""
    debug_print(config, "Checking CUDA status")
    
    try:
        # Try to import torch
        import torch
        
        # Check if CUDA is available
        cuda_available = torch.cuda.is_available()
        
        print(f"CUDA Available: {cuda_available}")
        
        if cuda_available:
            # Get device count
            device_count = torch.cuda.device_count()
            print(f"CUDA Device Count: {device_count}")
            
            # Get current device
            current_device = torch.cuda.current_device()
            print(f"Current CUDA Device: {current_device}")
            
            # Get device name
            device_name = torch.cuda.get_device_name(current_device)
            print(f"CUDA Device Name: {device_name}")
            
            # Get device properties
            properties = torch.cuda.get_device_properties(current_device)
            print(f"Total Memory: {properties.total_memory / 1024**3:.2f} GB")
            print(f"CUDA Capability: {properties.major}.{properties.minor}")
        
        # Check for MPS on Apple Silicon
        mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
        print(f"MPS Available: {mps_available}")
        
        # Determine best device for the system
        if cuda_available:
            best_device = "cuda"
        elif mps_available:
            best_device = "mps"
        else:
            best_device = "cpu"
        
        print(f"Recommended Device: {best_device}")
        
        # Update config if device is set to 'auto'
        if config.get('hardware.device') == 'auto':
            config.set('hardware.device', best_device)
            print(f"Device automatically set to: {best_device}")
    
    except ImportError:
        print("PyTorch not installed. Cannot check CUDA status.")
    except Exception as e:
        print(f"Error checking CUDA status: {str(e)}")
