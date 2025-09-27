#!/usr/bin/env python3
"""
Accelerate configuration for multi-GPU training
Automatically configures accelerate for different hardware setups
"""

import torch
import subprocess
import os
from pathlib import Path

def detect_gpu_setup():
    """Detect available GPUs and suggest configuration"""
    if not torch.cuda.is_available():
        print("No CUDA GPUs detected!")
        return None
        
    gpu_count = torch.cuda.device_count()
    gpu_names = []
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_properties(i).name
        gpu_names.append(gpu_name)
        
    print(f"Detected {gpu_count} GPU(s):")
    for i, name in enumerate(gpu_names):
        memory_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {name} ({memory_gb:.1f} GB)")
        
    return gpu_count, gpu_names

def create_accelerate_config(num_gpus: int, config_name: str = "default"):
    """Create accelerate configuration file"""
    
    config_content = f"""compute_environment: LOCAL_MACHINE
distributed_type: {'MULTI_GPU' if num_gpus > 1 else 'NO'}
downcast_bf16: 'no'
gpu_ids: {','.join(str(i) for i in range(num_gpus)) if num_gpus > 1 else '0'}
machine_rank: 0
main_training_function: main
mixed_precision: {'bf16' if torch.cuda.is_bf16_supported() else 'fp16'}
num_machines: 1
num_processes: {num_gpus}
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
"""

    # Create config directory
    config_dir = Path.home() / ".cache" / "huggingface" / "accelerate"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Write configuration file
    config_file = config_dir / "default_config.yaml"
    with open(config_file, 'w') as f:
        f.write(config_content)
        
    print(f"Accelerate config written to: {config_file}")
    return config_file

def test_accelerate_config():
    """Test the accelerate configuration"""
    try:
        result = subprocess.run(
            ["accelerate", "env"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        print("Accelerate configuration test:")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Accelerate configuration test failed: {e}")
        return False
    except FileNotFoundError:
        print("Accelerate not found. Install with: pip install accelerate")
        return False

def main():
    print("=== Accelerate Multi-GPU Configuration ===")
    
    # Detect hardware
    gpu_setup = detect_gpu_setup()
    if gpu_setup is None:
        return
        
    num_gpus, gpu_names = gpu_setup
    
    # Create configuration
    config_file = create_accelerate_config(num_gpus)
    
    # Test configuration
    print("\nTesting configuration...")
    if test_accelerate_config():
        print("✅ Accelerate configured successfully!")
    else:
        print("❌ Configuration test failed")
        return
        
    # Print usage instructions
    print(f"\n=== Usage Instructions ===")
    if num_gpus == 1:
        print("Single GPU detected. Use:")
        print("  python src/train_multigpu.py --config configs/single_gpu.yaml")
    else:
        print(f"Multi-GPU setup detected ({num_gpus} GPUs). Use:")
        print("  accelerate launch src/train_multigpu.py --config configs/dual_gpu.yaml")
        
    print(f"\nConfiguration saved to: {config_file}")
    print("Run 'accelerate env' to verify settings")

if __name__ == "__main__":
    main()
