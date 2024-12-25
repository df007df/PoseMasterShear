# -*- coding: utf-8 -*-
from pose_outline import PoseOutlineGenerator
import os
import torch

def main():
    # 检查设备可用性
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = "cuda"
        torch.cuda.set_device(0)  # 使用第一个 GPU
        torch.cuda.empty_cache()
        print("Using NVIDIA GPU (CUDA)")
    else:
        device = "cpu"
        print("Using CPU")
    
    print(f"Using device: {device}")

    # 使用 SAM 模型
    model_path = "sam_vit_b_01ec64.pth"
    generator = PoseOutlineGenerator(
        sam_checkpoint=model_path,
        model_type="vit_b",
        device=device
    )

    imgNameId = 5
    input_image = os.path.join("test", f"{imgNameId}.jpg")
    output_image = os.path.join("test", f"{imgNameId}_mask.png")
    
    if not os.path.exists(input_image):
        print(f"Error: Input file '{input_image}' does not exist!")
        return
    
    if generator.process_image(input_image, output_image):
        print("Outline generated successfully!")
    else:
        print("Failed to generate outline")

if __name__ == "__main__":
    main() 