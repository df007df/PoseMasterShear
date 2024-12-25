# 🎭 PoseMasterShear - AI Portrait Outline Generator

[中文文档](README_CN.md)

> An AI-powered portrait outline generation tool that combines SAM segmentation, depth estimation, and pose detection to achieve high-quality portrait extraction and outline generation while protecting privacy.

## 🌟 Features

- 🤖 Smart Human Pose Detection (MediaPipe)
- 🎯 Precise Portrait Segmentation (SAM)
- 🔍 Depth-Assisted Edge Enhancement
- ✨ Rich Detail Edge Generation (HED)
- 🎨 Multiple Output Format Support
- 🔒 Smart Face Privacy Protection

## Example Outputs

| Input | Pose | Mask | Portrait | Outline |
|:-----:|:----:|:----:|:--------:|:-------:|
| <img src="demo/5.jpg" width="150"> | <img src="demo/5_mask_pose.png" width="150"> | <img src="demo/5_mask_mask.png" width="150"> | <img src="demo/5_mask_person.png" width="150"> | <img src="demo/5_mask_outline.png" width="150"> |

## Quick Start

### Installation
```bash
# For Apple Silicon (M1/M2/M3/M4)
pip install torch torchvision torchaudio

# Other dependencies
pip install segment-anything mediapipe controlnet_aux opencv-python pillow
```

### Usage
```python
generator = PoseOutlineGenerator(
    sam_checkpoint="sam_vit_b_01ec64.pth",
    model_type="vit_b",
    device="mps"  # or "cuda"/"cpu"
)

generator.process_image("input.jpg", "output.png")
```

## Processing Pipeline

1. **Pose Detection**
   - Uses MediaPipe Pose for skeleton keypoint detection
   - Provides precise prompt points for SAM
   - Extracts facial keypoints for privacy protection

2. **Depth Estimation**
   - Generates scene depth map using MiDaS model
   - Estimates edge depth based on pose keypoints
   - Applies depth filtering only to edge regions
   - Preserves main subject segmentation results

3. **Portrait Segmentation**
   - Performs segmentation using SAM model
   - Precise positioning based on pose keypoints
   - Optimizes edges using depth information
   - Creates natural face mask using facial keypoints
   - Smooths face region with gradient edges
   - Quality control parameters:
     - mask_threshold: 0.95
     - iou_threshold: 0.99
     - stability_score_thresh: 0.99

## Output Files

- Mask (`*_mask.png`): Binary mask showing segmentation
- Portrait (`*_person.png`): Original portrait with transparency
- Outline (`*_outline.png`): Outline with transparent background
- Pose (`*_pose.png`): Visualization of pose keypoints

## Parameter Tuning

### SAM Parameters
- `mask_threshold`: Controls mask generation (0-1)
- `iou_threshold`: Controls region overlap (0-1)
- `stability_score_thresh`: Controls segmentation stability (0-1)

### Edge Processing
- Edge region definition: `kernel_size = 31`
- Depth filtering applied only to edges
- Main subject region preserved

### Morphological Processing
- Uses 5x5 kernel for closing operation
- Adjustable kernel size for strength control

## Device Support

### GPU Support
- Automatically detects available GPU
- Uses CUDA acceleration if available
- Manual device selection supported

### Apple Silicon Optimization
- Automatic MPS backend utilization
- Metal acceleration via `device="mps"`

## Notes

1. Ensure clear visibility of subject in input image
2. High-quality input images recommended
3. Parameters may need adjustment per image
4. Facial regions automatically removed for privacy

## Requirements

- segment-anything
- mediapipe
- controlnet_aux
- transformers
- torch >= 2.0
- opencv-python
- numpy
- Pillow
