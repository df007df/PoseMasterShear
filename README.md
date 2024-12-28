# PoseMasterShear

A portrait segmentation and outline extraction tool based on SAM (Segment Anything Model) and MediaPipe.

[中文文档](README_CN.md)

## Features

- Automatic portrait segmentation
- Precise outline extraction
- Pose detection
- Multiple output formats
- Automatic OSS storage
- Multi-user support

## Requirements

- Python 3.9+
- PyTorch
- CUDA (optional, for GPU acceleration)
- Apple Silicon (optional, supports MPS acceleration)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/PoseMasterShear.git
cd PoseMasterShear
```

2. Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download SAM model weights:
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## Configuration

1. Set OSS environment variables:
```bash
export OSS_ACCESS_KEY_ID="your-access-key-id"
export OSS_ACCESS_KEY_SECRET="your-access-key-secret"
export OSS_ENDPOINT="oss-cn-hangzhou.aliyuncs.com"
export OSS_BUCKET_NAME="your-bucket-name"
```

2. Modify configuration in `config.py`:
```python
IMAGE_CONFIG = {
    'base_dir': '/path/to/your/base/dir',  # Change to your base directory
    ...
}
```

## Usage

### Basic Usage

```python
from image_processor import ImageProcessor

# Initialize processor
processor = ImageProcessor(device="mps")  # Options: "mps", "cuda", "cpu"

# Process image
results = processor.process_image(
    user_id="123",              # User ID
    input_source="image.jpg",   # Input image path
    move_file=False,            # Whether to move source file
    custom_name=None            # Custom output filename (optional)
)
```

### Output File Structure

```
base_dir/
└── user123/
    ├── pose_input/
    │   ├── image1.jpg         # Original input image
    │   └── image1_outline.png # Final outline image
    └── pose_mask/
        └── image1/            # Processing results directory
            ├── mask.png       # Mask image
            ├── outline.png    # Outline image
            ├── pose.jpg       # Pose image
            └── person.png     # Portrait segmentation image
```

### OSS Storage Structure

```
PoseMasterShear/
└── user123/
    ├── pose_input/
    │   ├── image1.jpg
    │   └── image1_outline.png
    └── pose_mask/
        └── image1/
            ├── mask.png
            ├── outline.png
            ├── pose.jpg
            └── person.png
```

## Return Values

After successful processing, returns a dictionary containing OSS URLs for all generated images:

```python
{
    ImageType.INPUT: "https://...input.jpg",
    ImageType.MASK_OUTLINE: "https://...outline.png",
    ImageType.MASK_MASK: "https://...mask.png",
    ImageType.MASK_POSE: "https://...pose.jpg",
    ImageType.MASK_PERSON: "https://...person.png"
}
```

## Important Notes

1. Ensure all environment variables are correctly set
2. Ensure OSS Bucket has proper access permissions
3. Absolute paths are recommended for input paths
4. For batch processing, consider using multi-threading or async processing

## Error Handling

- Clear error messages for missing environment variables
- Appropriate error messages for file access issues
- Detailed error logs for OSS upload failures

## Performance Optimization

1. Device Selection:
   - Use MPS on Apple Silicon devices
   - Use CUDA on NVIDIA GPUs
   - Fallback to CPU if no GPU is available

2. Batch Processing:
   - Process multiple images in parallel
   - Use async I/O for file operations
   - Implement proper error handling for each thread

## License

MIT License
