# PoseMasterShear

基于 SAM (Segment Anything Model) 和 MediaPipe 的人像分割和轮廓提取工具。

[English](README.md)

## 演示

以下是 PoseMasterShear 的功能展示：

| 输入图片 | 姿态检测 | 人像分割 | 轮廓提取 |
|---------|---------|----------|----------|
| ![原始图片](demo/5.jpg) | ![姿态](demo/5_mask_pose.png) | ![人像](demo/5_mask_person.png) | ![轮廓](demo/5_mask_outline.png) |

- **输入图片**：待处理的原始图片
- **姿态检测**：精确的人体姿态关键点检测
- **人像分割**：将人物精确地从背景中分离
- **轮廓提取**：从人像中生成清晰的轮廓线

## 功能特点

- 自动人像分割
- 精确轮廓提取
- 姿态识别
- 支持多种输出格式
- 自动 OSS 存储
- 多用户支持

## 环境要求

- Python 3.9+
- PyTorch
- CUDA (可选，用于 GPU 加速)
- Apple Silicon (可选，支持 MPS 加速)

## 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/PoseMasterShear.git
cd PoseMasterShear
```

2. 创建虚拟环境：
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 下载 SAM 模型权重：
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## 配置说明

1. 设置 OSS 环境变量：
```bash
export OSS_ACCESS_KEY_ID="你的AccessKey ID"
export OSS_ACCESS_KEY_SECRET="你的AccessKey Secret"
export OSS_ENDPOINT="oss-cn-hangzhou.aliyuncs.com"
export OSS_BUCKET_NAME="你的Bucket名称"
```

2. 修改 `config.py` 中的配置：
```python
IMAGE_CONFIG = {
    'base_dir': '/path/to/your/base/dir',  # 修改为你的基础目录
    ...
}
```

## 使用方法

### 基本使用

```python
from image_processor import ImageProcessor

# 初始化处理器
processor = ImageProcessor(device="mps")  # 可选: "mps", "cuda", "cpu"

# 处理图片
results = processor.process_image(
    user_id="123",              # 用户ID
    input_source="image.jpg",   # 输入图片路径
    move_file=False,            # 是否移动源文件
    custom_name=None            # 自定义输出文件名（可选）
)
```

### 输出文件结构

```
base_dir/
└── user123/
    ├── pose_input/
    │   ├── image1.jpg         # 原始输入图片
    │   └── image1_outline.png # 最终轮廓图片
    └── pose_mask/
        └── image1/            # 处理结果目录
            ├── mask.png       # 遮罩图片
            ├── outline.png    # 轮廓图片
            ├── pose.jpg       # 姿态图片
            └── person.png     # 人像分割图片
```

### OSS 存储结构

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

## 返回结果

处理成功后返回一个字典，包含所有生成图片的 OSS URL：

```python
{
    ImageType.INPUT: "https://...input.jpg",
    ImageType.MASK_OUTLINE: "https://...outline.png",
    ImageType.MASK_MASK: "https://...mask.png",
    ImageType.MASK_POSE: "https://...pose.jpg",
    ImageType.MASK_PERSON: "https://...person.png"
}
```

## 注意事项

1. 确保已正确设置所有环境变量
2. 确保 OSS Bucket 具有正确的访问权限
3. 建议使用绝对路径作为输入路径
4. 对于批量处理，建议使用多线程或异步处理

## 错误处理

- 如果环境变量未设置，会抛出明确的错误信息
- 文件不存在或无法访问时会返回相应的错误信息
- OSS 上传失败会提供详细的错误日志

## 性能优化

1. 设备选择：
   - Apple Silicon 设备使用 MPS
   - NVIDIA 显卡使用 CUDA
   - 无 GPU 时自动使用 CPU

2. 批量处理：
   - 支持并行处理多张图片
   - 使用异步 I/O 进行文件操作
   - 为每个线程实现适当的错误处理

## 许可证

MIT License
