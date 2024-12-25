# PoseMasterShear

一个基于 SAM (Segment Anything Model) 和 HED (Holistically-Nested Edge Detection) 的人物轮廓生成工具。

## 功能特点

- 使用 MediaPipe 进行人体骨骼检测
- 使用 SAM 进行精确的人物分割
- 使用 HED 生成细节丰富的边缘轮廓
- 支持输出多种中间结果
- 智能移除人脸区域，保护隐私

## 处理流程

1. **骨骼检测**
   - 使用 MediaPipe Pose 检测人体骨骼关键点
   - 为 SAM 提供精确的提示点
   - 同时获取人脸关键点用于人脸区域识别

2. **人物分割**
   - 使用 SAM 模型进行分割
   - 基于骨骼关键点进行精确定位
   - 使用人脸关键点创建自然的人脸遮罩
   - 通过渐变边缘平滑处理人脸区域
   - 通过参数控制分割质量：
     - mask_threshold: 0.7 (掩码阈值)
     - iou_threshold: 0.98 (IoU 阈值)
     - stability_score_thresh: 0.98 (稳定性分数阈值)

3. **人物抠图**
   - 基于分割结果生成带透明通道的人物图像
   - 保留原始图像的颜色信息
   - 自动移除人脸区域

4. **边缘检测**
   - 使用 HED 检测器生成细节丰富的边缘
   - 处理分辨率为 1024
   - 保持人脸区域的隐私保护

5. **输出处理**
   - 生成三个输出文件：
     - `*_mask.png`: 分割掩码
     - `*_person.png`: 人物抠图结果（带透明背景）
     - `*_outline.png`: 轮廓图像（带透明通道）

## 依赖项

- segment-anything
- mediapipe
- controlnet_aux
- opencv-python
- numpy
- Pillow
- torch

## 使用方法

``` python
 from pose_outline import PoseOutlineGenerator
    # 初始化生成器
    generator = PoseOutlineGenerator(
    sam_checkpoint="sam_vit_b_01ec64.pth",
    model_type="vit_b"
    )
    # 处理图像
    generator.process_image(
    input_path="input.jpg",
    output_path="output.png"
    )
```

## 输出示例

- 掩码图像 (`*_mask.png`): 黑白二值图，显示分割结果
- 人物图像 (`*_person.png`): 带透明背景的原始人物图像
- 轮廓图像 (`*_outline.png`): 黑色背景的透明轮廓图

## 参数调优

### SAM 参数

- `mask_threshold`: 控制掩码生成的阈值
- `iou_threshold`: 控制区域重叠的阈值
- `stability_score_thresh`: 控制分割稳定性的阈值

### 边缘检测参数

- `detect_resolution`: 控制边缘检测的分辨率

### 人脸处理参数

- 人脸区域扩展：使用 15x15 核心进行膨胀
- 边缘平滑：使用 21x21 高斯模糊

## 注意事项

1. 确保输入图像中有清晰可见的人物
2. 建议使用高质量的输入图像以获得更好的效果
3. 可能需要根据具体图像调整参数
4. 人脸区域会被自动移除以保护隐私
