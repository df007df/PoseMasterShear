# -*- coding: utf-8 -*-
from image_processor import ImageProcessor

def main():
    # 初始化图片处理器（会自动选择最佳设备）
    processor = ImageProcessor(device="mps")  # 可以是 "mps", "cuda" 或 "cpu"

    # 示例1：使用原始文件名
    user_id = "-1"
    input_file = "/Users/df007df/work/ios/PoseMasterShear/test/_demo/1.jpg"
    results = processor.process_image(
        user_id=user_id,
        input_source=input_file
    )
    # 输出结果将使用原始文件名 "1" 作为基础名称


if __name__ == "__main__":
    main() 