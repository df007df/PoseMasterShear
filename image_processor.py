# -*- coding: utf-8 -*-
from pose_outline import PoseOutlineGenerator
from oss_uploader import OSSUploader, ImageType
import os
import torch
from config import OSS_CONFIG, IMAGE_CONFIG, MODEL_CONFIG
from typing import Dict, Optional, Union, BinaryIO, List, Tuple
import shutil

class ImageProcessor:
    def __init__(self, device: str = "cpu"):
        """
        初始化图片处理器
        :param device: 设备类型 ("cpu", "cuda", "mps")
        """
        self.device = self._init_device(device)
        self.generator = PoseOutlineGenerator(
            sam_checkpoint=MODEL_CONFIG['sam_checkpoint'],
            model_type=MODEL_CONFIG['model_type'],
            device=self.device
        )
    
    def _init_device(self, preferred_device: str) -> str:
        """
        初始化并返回可用的设备
        :param preferred_device: 首选设备
        :return: 实际使用的设备
        """
        if preferred_device == "mps" and torch.backends.mps.is_available():
            print("Using Apple Silicon GPU (MPS)")
            return "mps"
        elif preferred_device == "cuda" and torch.cuda.is_available():
            torch.cuda.set_device(0)
            torch.cuda.empty_cache()
            print("Using NVIDIA GPU (CUDA)")
            return "cuda"
        else:
            print("Using CPU")
            return "cpu"

    def process_image(
        self,
        user_id: str,
        input_source: Union[str, BinaryIO],
        move_file: bool = False,
        custom_name: Optional[str] = None
    ) -> Dict[str, str]:
        """
        处理用户图片并上传
        :param user_id: 用户ID
        :param input_source: 输入图片的路径或文件流
        :param move_file: 是否移动源文件（仅对文件路径有效）
        :param custom_name: 自定义文件名（可选，不包含扩展名），如果不提供则使用原始文件名
        :return: 上传成功的图片URL字典
        """
        # 第一步：处理图片
        input_path, generated_paths = self._generate_images(user_id, input_source, move_file, custom_name)
        if not input_path or not generated_paths:
            return {}

        # 第二步：上传文件
        return self._upload_files(user_id, input_path, generated_paths)

    def _get_base_name(self, input_source: Union[str, BinaryIO], custom_name: Optional[str] = None) -> Optional[str]:
        """获取基础文件名（含扩展名）"""
        if isinstance(input_source, str):
            return custom_name or os.path.splitext(os.path.basename(input_source))[0]
        elif custom_name:
            return custom_name
        else:
            print("Custom name is required when using file stream")
            return None

    def _generate_images(
        self,
        user_id: str,
        input_source: Union[str, BinaryIO],
        move_file: bool = False,
        custom_name: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[Dict[str, str]]]:
        """
        生成所有需要的图片
        :return: (输入文件路径, 生成的文件路径字典)
        """
        base_name = self._get_base_name(input_source, custom_name)
        if not base_name:
            return None, None

        # 创建输入和输出目录
        input_dir = os.path.join(IMAGE_CONFIG['base_dir'], user_id, IMAGE_CONFIG['subdirs']['input'])
        mask_base_dir = os.path.join(IMAGE_CONFIG['base_dir'], user_id, IMAGE_CONFIG['subdirs']['mask'])
        process_dir = os.path.join(mask_base_dir, base_name)  # 每个图片的处理结果都放在单独的目录中

        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(process_dir, exist_ok=True)

        # 保存输入文件
        input_path = os.path.join(input_dir, f"{base_name}.jpg")
        try:
            if isinstance(input_source, str):
                if not os.path.exists(input_source):
                    print(f"Source file not found: {input_source}")
                    return None, None
                if move_file:
                    shutil.move(input_source, input_path)
                else:
                    shutil.copy2(input_source, input_path)
            else:
                with open(input_path, 'wb') as f:
                    shutil.copyfileobj(input_source, f)
        except Exception as e:
            print(f"Failed to save input file: {str(e)}")
            return None, None

        print(f"Processing image: {input_path}")

        # 调用生成器处理图片
        success, image_results = self.generator.process_image(input_path, process_dir)
        if not success or not image_results:
            print("Failed to process image")
            return None, None

        print("Image processed successfully!")

        # 把生成的outline文件复制到输入文件的目录中，要给outline增加相同的文件名前缀
        outline_path = os.path.join(input_dir, f"{base_name}_outline.png")
        shutil.copy2(image_results['outline'], outline_path)
        
        # 返回所有生成的文件路径
        generated_paths = {
            ImageType.INPUT: input_path,
            ImageType.MASK_MASK: image_results['mask'],
            ImageType.MASK_OUTLINE: [
                image_results['outline'],
                outline_path
            ],
            ImageType.MASK_POSE: image_results['pose'],
            ImageType.MASK_PERSON: image_results['person']
        }

        return input_path, generated_paths

    def _upload_files(
        self,
        user_id: str,
        input_path: str,
        generated_paths: Dict[ImageType, str]
    ) -> Dict[str, str]:
        """
        上传生成的文件
        :param user_id: 用户ID
        :param input_path: 输入文件路径
        :param generated_paths: 生成的文件路径字典
        :return: 上传成功的图片URL字典
        """
        uploader = OSSUploader(user_id)
        results = uploader.upload_files(generated_paths)
        
        if results:
            print("\nUploaded files:")
            for img_type, url in results.items():
                print(f"- {img_type}: {url}")
            return results
        else:
            print("No images were uploaded successfully")
            return {} 