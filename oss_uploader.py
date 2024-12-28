# -*- coding: utf-8 -*-

import oss2
import os
import shutil
from config import OSS_CONFIG, IMAGE_CONFIG
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union, BinaryIO

class ImageType(Enum):
    INPUT = auto()          # 输入的原始图片
    MASK_OUTLINE = auto()   # 遮罩轮廓图片（存储在input和mask目录）
    MASK_MASK = auto()      # 遮罩图片（存储在mask目录）
    MASK_PERSON = auto()    # 人像分割图片（存储在mask目录）
    MASK_POSE = auto()      # 姿态图片（存储在mask目录）

    @classmethod
    def get_subdir(cls, img_type) -> Optional[str]:
        """获取图片类型对应的子目录"""
        if img_type == cls.INPUT:
            return IMAGE_CONFIG['subdirs']['input']
        elif img_type == cls.MASK_OUTLINE:
            # 特殊处理：MASK_OUTLINE 需要同时存储在 input 和 mask 目录
            return None  # 返回None，让上层逻辑处理双重存储
        else:
            return IMAGE_CONFIG['subdirs']['mask']

    @classmethod
    def get_extension(cls, img_type) -> Optional[str]:
        """获取图片类型对应的文件扩展��"""
        type_to_key = {
            cls.INPUT: 'input',
            cls.MASK_OUTLINE: 'mask_outline',
            cls.MASK_MASK: 'mask_mask',
            cls.MASK_PERSON: 'mask_person',
            cls.MASK_POSE: 'mask_pose'
        }
        key = type_to_key.get(img_type)
        return IMAGE_CONFIG['extensions'].get(key) if key else None

    @classmethod
    def get_suffix(cls, img_type) -> Optional[str]:
        """获取图片类型对应的文件名后缀"""
        type_to_key = {
            cls.INPUT: 'input',
            cls.MASK_OUTLINE: 'mask_outline',
            cls.MASK_MASK: 'mask_mask',
            cls.MASK_PERSON: 'mask_person',
            cls.MASK_POSE: 'mask_pose'
        }
        key = type_to_key.get(img_type)
        return IMAGE_CONFIG['suffixes'].get(key) if key else None

class OSSUploader:
    def __init__(self, user_id: str):
        """
        初始化OSS上传器
        :param user_id: 用户ID，用于区分不同用户的图片
        """
        if not user_id:
            raise ValueError("user_id is required")
            
        # 验证 OSS 配置
        if not all([OSS_CONFIG['access_key_id'], 
                   OSS_CONFIG['access_key_secret'], 
                   OSS_CONFIG['bucket_name'],
                   OSS_CONFIG['endpoint']]):
            raise ValueError("Missing required OSS configuration")
            
        self.user_id = user_id
        try:
            self.auth = oss2.Auth(OSS_CONFIG['access_key_id'], OSS_CONFIG['access_key_secret'])
            self.bucket = oss2.Bucket(
                self.auth, 
                OSS_CONFIG['endpoint'],
                OSS_CONFIG['bucket_name']
            )
            # 测试连接
            self.bucket.get_bucket_info()
        except oss2.exceptions.ServerError as e:
            raise ValueError(f"Failed to connect to OSS: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to initialize OSS uploader: {str(e)}")
            
        self.base_dir = IMAGE_CONFIG['base_dir']
    
    def get_file_name(self, img_type: ImageType, base_id: str) -> str:
        """
        生成整的文件名
        :param img_type: 图片类型
        :param base_id: 基础ID（如 "5"）
        :return: 完整的文件名（如 "5_mask_outline.png"）
        """
        suffix = ImageType.get_suffix(img_type)
        extension = ImageType.get_extension(img_type)
        return f"{base_id}{suffix}{extension}"

    def get_local_paths(self, img_type: ImageType, base_id: str) -> List[str]:
        """
        获取本地文件完整路径（可能返回多个路径，如mask_outline）
        对于 MASK_OUTLINE，返回顺序固定为 [mask_path, input_path]
        :param img_type: 图片类型
        :param base_id: 基础ID
        :return: 完整的本地文件路径列表
        """
        paths = []
        file_name = self.get_file_name(img_type, base_id)
        
        if img_type == ImageType.MASK_OUTLINE:
            # MASK_OUTLINE 需要存储在两个目录
            mask_dir = os.path.join(self.base_dir, self.user_id, IMAGE_CONFIG['subdirs']['mask'])
            input_dir = os.path.join(self.base_dir, self.user_id, IMAGE_CONFIG['subdirs']['input'])
            
            os.makedirs(mask_dir, exist_ok=True)
            os.makedirs(input_dir, exist_ok=True)
            
            paths.extend([
                os.path.join(mask_dir, file_name),
                os.path.join(input_dir, file_name)
            ])
        else:
            # 其他类型只存储在一个目录
            subdir = ImageType.get_subdir(img_type)
            if subdir:
                full_dir = os.path.join(self.base_dir, self.user_id, subdir)
                os.makedirs(full_dir, exist_ok=True)
                paths.append(os.path.join(full_dir, file_name))
        
        return paths

    def save_input_file(self, source: Union[str, BinaryIO], base_id: str, move: bool = False) -> Optional[str]:
        """
        保存输入文件到对应的input目录
        :param source: 源文件路径或文件流
        :param base_id: 基础ID
        :param move: 是否移动文件（仅对文件路径有效）
        :return: 保存后的文件路径，失败返回None
        """
        paths = self.get_local_paths(ImageType.INPUT, base_id)
        if not paths:
            return None
        
        target_path = paths[0]

        try:
            if isinstance(source, str):
                if not os.path.exists(source):
                    print(f"Source file not found: {source}")
                    return None
                
                if move:
                    shutil.move(source, target_path)
                else:
                    shutil.copy2(source, target_path)
            else:
                with open(target_path, 'wb') as f:
                    shutil.copyfileobj(source, f)
            
            return target_path
        except Exception as e:
            print(f"Failed to save input file: {str(e)}")
            return None

    def _get_oss_key(self, file_path: str) -> str:
        """
        根据本地文件路径生成 OSS 存储路径
        :param file_path: 本地文件路径
        :return: OSS 存储路径
        """
        # 获取相对于 base_dir 的路径
        rel_path = os.path.relpath(file_path, IMAGE_CONFIG['base_dir'])
        
        # 分解路径以获取用户ID和子目录
        parts = rel_path.split(os.sep)
        if len(parts) < 3:  # 确保路径至少包含：用户ID/子目录/文件名
            raise ValueError(f"Invalid file path structure: {file_path}")
            
        user_id = parts[0]  # 用户ID
        local_subdir = parts[1]  # 本地子目录 (pose_input 或 pose_mask)
        filename = parts[-1]  # 文件名
        
        # 如果是在 pose_mask 目录下的子目录中，保留该子目录结构
        if local_subdir == IMAGE_CONFIG['subdirs']['mask'] and len(parts) > 3:
            mask_subdir = parts[2]  # 获取 mask 下的子目录名
            return os.path.join(
                OSS_CONFIG['base_dir'],
                user_id,
                local_subdir,
                mask_subdir,
                filename
            )
            
        # 构建 OSS 路径
        return os.path.join(
            OSS_CONFIG['base_dir'],
            user_id,
            local_subdir,
            filename
        )

    def upload_file(self, file_path: str) -> Optional[str]:
        """
        上传单个文件到 OSS
        :param file_path: 本地文件路径
        :return: 上传成功返回文件 URL，失败返回 None
        """
        try:
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return None

            # 生成 OSS 存储路径
            oss_key = self._get_oss_key(file_path)

            print(f"oss_key: {oss_key}: {file_path}")
            
            # 使用 OSS SDK 上传文件
            result = self.bucket.put_object_from_file(oss_key, file_path)
            
            # 检查上传是否成功
            if result.status == 200:
                # 返回文件 URL
                return f"https://{OSS_CONFIG['domain']}/{oss_key}"
            else:
                print(f"Failed to upload file {file_path}: status code {result.status}")
                return None
                
        except oss2.exceptions.OssError as e:
            print(f"OSS error while uploading file {file_path}: {str(e)}")
            return None
        except Exception as e:
            print(f"Failed to upload file {file_path}: {str(e)}")
            return None

    def upload_files(self, file_paths: Dict[ImageType, Union[str, List[str]]]) -> Dict[ImageType, Union[str, List[str]]]:
        """
        批量上传文件到 OSS
        :param file_paths: 文件路径字典，key 是 ImageType，value 可以是单个路径或路径列表
        :return: 上传成功的文件 URL 字典
        """
        results = {}
        
        for img_type, paths in file_paths.items():
            if isinstance(paths, str):
                # 单个文件路径
                url = self.upload_file(paths)
                if url:
                    results[img_type] = url
            elif isinstance(paths, list):
                # 文件路径列表
                urls = []
                for path in paths:
                    url = self.upload_file(path)
                    if url:
                        urls.append(url)
                if urls:
                    results[img_type] = urls if len(urls) > 1 else urls[0]
        
        return results 