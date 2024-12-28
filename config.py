import os

# OSS Configuration
OSS_CONFIG = {
    'access_key_id': os.getenv('OSS_ACCESS_KEY_ID', ''),      # 从环境变量获取
    'access_key_secret': os.getenv('OSS_ACCESS_KEY_SECRET', ''),
    'endpoint': os.getenv('OSS_ENDPOINT', 'oss-cn-hangzhou.aliyuncs.com'),  # 使用实际的阿里云 endpoint
    'bucket_name': os.getenv('OSS_BUCKET_NAME', ''),
    'domain': os.getenv('OSS_DOMAIN', ''),
    'base_dir': 'PoseMasterShear'  # OSS 存储的基础目录
}

# 确保所有必需的配置都已设置
if not all([OSS_CONFIG['access_key_id'], 
           OSS_CONFIG['access_key_secret'], 
           OSS_CONFIG['bucket_name']]):
    raise ValueError(
        "Missing required OSS configuration. Please set the following environment variables:\n"
        "- OSS_ACCESS_KEY_ID\n"
        "- OSS_ACCESS_KEY_SECRET\n"
        "- OSS_BUCKET_NAME"
    )

# 图片路径和文件配置
IMAGE_CONFIG = {
    'base_dir': '/Users/df007df/work/ios/PoseMasterShear/test',  # 基础目录
    'subdirs': {
        'input': 'pose_input',      # 输入图片和最终结果目录
        'mask': 'pose_mask',        # 处理过程中生成的所有图片目录
    },
    'extensions': {
        'input': '.jpg',       # 输入图片格式
        'mask_outline': '.png', # 遮罩轮廓图片格式
        'mask_mask': '.png',        # 遮罩图片格式
        'mask_person': '.png',      # 人像分割图片格式
        'mask_pose': '.jpg',        # 姿态图片格式
    },
    'suffixes': {
        'input': '',           # 输入图片后缀
        'mask_outline': '_mask_outline', # 遮罩轮廓图片后缀
        'mask_mask': '_mask',      # 遮罩图片后缀
        'mask_person': '_person',   # 人像分割图片后缀
        'mask_pose': '_pose',      # 姿态图片后缀
    }
}

# 模型配置
MODEL_CONFIG = {
    'sam_checkpoint': 'sam_vit_b_01ec64.pth',
    'model_type': 'vit_b'
}

# API Authentication
API_CONFIG = {
    'api_key': os.getenv('API_KEY', 'your-default-api-key'),  # 从环境变量获取API Key
    'header_name': 'X-API-Key'  # API Key的请求头名称
} 