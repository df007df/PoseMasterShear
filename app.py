# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
from image_processor import ImageProcessor
from image_processor import ImageType
from config import API_CONFIG
from functools import wraps
import requests
from io import BytesIO
import os
import torch
from urllib.parse import urlparse
import uuid
import time
import random
import string

app = Flask(__name__)
processor = None

def require_api_key(f):
    """API Key 验证装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        api_key = request.headers.get(API_CONFIG['header_name'])
        
        if not api_key:
            return jsonify({
                'success': False,
                'error': 'Missing API Key',
                'execution_time': time.time() - start_time
            }), 401
            
        if api_key != API_CONFIG['api_key']:
            return jsonify({
                'success': False,
                'error': 'Invalid API Key',
                'execution_time': time.time() - start_time
            }), 401
            
        return f(*args, **kwargs)
    return decorated_function

def init_processor():
    global processor
    if processor is None:
        # 自动选择最佳设备
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        processor = ImageProcessor(device=device)

def is_valid_url(url):
    """检查是否是有效的URL"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def download_image(url):
    """从URL下载图片"""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return BytesIO(response.content)
        return None
    except Exception as e:
        print(f"Failed to download image: {str(e)}")
        return None

def generate_unique_filename():
    """生成唯一的文件名
    格式: 15位字符（数字和英文字母组合）
    例如: a2b3c4d5e6f7g8h
    """
    # 获取当前时间戳的后6位作为前缀
    timestamp = str(int(time.time()))[-6:]
    
    # 生成剩余的9位随机字符（数字和字母）
    chars = string.ascii_lowercase + string.digits
    random_chars = ''.join(random.choice(chars) for _ in range(9))
    
    return f"{timestamp}{random_chars}"

@app.route('/process_image', methods=['POST'])
@require_api_key
def process_image():
    """
    处理图片的API接口
    
    请求头：
    X-API-Key: your-api-key    # API验证密钥
    
    请求体格式：
    {
        "user_id": "123",                    # 必需，用户ID
        "image_url": "http://...",           # 可选，图片URL
        "custom_name": "custom_filename"      # 可选，自定义文件名
    }
    
    或者使用 multipart/form-data 格式上传文件：
    - user_id: 用户ID
    - image_file: 图片文件
    - custom_name: 自定义文件名（可选）
    
    返回格式：
    {
        "success": true/false,
        "outline_url": "https://...",  # 成功时返回轮廓图片的OSS URL
        "input_url": "https://...",    # 输入图片的OSS URL
        "filename": "xxx",             # 使用的文件名
        "execution_time": 1.23,        # 执行耗时（秒）
        "error": "错误信息"            # 失败时返回错误信息
    }
    """
    start_time = time.time()
    try:
        # 记录请求参数
        request_data = {
            'form_data': dict(request.form),
            'json_data': request.json if request.is_json else None,
            'files': [f.filename for f in request.files.values()] if request.files else None,
            'headers': {
                k: v for k, v in request.headers.items() 
                if k.lower() not in ['x-api-key']  # 不记录敏感信息
            }
        }
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] New request:")
        print(f"Request Data: {request_data}")
        
        # 初始化处理器
        init_processor()
        
        # 获取用户ID
        user_id = request.form.get('user_id') or request.json.get('user_id')
        if not user_id:
            error_msg = 'Missing user_id'
            print(f"Error: {error_msg}")
            return jsonify({
                'success': False,
                'error': error_msg,
                'execution_time': time.time() - start_time
            }), 400
            
        # 获取自定义文件名（可选）或生成随机文件名
        custom_name = request.form.get('custom_name') or request.json.get('custom_name')
        if not custom_name:
            custom_name = generate_unique_filename()
            
        # 处理图片来源
        image_source = None
        
        # 检查是否有文件上传
        if 'image_file' in request.files:
            image_file = request.files['image_file']
            if image_file.filename:
                image_source = image_file.stream
                
        # 如果没有文件上传，检查是否提供了URL
        elif 'image_url' in request.json:
            image_url = request.json['image_url']
            if not is_valid_url(image_url):
                error_msg = 'Invalid image URL'
                print(f"Error: {error_msg} - URL: {image_url}")
                return jsonify({
                    'success': False,
                    'error': error_msg,
                    'execution_time': time.time() - start_time
                }), 400
                
            image_source = download_image(image_url)
            if image_source is None:
                error_msg = 'Failed to download image from URL'
                print(f"Error: {error_msg} - URL: {image_url}")
                return jsonify({
                    'success': False,
                    'error': error_msg,
                    'execution_time': time.time() - start_time
                }), 400
        else:
            error_msg = 'No image provided (neither file nor URL)'
            print(f"Error: {error_msg}")
            return jsonify({
                'success': False,
                'error': error_msg,
                'execution_time': time.time() - start_time
            }), 400
            
        # 处理图片
        print(f"Processing image for user_id: {user_id}, filename: {custom_name}")
        results = processor.process_image(
            user_id=user_id,
            input_source=image_source,
            custom_name=custom_name
        )
        
        # 检查处理结果
        if not results or ImageType.MASK_OUTLINE not in results or ImageType.INPUT not in results:
            error_msg = 'Failed to process image'
            print(f"Error: {error_msg} - Results: {results}")
            return jsonify({
                'success': False,
                'error': error_msg,
                'execution_time': time.time() - start_time
            }), 500
            
        # 返回结果
        execution_time = time.time() - start_time
        print(f"Success: Processed image in {round(execution_time, 3)}s")
        print(f"Results: {results}")
        return jsonify({
            'success': True,
            # 注意这里返回的是数组，只取1
            'outline_url': results[ImageType.MASK_OUTLINE][1],
            'input_url': results[ImageType.INPUT],
            'filename': custom_name,
            'execution_time': round(execution_time, 3)  # 保留3位小数
        })
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = str(e)
        print(f"Error: Unexpected exception - {error_msg}")
        return jsonify({
            'success': False,
            'error': error_msg,
            'execution_time': round(execution_time, 3)
        }), 500

if __name__ == '__main__':
    # 在生产环境，应该使用合适的WSGI服务器
    app.run(host='0.0.0.0', port=5001) 