# -*- coding: utf-8 -*-
import cv2
import numpy as np
from PIL import Image
import os
import mediapipe as mp
from segment_anything import sam_model_registry, SamPredictor
import torch
from controlnet_aux import HEDdetector
from transformers import pipeline
import torch.nn.functional as F
from oss_uploader import OSSUploader

class PoseOutlineGenerator:
    def __init__(self, sam_checkpoint, model_type="vit_b", device=None, oss_config=None):
        """
        初始化生成器
        Args:
            sam_checkpoint: SAM 模型权重路径
            model_type: SAM 模型类型
            device: 运行设备
            oss_config: OSS 配置信息
        """
        # 自动选择最佳设备
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        self.device = device
        print(f"Using device: {self.device}")
        if self.device == "cuda" and torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        elif self.device == "mps":
            print("GPU: Apple Silicon (MPS)")
        
        # 初始化 SAM
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.eval()  # 设置为评估模式
        self.sam.to(device)
        
        # 确保输入数据也在正确的设备上
        def to_device(x):
            if isinstance(x, torch.Tensor):
                return x.to(device)
            return x
        
        # 将模型参数移动到指定设备
        self.sam = self.sam.to(device)
        
        # 初始化边缘检测器
        self.edge_detector = HEDdetector.from_pretrained("lllyasviel/Annotators")
        if hasattr(self.edge_detector, 'to'):
            self.edge_detector = self.edge_detector.to(device)
        
        # 初始化 MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils  # 添加绘图工具
        self.mp_drawing_styles = mp.solutions.drawing_styles  # 添加绘图样式
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5
        )
        
        self.device = device
        
        # 设置预测器参数
        self.predictor = SamPredictor(self.sam)
        # 设置 SAM 的预测参数
        self.predictor.model.mask_threshold = 0.95        # 大幅提高掩码阈值
        self.predictor.model.iou_threshold = 0.99        # 保持高 IoU 阈值
        self.predictor.model.stability_score_thresh = 0.99  # 提高稳定性要求
        
        # 初始化 MiDaS 深度估计模型
        self.depth_estimator = pipeline(
            "depth-estimation",
            model="vinvino02/glpn-nyu",
            device="cpu"  # 目前 transformers 在 MPS 上可能不稳定，统一使用 CPU
        )
        print(f"Depth estimator device: cpu")
        
        # 初始化 OSS 上传器
        self.oss_uploader = None
        if oss_config:
            self.oss_uploader = OSSUploader(
                access_key_id=oss_config['access_key_id'],
                access_key_secret=oss_config['access_key_secret'],
                endpoint=oss_config['endpoint'],
                bucket_name=oss_config['bucket_name']
            )

    def get_pose_points(self, image):
        """
        获取姿态关键点
        Args:
            image: RGB 图像
        Returns:
            list: 关键点列表 [(x, y), ...]
        """
        results = self.pose.process(image)
        if not results.pose_landmarks:
            return None
            
        points = []
        height, width = image.shape[:2]
        
        # 获取所有关键点
        for landmark in results.pose_landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            points.append([x, y])
            
        return np.array(points)

    def get_face_points(self, landmarks, image_shape):
        """
        从 pose landmarks 中提取人脸关键点
        """
        height, width = image_shape[:2]
        # MediaPipe Pose 的人脸关键点索引
        face_indices = [
            0,    # nose
            2,    # left_eye_inner
            3,    # left_eye
            4,    # left_eye_outer
            1,    # right_eye_inner
            6,    # right_eye
            5,    # right_eye_outer
            7,    # left_ear
            8,    # right_ear
            9,    # mouth_left
            10,   # mouth_right
        ]
        
        points = []
        for idx in face_indices:
            landmark = landmarks[idx]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            points.append([x, y])
            
        return np.array(points)

    def get_depth_mask(self, image, pose_points):
        """
        使用深度信息生成前景掩码
        """
        # 将 numpy array 转换为 PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # 获取深度图
        depth = self.depth_estimator(image)["depth"]
        depth = np.array(depth)
        
        # 归一化深度图到 0-1
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        
        # 将深度图调整到与输入图像相同的尺寸
        depth = cv2.resize(depth, (image.size[0], image.size[1]))
        
        # 使用姿态关键点位置的深度作为参考
        person_depths = []
        for point in pose_points:
            x, y = point.astype(int)
            if 0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]:
                person_depths.append(depth[y, x])
        
        if not person_depths:
            return None
            
        # 计算人物区域的平均深度和标准差
        mean_depth = np.mean(person_depths)
        std_depth = np.std(person_depths)
        
        # 创建深度掩码
        depth_mask = np.zeros_like(depth)
        # 将接近人物深度的区域标记为前景
        depth_mask = np.abs(depth - mean_depth) < (std_depth * 2)
        
        # 形态学处理改善掩码质量
        kernel = np.ones((5,5), np.uint8)
        depth_mask = cv2.morphologyEx(depth_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        return depth_mask

    def segment_person(self, image, results=None):
        """
        使用 SAM 分割人物，并移除人脸区域
        """
        # 获取姿态和人脸关键点
        if results is None:
            results = self.pose.process(image)
        if not results.pose_landmarks:
            print("No pose detected in the image")
            return None
            
        # 获取所有姿态关键点
        height, width = image.shape[:2]
        pose_points = []
        for landmark in results.pose_landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            pose_points.append([x, y])
        pose_points = np.array(pose_points)
        
        # 设置图像
        self.predictor.set_image(image)
        
        # 使用姿态关键点作为提示点
        input_points = pose_points
        input_labels = np.array([1] * len(input_points))
        
        # 获取 SAM 的基础掩码
        masks, scores, _ = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True
        )
        
        # 选择得分最高的掩码
        top_2_indices = np.argsort(scores)[-2:]
        mask = np.logical_or(masks[top_2_indices[0]], masks[top_2_indices[1]])
        mask = mask.astype(np.uint8)
        
        # 获取深度掩码
        depth_mask = self.get_depth_mask(image, pose_points)
        if depth_mask is not None:
            # 1. 定义边缘区域
            kernel_size = 31  # 控制边缘区域的宽度
            dilated = cv2.dilate(mask, np.ones((kernel_size, kernel_size), np.uint8))
            eroded = cv2.erode(mask, np.ones((kernel_size, kernel_size), np.uint8))
            edge_region = dilated & ~eroded
            
            # 2. 保持原始掩码的主体部分不变
            combined_mask = mask.copy()
            
            # 3. 只在边缘区域应用深度过滤
            edge_depth_mask = np.logical_and(edge_region, depth_mask)
            combined_mask[edge_region > 0] = edge_depth_mask[edge_region > 0]
            
            # 移除小区域
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                combined_mask.astype(np.uint8), connectivity=8
            )
            if len(stats) > 1:
                # 只保留最大的连通区域
                max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                mask = (labels == max_label).astype(np.uint8)
        
        # 移除小的噪点区域
        # 1. 找到所有连通区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        # 2. 找到最大区域的面积
        areas = stats[1:, cv2.CC_STAT_AREA]  # 跳过背景（索引0）
        if len(areas) > 0:
            max_area = np.max(areas)
            
            # 3. 创建新的掩码，只保留大于最大区域1%的区域
            clean_mask = np.zeros_like(mask)
            min_area = max_area * 0.01  # 可以调整这个比例
            for i in range(1, num_labels):  # 跳过背景（索引0）
                if stats[i, cv2.CC_STAT_AREA] >= min_area:
                    clean_mask[labels == i] = 1
            
            mask = clean_mask
        
        # 使用人脸关键点创建掩码
        face_points = self.get_face_points(results.pose_landmarks.landmark, image.shape)
        if len(face_points) > 0:
            # 创建凸包
            hull = cv2.convexHull(face_points)
            
            # 创建人脸掩码
            face_mask = np.zeros_like(mask)
            cv2.fillConvexPoly(face_mask, hull, 1)
            
            # 扩大人脸区域
            kernel = np.ones((15,15), np.uint8)
            face_mask = cv2.dilate(face_mask, kernel)
            
            # 应用渐变边缘
            face_mask = cv2.GaussianBlur(face_mask, (21, 21), 0)
            
            # 在掩码中移除人脸区域（使用渐变）
            mask = mask * (1 - face_mask)
        
        return mask

    def generate_soft_edge(self, image, mask):
        """
        使用边缘检测器生成细节轮廓
        """
        # 1. 使用 mask 提取人物
        person = image.copy()
        person[mask == 0] = [0, 0, 0]  # 将背景设为黑色
        
        # 2. 使用边缘检测器处理
        edge_image = self.edge_detector(person, detect_resolution=1024)
        
        # 转换为 numpy array
        edge_array = np.array(edge_image)
        
        # 确保边缘图像与原图大小一致
        if edge_array.shape[:2] != image.shape[:2]:
            edge_array = cv2.resize(edge_array, (image.shape[1], image.shape[0]))
        
        return edge_array

    def process_edges(self, edges):
        """
        处理边缘图像
        """
        # 获取原始图像尺寸
        height, width = edges.shape[:2]
        outline = np.zeros((height, width, 4), dtype=np.uint8)
        
        # 设置黑色背景
        outline[..., :3] = 0
        
        # 如果是彩色图像，转换为灰度图
        if len(edges.shape) == 3:
            edges_gray = cv2.cvtColor(edges, cv2.COLOR_RGB2GRAY)
        else:
            edges_gray = edges
            
        # 确保灰度图尺寸正确
        if edges_gray.shape[:2] != (height, width):
            edges_gray = cv2.resize(edges_gray, (width, height))
            
        # 使用灰度图作为 alpha 通道
        outline[..., 3] = edges_gray
        
        return outline

    def draw_pose_landmarks(self, image, results):
        """
        绘制骨骼关键点和连接线
        """
        # 创建副本以免修改原图
        pose_image = image.copy()
        
        # 绘制骨骼点和连接线
        self.mp_drawing.draw_landmarks(
            pose_image,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        return pose_image

    def process_image(self, input_path, output_path):
        """
        处理图像并保��结果
        Args:
            input_path: 输入图像路径
            output_path: 输出图像路径
        Returns:
            tuple: (是否成功, 上传结果)
        """
        try:
            # 获取输出路径的目录和文件名
            output_dir = output_path
          
            # 构建输出路径
            mask_output_path = os.path.join(output_dir, f"mask.png")
            person_output_path = os.path.join(output_dir, f"person.png")
            outline_output_path = os.path.join(output_dir, f"outline.png")
            pose_output_path = os.path.join(output_dir, f"pose.png")
            
            # 读取图像
            image = cv2.imread(input_path)
            if image is None:
                print(f"Error: Could not read image '{input_path}'")
                return False, None
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 1. 获取姿态信息并生成 mask
            results = self.pose.process(image_rgb)
            if not results.pose_landmarks:
                print("No pose detected in the image")
                return False, None
            
            # 生成并保存骨骼图
            pose_image = self.draw_pose_landmarks(image_rgb, results)
            pose_image = Image.fromarray(pose_image)
            pose_image.save(pose_output_path)
            
            # 生成 mask
            mask = self.segment_person(image_rgb, results)
            if mask is None:
                return False, None
            
            # 保存 mask 图像
            mask_image = Image.fromarray(mask * 255)
            mask_image.save(mask_output_path)
            
            # 2. 生成并保存人物抠图（带透明通道）
            person = np.zeros((image_rgb.shape[0], image_rgb.shape[1], 4), dtype=np.uint8)
            person[..., :3] = image_rgb  # 复制原始RGB颜色
            person[..., 3] = mask * 255  # 设置透明通道
            person_image = Image.fromarray(person)
            person_image.save(person_output_path)
            
            # 3. 生成并保存轮廓
            edges = self.generate_soft_edge(image_rgb, mask)
            outline = self.process_edges(edges)
            
            # 验证输出尺寸
            if outline.shape[:2] != image_rgb.shape[:2]:
                print(f"Warning: Output size mismatch. Original: {image_rgb.shape[:2]}, Output: {outline.shape[:2]}")
                outline = cv2.resize(outline, (image_rgb.shape[1], image_rgb.shape[0]))
            
            outline_image = Image.fromarray(outline)
            outline_image.save(outline_output_path)
            
            image_results = {
                    'mask': mask_output_path,
                    'person':person_output_path,
                    'outline': outline_output_path,
                    'pose': pose_output_path,
            }
            
            return True, image_results
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return False, None 