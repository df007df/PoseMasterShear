# -*- coding: utf-8 -*-
import cv2
import numpy as np
from PIL import Image
import os
import mediapipe as mp
from segment_anything import sam_model_registry, SamPredictor
import torch
from controlnet_aux import HEDdetector

class PoseOutlineGenerator:
    def __init__(self, sam_checkpoint, model_type="vit_b", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        初始化生成器
        Args:
            sam_checkpoint: SAM 模型权重路径
            model_type: SAM 模型类型
            device: 运行设备
        """
        # 初始化 SAM
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.eval()  # 设置为评估模式
        self.sam.to(device)
        
        # 初始化边缘检测器
        self.edge_detector = HEDdetector.from_pretrained("lllyasviel/Annotators")
        
        # 初始化 MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5
        )
        
        self.device = device
        
        # 设置预测器参数
        self.predictor = SamPredictor(self.sam)
        # 设置 SAM 的预测参数
        self.predictor.model.mask_threshold = 0.7        # 提高掩码阈值，减少噪点
        self.predictor.model.iou_threshold = 0.98       # 提高 IoU 阈值，增加区域一致性
        self.predictor.model.stability_score_thresh = 0.98  # 提高稳定性要求

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

    def segment_person(self, image):
        """
        使用 SAM 分割人物，并移除人脸区域
        """
        # 获取姿态和人脸关键点
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
        
        # 生成掩码
        masks, scores, _ = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True
        )
        
        # 选择得分最高的两个掩码并合并
        top_2_indices = np.argsort(scores)[-2:]
        mask = np.logical_or(masks[top_2_indices[0]], masks[top_2_indices[1]])
        mask = mask.astype(np.uint8)
        
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
        return edge_array

    def process_edges(self, edges):
        """
        处理边缘图像
        """
        # 确保边缘图像尺寸正确
        height, width = edges.shape[:2]
        outline = np.zeros((height, width, 4), dtype=np.uint8)
        
        # 设置黑色背景
        outline[..., :3] = 0
        
        # 如果是彩色图像，转换为灰度图
        if len(edges.shape) == 3:
            edges_gray = cv2.cvtColor(edges, cv2.COLOR_RGB2GRAY)
        else:
            edges_gray = edges
            
        # 使用灰度图作为 alpha 通道
        outline[..., 3] = edges_gray
        
        return outline

    def process_image(self, input_path, output_path):
        """
        处理图像并保存结果
        Args:
            input_path: 输入图像路径
            output_path: 输出图像路径
        Returns:
            bool: 是否成功
        """
        try:
            # 获取输出路径的目录和文件名
            output_dir = os.path.dirname(output_path)
            filename_without_ext = os.path.splitext(os.path.basename(output_path))[0]
            
            # 构建输出路径
            mask_output_path = os.path.join(output_dir, f"{filename_without_ext}_mask.png")
            person_output_path = os.path.join(output_dir, f"{filename_without_ext}_person.png")
            outline_output_path = os.path.join(output_dir, f"{filename_without_ext}_outline.png")
            
            # 读取图像
            image = cv2.imread(input_path)
            if image is None:
                print(f"Error: Could not read image '{input_path}'")
                return False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 1. 生成 mask
            mask = self.segment_person(image_rgb)
            if mask is None:
                return False
            
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
            outline_image = Image.fromarray(outline)
            outline_image.save(outline_output_path)
            
            return True
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return False 