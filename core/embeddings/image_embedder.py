"""
图像嵌入模块 - 使用 OpenCLIP
"""
import torch
from typing import List, Union
from pathlib import Path
from PIL import Image
import open_clip
from loguru import logger

from config.settings import settings


class ImageEmbedder:
    """图像向量化器（基于CLIP）"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        logger.info(f"加载CLIP模型: {settings.clip_model}")
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            settings.clip_model,
            pretrained=settings.clip_pretrained,
            device=settings.device
        )
        self.tokenizer = open_clip.get_tokenizer(settings.clip_model)
        
        self.model.eval()
        self.embedding_dim = self.model.visual.output_dim
        logger.info(f"CLIP嵌入维度: {self.embedding_dim}")
        self._initialized = True
    
    @torch.no_grad()
    def embed_image(self, image_path: Union[str, Path]) -> List[float]:
        """
        将单张图像转换为向量
        
        Args:
            image_path: 图像路径
            
        Returns:
            嵌入向量（列表格式）
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"图像不存在: {image_path}")
        
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(settings.device)
        
        image_features = self.model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()[0].tolist()
    
    @torch.no_grad()
    def embed_images(self, image_paths: List[Union[str, Path]], 
                     batch_size: int = 32) -> List[List[float]]:
        """批量嵌入图像"""
        all_embeddings = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            
            for path in batch_paths:
                try:
                    image = Image.open(path).convert("RGB")
                    batch_images.append(self.preprocess(image))
                except Exception as e:
                    logger.warning(f"处理图像失败 {path}: {e}")
                    continue
            
            if batch_images:
                batch_tensor = torch.stack(batch_images).to(settings.device)
                features = self.model.encode_image(batch_tensor)
                features = features / features.norm(dim=-1, keepdim=True)
                all_embeddings.extend(features.cpu().numpy().tolist())
        
        return all_embeddings
    
    @torch.no_grad()
    def embed_text(self, text: str) -> List[float]:
        """
        将文本转换为CLIP文本向量（用于以文搜图）
        
        Args:
            text: 查询文本
            
        Returns:
            嵌入向量
        """
        text_tokens = self.tokenizer([text]).to(settings.device)
        text_features = self.model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy()[0].tolist()
    
    @torch.no_grad()
    def compute_similarity(self, text: str, 
                           image_paths: List[Union[str, Path]]) -> List[float]:
        """
        计算文本与图像列表的相似度
        
        Args:
            text: 查询文本
            image_paths: 图像路径列表
            
        Returns:
            相似度分数列表
        """
        text_embedding = torch.tensor(self.embed_text(text)).to(settings.device)
        
        similarities = []
        for path in image_paths:
            try:
                image_embedding = torch.tensor(self.embed_image(path)).to(settings.device)
                sim = torch.cosine_similarity(
                    text_embedding.unsqueeze(0), 
                    image_embedding.unsqueeze(0)
                ).item()
                similarities.append(sim)
            except Exception as e:
                logger.warning(f"计算相似度失败 {path}: {e}")
                similarities.append(0.0)
        
        return similarities


def get_image_embedder() -> ImageEmbedder:
    """获取图像嵌入器实例"""
    return ImageEmbedder()