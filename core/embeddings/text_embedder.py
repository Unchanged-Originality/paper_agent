"""
文本嵌入模块 - 使用 SentenceTransformers
"""
import torch
from typing import List, Union
from sentence_transformers import SentenceTransformer
from loguru import logger

from config.settings import settings


class TextEmbedder:
    """文本向量化器"""
    
    _instance = None
    
    def __new__(cls):
        """单例模式，避免重复加载模型"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        logger.info(f"加载文本嵌入模型: {settings.text_embedding_model}")
        self.model = SentenceTransformer(
            settings.text_embedding_model,
            device=settings.device
        )
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"文本嵌入维度: {self.embedding_dim}")
        self._initialized = True
    
    def embed(self, texts: Union[str, List[str]], 
              show_progress: bool = False) -> torch.Tensor:
        """
        将文本转换为向量
        
        Args:
            texts: 单个文本或文本列表
            show_progress: 是否显示进度条
            
        Returns:
            嵌入向量 (numpy array)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=show_progress,
            device=settings.device
        )
        
        return embeddings.cpu().numpy()
    
    def embed_query(self, query: str) -> List[float]:
        """嵌入查询文本，返回列表格式（适配ChromaDB）"""
        embedding = self.embed(query)
        return embedding[0].tolist()
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """嵌入文档列表，返回列表格式"""
        embeddings = self.embed(documents, show_progress=True)
        return embeddings.tolist()


# 便捷函数
def get_text_embedder() -> TextEmbedder:
    """获取文本嵌入器实例"""
    return TextEmbedder()