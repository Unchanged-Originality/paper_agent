"""
向量数据库模块 - 使用 ChromaDB
"""
import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional
from pathlib import Path
from loguru import logger

from config.settings import settings


class VectorStore:
    """ChromaDB向量存储封装"""
    
    def __init__(self, collection_name: str):
        """
        初始化向量存储
        
        Args:
            collection_name: 集合名称
        """
        self.collection_name = collection_name
        
        # 初始化ChromaDB客户端（持久化存储）
        self.client = chromadb.PersistentClient(
            path=str(settings.db_dir),
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
        )
        
        logger.info(f"向量集合 '{collection_name}' 已加载，当前文档数: {self.collection.count()}")
    
    def add(self, 
            ids: List[str],
            embeddings: List[List[float]],
            documents: Optional[List[str]] = None,
            metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        添加向量到数据库
        
        Args:
            ids: 文档ID列表
            embeddings: 嵌入向量列表
            documents: 原始文档列表（可选）
            metadatas: 元数据列表（可选）
        """
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        logger.debug(f"添加 {len(ids)} 条记录到 {self.collection_name}")
    
    def query(self,
              query_embedding: List[float],
              n_results: int = 5,
              where: Optional[Dict] = None,
              include: List[str] = ["documents", "metadatas", "distances"]
              ) -> Dict[str, Any]:
        """
        查询相似向量
        
        Args:
            query_embedding: 查询向量
            n_results: 返回结果数量
            where: 过滤条件
            include: 返回字段
            
        Returns:
            查询结果
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=include
        )
        return results
    
    def get(self, ids: Optional[List[str]] = None, 
            where: Optional[Dict] = None) -> Dict[str, Any]:
        """获取指定文档"""
        return self.collection.get(ids=ids, where=where)
    
    def update(self,
               ids: List[str],
               embeddings: Optional[List[List[float]]] = None,
               documents: Optional[List[str]] = None,
               metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """更新文档"""
        self.collection.update(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
    
    def delete(self, ids: Optional[List[str]] = None,
               where: Optional[Dict] = None) -> None:
        """删除文档"""
        self.collection.delete(ids=ids, where=where)
        logger.debug(f"从 {self.collection_name} 删除记录")
    
    def count(self) -> int:
        """获取文档数量"""
        return self.collection.count()
    
    def clear(self) -> None:
        """清空集合"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"集合 {self.collection_name} 已清空")


# 论文和图像的向量存储实例
_paper_store: Optional[VectorStore] = None
_image_store: Optional[VectorStore] = None


def get_paper_store() -> VectorStore:
    """获取论文向量存储实例"""
    global _paper_store
    if _paper_store is None:
        _paper_store = VectorStore(settings.chroma_collection_papers)
    return _paper_store


def get_image_store() -> VectorStore:
    """获取图像向量存储实例"""
    global _image_store
    if _image_store is None:
        _image_store = VectorStore(settings.chroma_collection_images)
    return _image_store