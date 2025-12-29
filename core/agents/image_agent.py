"""
图像管理Agent - 以文搜图功能
"""
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
from tqdm import tqdm

from config.settings import settings
from core.embeddings.image_embedder import get_image_embedder
from core.database.vector_store import get_image_store


@dataclass
class ImageSearchResult:
    """图像搜索结果"""
    file_path: str
    filename: str
    score: float
    metadata: Dict


class ImageAgent:
    """图像管理智能体"""
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
    
    def __init__(self):
        """初始化Agent"""
        self.embedder = get_image_embedder()
        self.store = get_image_store()
        
        logger.info("ImageAgent 初始化完成")
    
    def index_image(self, image_path: str, metadata: Optional[Dict] = None) -> Dict:
        """
        索引单张图像
        
        Args:
            image_path: 图像路径
            metadata: 附加元数据
            
        Returns:
            索引结果
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"图像不存在: {image_path}")
        
        if image_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"不支持的图像格式: {image_path.suffix}")
        
        logger.debug(f"索引图像: {image_path.name}")
        
        # 生成图像嵌入
        embedding = self.embedder.embed_image(image_path)
        
        # 准备元数据
        image_metadata = {
            "filename": image_path.name,
            "file_path": str(image_path.absolute()),
            "suffix": image_path.suffix.lower(),
            **(metadata or {})
        }
        
        # 存储到向量数据库
        doc_id = f"img_{image_path.stem}_{hash(str(image_path))}"
        
        self.store.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[image_path.name],
            metadatas=[image_metadata]
        )
        
        return {
            "status": "success",
            "id": doc_id,
            "file": image_path.name
        }
    
    def index_folder(self, 
                     folder_path: str,
                     recursive: bool = True) -> Dict:
        """
        批量索引文件夹中的图像
        
        Args:
            folder_path: 文件夹路径
            recursive: 是否递归扫描子文件夹
            
        Returns:
            索引结果统计
        """
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"文件夹不存在: {folder_path}")
        
        # 扫描图像文件
        image_files = []
        pattern = "**/*" if recursive else "*"
        
        for ext in self.SUPPORTED_FORMATS:
            image_files.extend(folder_path.glob(f"{pattern}{ext}"))
            image_files.extend(folder_path.glob(f"{pattern}{ext.upper()}"))
        
        logger.info(f"找到 {len(image_files)} 张图像")
        
        results = {
            "total": len(image_files),
            "success": 0,
            "failed": 0,
            "errors": []
        }
        
        # 批量处理
        batch_paths = []
        batch_metadatas = []
        batch_ids = []
        batch_size = 32
        
        for img_path in tqdm(image_files, desc="索引图像"):
            try:
                batch_paths.append(img_path)
                batch_metadatas.append({
                    "filename": img_path.name,
                    "file_path": str(img_path.absolute()),
                    "suffix": img_path.suffix.lower()
                })
                batch_ids.append(f"img_{img_path.stem}_{hash(str(img_path))}")
                
                # 达到批次大小时处理
                if len(batch_paths) >= batch_size:
                    self._process_batch(batch_paths, batch_ids, batch_metadatas)
                    results["success"] += len(batch_paths)
                    batch_paths = []
                    batch_metadatas = []
                    batch_ids = []
                    
            except Exception as e:
                logger.error(f"索引失败 {img_path.name}: {e}")
                results["failed"] += 1
                results["errors"].append({"file": img_path.name, "error": str(e)})
        
        # 处理剩余的图像
        if batch_paths:
            try:
                self._process_batch(batch_paths, batch_ids, batch_metadatas)
                results["success"] += len(batch_paths)
            except Exception as e:
                results["failed"] += len(batch_paths)
        
        logger.info(f"索引完成: 成功 {results['success']}, 失败 {results['failed']}")
        return results
    
    def _process_batch(self, paths: List[Path], ids: List[str], 
                       metadatas: List[Dict]) -> None:
        """批量处理图像"""
        embeddings = self.embedder.embed_images(paths)
        documents = [p.name for p in paths]
        
        self.store.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
    
    def search(self, 
               query: str,
               top_k: int = None) -> List[ImageSearchResult]:
        """
        以文搜图 - 使用自然语言查找图像
        
        Args:
            query: 自然语言查询（如"日落的海滩"）
            top_k: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        top_k = top_k or settings.default_top_k
        
        logger.info(f"图像搜索: {query}")
        
        # 使用CLIP的文本编码器生成查询嵌入
        query_embedding = self.embedder.embed_text(query)
        
        # 查询向量数据库
        results = self.store.query(
            query_embedding=query_embedding,
            n_results=top_k
        )
        
        # 处理结果
        search_results = []
        
        if results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                distance = results['distances'][0][i] if results['distances'] else 0
                
                # 转换为相似度分数
                score = 1 - distance if distance < 1 else 1 / (1 + distance)
                
                search_results.append(ImageSearchResult(
                    file_path=metadata.get('file_path', ''),
                    filename=metadata.get('filename', 'Unknown'),
                    score=score,
                    metadata=metadata
                ))
        
        return search_results
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "total_images": self.store.count()
        }


def get_image_agent() -> ImageAgent:
    """获取图像管理Agent实例"""
    return ImageAgent()