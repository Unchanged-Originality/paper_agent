"""
论文管理Agent - 核心业务逻辑
"""
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
from tqdm import tqdm

from config.settings import settings
from core.embeddings.text_embedder import get_text_embedder
from core.database.vector_store import get_paper_store, VectorStore
from core.processors.pdf_processor import get_pdf_processor, PDFDocument
from core.agents.classifier import get_classifier


@dataclass
class SearchResult:
    """搜索结果"""
    file_path: str
    title: str
    score: float
    snippet: str
    page: int
    topic: str


class PaperAgent:
    """论文管理智能体"""
    
    def __init__(self):
        """初始化Agent"""
        self.embedder = get_text_embedder()
        self.store = get_paper_store()
        self.processor = get_pdf_processor()
        self.classifier = get_classifier()
        
        logger.info("PaperAgent 初始化完成")
    
    def add_paper(self, 
                  pdf_path: str,
                  topics: Optional[List[str]] = None,
                  auto_classify: bool = True,
                  move_file: bool = True) -> Dict:
        """
        添加单篇论文到系统
        
        Args:
            pdf_path: PDF文件路径
            topics: 可选的分类主题列表
            auto_classify: 是否自动分类
            move_file: 是否移动文件到分类目录
            
        Returns:
            处理结果
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"文件不存在: {pdf_path}")
        
        topics = topics or settings.default_topics
        
        logger.info(f"添加论文: {pdf_path.name}")
        
        # 1. 处理PDF
        doc = self.processor.process(pdf_path)
        
        # 2. 分类
        classification = None
        if auto_classify:
            # 使用摘要或前2000字符进行分类
            classify_text = doc.full_text[:3000]
            classification = self.classifier.classify(classify_text, topics)
            logger.info(f"分类结果: {classification['topic']} (置信度: {classification['confidence']:.2f})")
        
        # 3. 生成嵌入并存储
        chunk_texts = [chunk.text for chunk in doc.chunks]
        if chunk_texts:
            embeddings = self.embedder.embed_documents(chunk_texts)
            
            # 准备存储数据
            ids = [f"{pdf_path.stem}_{i}" for i in range(len(chunk_texts))]
            metadatas = [
                {
                    "filename": pdf_path.name,
                    "title": doc.title,
                    "page": chunk.page_num,
                    "chunk_id": chunk.chunk_id,
                    "topic": classification['topic'] if classification else "Unknown",
                    "file_path": str(pdf_path)
                }
                for chunk in doc.chunks
            ]
            
            self.store.add(
                ids=ids,
                embeddings=embeddings,
                documents=chunk_texts,
                metadatas=metadatas
            )
        
        # 4. 移动文件到对应目录
        new_path = pdf_path
        if move_file and classification:
            topic_dir = settings.papers_dir / classification['topic'].replace(" ", "_")
            topic_dir.mkdir(parents=True, exist_ok=True)
            new_path = topic_dir / pdf_path.name
            
            if new_path != pdf_path:
                shutil.copy2(pdf_path, new_path)
                logger.info(f"文件已复制到: {new_path}")
        
        return {
            "status": "success",
            "title": doc.title,
            "pages": doc.total_pages,
            "chunks": len(doc.chunks),
            "topic": classification['topic'] if classification else None,
            "confidence": classification['confidence'] if classification else None,
            "new_path": str(new_path)
        }
    
    def search(self, 
               query: str,
               top_k: int = None,
               topic_filter: Optional[str] = None) -> List[SearchResult]:
        """
        语义搜索论文
        
        Args:
            query: 自然语言查询
            top_k: 返回结果数量
            topic_filter: 按主题筛选
            
        Returns:
            搜索结果列表
        """
        top_k = top_k or settings.default_top_k
        
        logger.info(f"搜索查询: {query}")
        
        # 1. 生成查询嵌入
        query_embedding = self.embedder.embed_query(query)
        
        # 2. 构建过滤条件
        where = None
        if topic_filter:
            where = {"topic": topic_filter}
        
        # 3. 查询向量数据库
        results = self.store.query(
            query_embedding=query_embedding,
            n_results=top_k * 2,  # 多查一些用于去重
            where=where
        )
        
        # 4. 处理结果（按文件去重并取最高分）
        file_scores: Dict[str, SearchResult] = {}
        
        if results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                metadata = results['metadatas'][0][i]
                document = results['documents'][0][i] if results['documents'] else ""
                distance = results['distances'][0][i] if results['distances'] else 0
                
                # 转换距离为相似度分数 (ChromaDB使用L2距离或余弦距离)
                score = 1 - distance if distance < 1 else 1 / (1 + distance)
                
                file_path = metadata.get('file_path', metadata.get('filename', ''))
                
                # 去重：保留每个文件的最高分
                if file_path not in file_scores or score > file_scores[file_path].score:
                    file_scores[file_path] = SearchResult(
                        file_path=file_path,
                        title=metadata.get('title', 'Unknown'),
                        score=score,
                        snippet=document[:200] + "..." if len(document) > 200 else document,
                        page=metadata.get('page', 0),
                        topic=metadata.get('topic', 'Unknown')
                    )
        
        # 5. 排序并返回top_k结果
        sorted_results = sorted(file_scores.values(), key=lambda x: x.score, reverse=True)
        return sorted_results[:top_k]
    
    def batch_organize(self,
                       folder_path: str,
                       topics: Optional[List[str]] = None,
                       move_files: bool = True) -> Dict:
        """
        批量整理文件夹中的论文
        
        Args:
            folder_path: 待整理的文件夹路径
            topics: 分类主题列表
            move_files: 是否移动文件
            
        Returns:
            整理结果统计
        """
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"文件夹不存在: {folder_path}")
        
        topics = topics or settings.default_topics
        
        # 扫描所有PDF
        pdf_files = list(folder_path.glob("**/*.pdf"))
        logger.info(f"找到 {len(pdf_files)} 个PDF文件")
        
        results = {
            "total": len(pdf_files),
            "success": 0,
            "failed": 0,
            "by_topic": {topic: 0 for topic in topics},
            "details": []
        }
        
        for pdf_path in tqdm(pdf_files, desc="处理论文"):
            try:
                result = self.add_paper(
                    pdf_path=str(pdf_path),
                    topics=topics,
                    auto_classify=True,
                    move_file=move_files
                )
                results["success"] += 1
                results["by_topic"][result["topic"]] = results["by_topic"].get(result["topic"], 0) + 1
                results["details"].append({
                    "file": pdf_path.name,
                    "topic": result["topic"],
                    "status": "success"
                })
            except Exception as e:
                logger.error(f"处理失败 {pdf_path.name}: {e}")
                results["failed"] += 1
                results["details"].append({
                    "file": pdf_path.name,
                    "error": str(e),
                    "status": "failed"
                })
        
        logger.info(f"批量整理完成: 成功 {results['success']}, 失败 {results['failed']}")
        return results
    
    def list_papers(self, topic: Optional[str] = None) -> List[Dict]:
        """
        列出已索引的论文
        
        Args:
            topic: 按主题筛选
            
        Returns:
            论文列表
        """
        where = {"topic": topic} if topic else None
        results = self.store.get(where=where)
        
        # 去重（按文件）
        seen_files = set()
        papers = []
        
        if results['metadatas']:
            for metadata in results['metadatas']:
                file_path = metadata.get('file_path', metadata.get('filename', ''))
                if file_path not in seen_files:
                    seen_files.add(file_path)
                    papers.append({
                        "title": metadata.get('title', 'Unknown'),
                        "file_path": file_path,
                        "topic": metadata.get('topic', 'Unknown')
                    })
        
        return papers
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        total_chunks = self.store.count()
        papers = self.list_papers()
        
        topic_counts = {}
        for paper in papers:
            topic = paper.get('topic', 'Unknown')
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        return {
            "total_papers": len(papers),
            "total_chunks": total_chunks,
            "by_topic": topic_counts
        }




    def search_files(self, 
                    query: str,
                    top_k: int = None,
                    topic_filter: Optional[str] = None) -> List[Dict]:
        """
        文件索引搜索 - 仅返回相关文件列表
        
        Args:
            query: 自然语言查询
            top_k: 返回结果数量
            topic_filter: 按主题筛选
            
        Returns:
            文件列表 [{"file_path": str, "title": str, "topic": str, "score": float}, ...]
        """
        top_k = top_k or settings.default_top_k
        
        logger.info(f"文件索引查询: {query}")
        
        # 1. 生成查询嵌入
        query_embedding = self.embedder.embed_query(query)
        
        # 2. 构建过滤条件
        where = None
        if topic_filter:
            where = {"topic": topic_filter}
        
        # 3. 查询向量数据库（多查一些用于去重）
        results = self.store.query(
            query_embedding=query_embedding,
            n_results=top_k * 5,
            where=where
        )
        
        # 4. 按文件去重，保留最高分
        file_scores: Dict[str, Dict] = {}
        
        if results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i] if results['distances'] else 0
                
                # 转换为相似度分数
                score = 1 - distance if distance < 1 else 1 / (1 + distance)
                
                file_path = metadata.get('file_path', metadata.get('filename', ''))
                
                # 只保留每个文件的最高分
                if file_path and (file_path not in file_scores or score > file_scores[file_path]['score']):
                    file_scores[file_path] = {
                        "file_path": file_path,
                        "filename": Path(file_path).name if file_path else "Unknown",
                        "title": metadata.get('title', 'Unknown'),
                        "topic": metadata.get('topic', 'Unknown'),
                        "score": score
                    }
        
        # 5. 排序并返回top_k
        sorted_files = sorted(file_scores.values(), key=lambda x: x['score'], reverse=True)
        return sorted_files[:top_k]


    def list_all_files(self, topic: Optional[str] = None) -> List[Dict]:
        """
        列出所有已索引的文件
        
        Args:
            topic: 按主题筛选
            
        Returns:
            文件列表
        """
        where = {"topic": topic} if topic else None
        results = self.store.get(where=where)
        
        # 去重
        seen_files = set()
        files = []
        
        if results['metadatas']:
            for metadata in results['metadatas']:
                file_path = metadata.get('file_path', metadata.get('filename', ''))
                if file_path and file_path not in seen_files:
                    seen_files.add(file_path)
                    files.append({
                        "file_path": file_path,
                        "filename": Path(file_path).name if file_path else "Unknown",
                        "title": metadata.get('title', 'Unknown'),
                        "topic": metadata.get('topic', 'Unknown')
                    })
        
        return files

def get_paper_agent() -> PaperAgent:
    """获取论文管理Agent实例"""
    return PaperAgent()