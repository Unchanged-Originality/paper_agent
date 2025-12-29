"""
PDF处理模块 - 提取文本内容
"""
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from loguru import logger

from config.settings import settings


@dataclass
class PDFChunk:
    """PDF文本块"""
    text: str
    page_num: int
    chunk_id: int
    metadata: Dict


@dataclass
class PDFDocument:
    """PDF文档"""
    path: Path
    title: str
    total_pages: int
    full_text: str
    chunks: List[PDFChunk]
    metadata: Dict


class PDFProcessor:
    """PDF处理器"""
    
    def __init__(self, 
                 chunk_size: int = None,
                 chunk_overlap: int = None):
        """
        初始化PDF处理器
        
        Args:
            chunk_size: 文本块大小
            chunk_overlap: 块之间的重叠
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
    
    def extract_text(self, pdf_path: Path) -> Tuple[str, int]:
        """
        提取PDF全文
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            (全文文本, 总页数)
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
        
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            
            for page_num, page in enumerate(doc):
                text = page.get_text()
                full_text += f"\n[Page {page_num + 1}]\n{text}"
            
            total_pages = len(doc)
            doc.close()
            
            return full_text, total_pages
            
        except Exception as e:
            logger.error(f"提取PDF文本失败 {pdf_path}: {e}")
            raise
    
    def extract_text_by_page(self, pdf_path: Path) -> List[Tuple[int, str]]:
        """
        按页提取PDF文本
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            [(页码, 文本), ...]
        """
        pdf_path = Path(pdf_path)
        doc = fitz.open(pdf_path)
        
        pages = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            pages.append((page_num + 1, text))
        
        doc.close()
        return pages
    
    def chunk_text(self, text: str, 
                   source_path: str = "",
                   page_num: int = 0) -> List[PDFChunk]:
        """
        将文本分割成块
        
        Args:
            text: 原始文本
            source_path: 来源文件路径
            page_num: 页码
            
        Returns:
            文本块列表
        """
        chunks = []
        
        # 简单的滑动窗口分块
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # 尝试在句子边界切分
            if end < len(text):
                # 寻找最后一个句号、问号或换行
                for sep in ['. ', '。', '?\n', '\n\n']:
                    last_sep = chunk_text.rfind(sep)
                    if last_sep > self.chunk_size // 2:
                        chunk_text = chunk_text[:last_sep + len(sep)]
                        end = start + len(chunk_text)
                        break
            
            chunk = PDFChunk(
                text=chunk_text.strip(),
                page_num=page_num,
                chunk_id=chunk_id,
                metadata={
                    "source": str(source_path),
                    "page": page_num,
                    "chunk_id": chunk_id
                }
            )
            
            if chunk.text:  # 只添加非空块
                chunks.append(chunk)
            
            chunk_id += 1
            start = end - self.chunk_overlap
        
        return chunks
    
    def process(self, pdf_path: Path) -> PDFDocument:
        """
        完整处理PDF文件
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            PDFDocument对象
        """
        pdf_path = Path(pdf_path)
        logger.info(f"处理PDF: {pdf_path.name}")
        
        # 提取文本
        full_text, total_pages = self.extract_text(pdf_path)
        
        # 提取标题（尝试从文件名或首页获取）
        title = self._extract_title(pdf_path, full_text)
        
        # 分块
        all_chunks = []
        pages = self.extract_text_by_page(pdf_path)
        
        for page_num, page_text in pages:
            page_chunks = self.chunk_text(
                page_text, 
                source_path=str(pdf_path),
                page_num=page_num
            )
            all_chunks.extend(page_chunks)
        
        # 构建文档对象
        doc = PDFDocument(
            path=pdf_path,
            title=title,
            total_pages=total_pages,
            full_text=full_text,
            chunks=all_chunks,
            metadata={
                "filename": pdf_path.name,
                "path": str(pdf_path),
                "total_pages": total_pages,
                "total_chunks": len(all_chunks)
            }
        )
        
        logger.info(f"PDF处理完成: {title}, {total_pages}页, {len(all_chunks)}个文本块")
        return doc
    
    def _extract_title(self, pdf_path: Path, full_text: str) -> str:
        """尝试提取论文标题"""
        # 方法1: 使用文件名（去掉扩展名和常见前缀）
        filename = pdf_path.stem
        
        # 清理文件名
        for prefix in ['[', '(', '【']:
            if prefix in filename:
                filename = filename.split(prefix)[-1]
        
        # 方法2: 尝试从PDF元数据获取
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            if metadata.get('title'):
                title = metadata['title']
                doc.close()
                return title
            doc.close()
        except:
            pass
        
        # 方法3: 从首页文本提取（假设标题在前200字符内）
        first_text = full_text[:500].strip()
        lines = [l.strip() for l in first_text.split('\n') if l.strip()]
        if lines:
            # 取最长的前几行作为可能的标题
            potential_title = lines[0]
            if len(potential_title) < 10 and len(lines) > 1:
                potential_title = lines[1]
            return potential_title[:100]
        
        return filename


def get_pdf_processor() -> PDFProcessor:
    """获取PDF处理器实例"""
    return PDFProcessor()