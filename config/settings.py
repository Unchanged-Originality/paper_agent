#全局配置
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

@dataclass
class Settings:
    # 路径配置
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    papers_dir: Path = PROJECT_ROOT / "data" / "papers"
    images_dir: Path = PROJECT_ROOT / "data" / "images"
    db_dir: Path = PROJECT_ROOT / "data" / "db"
    
    # 模型配置
    text_embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    clip_model: str = "ViT-L-14"  
    clip_pretrained: str = "openai"
    
    # Ollama LLM配置
    ollama_model: str = "qwen2:7b"
    ollama_base_url: str = "http://localhost:11434"
    
    # ChromaDB配置
    chroma_collection_papers: str = "papers"
    chroma_collection_images: str = "images"
    
    # 默认分类主题
    default_topics: List[str] = field(default_factory=lambda: [
        "Computer Vision", 
        "Natural Language Processing", 
        "Reinforcement Learning",
        "Machine Learning",
        "Deep Learning",
        "Other"
    ])
    
    # 搜索配置
    default_top_k: int = 5
    chunk_size: int = 512  # 文本分块大小
    chunk_overlap: int = 50
    
    # 设备配置
    device: str = "cuda:1"  
    
    def __post_init__(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        
        # 为每个主题创建子目录
        for topic in self.default_topics:
            topic_dir = self.papers_dir / topic.replace(" ", "_")
            topic_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()