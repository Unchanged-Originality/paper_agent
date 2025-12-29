"""
LLM分类器 - 使用本地Ollama进行论文分类
"""
import json
import requests
from typing import List, Optional, Dict
from loguru import logger

from config.settings import settings


class LLMClassifier:
    """基于LLM的论文分类器"""
    
    def __init__(self, model: str = None):
        """
        初始化分类器
        
        Args:
            model: Ollama模型名称
        """
        self.model = model or settings.ollama_model
        self.base_url = settings.ollama_base_url
        self.api_url = f"{self.base_url}/api/generate"
        
        # 验证Ollama服务
        self._check_ollama()
    
    def _check_ollama(self):
        """检查Ollama服务是否可用"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                if not any(self.model in name for name in model_names):
                    logger.warning(f"模型 {self.model} 未找到，可用模型: {model_names}")
                else:
                    logger.info(f"Ollama服务正常，使用模型: {self.model}")
            else:
                logger.warning("Ollama服务响应异常")
        except Exception as e:
            logger.warning(f"无法连接Ollama服务: {e}")
            logger.info("将使用基于关键词的备用分类方法")
    
    def classify(self, text: str, topics: List[str]) -> Dict[str, any]:
        """
        对文本进行分类
        
        Args:
            text: 待分类文本（论文摘要或全文）
            topics: 可选的主题列表
            
        Returns:
            分类结果 {"topic": str, "confidence": float, "reason": str}
        """
        # 构建提示词
        prompt = self._build_prompt(text, topics)
        
        try:
            # 调用Ollama API
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # 低温度，更确定性
                        "num_predict": 256
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                output = result.get('response', '')
                return self._parse_response(output, topics)
            else:
                logger.warning(f"LLM请求失败: {response.status_code}")
                return self._keyword_classify(text, topics)
                
        except Exception as e:
            logger.warning(f"LLM分类失败: {e}, 使用关键词分类")
            return self._keyword_classify(text, topics)
    
    def _build_prompt(self, text: str, topics: List[str]) -> str:
        """构建分类提示词"""
        topics_str = ", ".join(topics)
        
        # 截取文本前2000字符避免过长
        text = text[:2000] if len(text) > 2000 else text
        
        prompt = f"""You are a research paper classifier. Analyze the following paper content and classify it into one of these topics: {topics_str}.

Paper content:
---
{text}
---

Instructions:
1. Read the paper content carefully
2. Identify the main research area
3. Choose the most appropriate topic from the given list
4. Provide your classification in the following JSON format:

{{"topic": "chosen_topic", "confidence": 0.95, "reason": "brief explanation"}}

Only respond with the JSON, no other text."""

        return prompt
    
    def _parse_response(self, response: str, topics: List[str]) -> Dict:
        """解析LLM响应"""
        try:
            # 尝试提取JSON
            response = response.strip()
            
            # 查找JSON对象
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start != -1 and end > start:
                json_str = response[start:end]
                result = json.loads(json_str)
                
                # 验证topic是否在列表中
                if result.get('topic') in topics:
                    return result
                else:
                    # 尝试模糊匹配
                    for topic in topics:
                        if topic.lower() in result.get('topic', '').lower():
                            result['topic'] = topic
                            return result
            
            logger.warning(f"无法解析LLM响应: {response}")
            return {"topic": topics[-1], "confidence": 0.5, "reason": "Parse failed"}
            
        except json.JSONDecodeError:
            logger.warning(f"JSON解析失败: {response}")
            return {"topic": topics[-1], "confidence": 0.5, "reason": "JSON parse error"}
    
    def _keyword_classify(self, text: str, topics: List[str]) -> Dict:
        """基于关键词的备用分类方法"""
        text_lower = text.lower()
        
        # 主题关键词映射
        keyword_map = {
            "Computer Vision": ["image", "vision", "cnn", "object detection", "segmentation", 
                               "visual", "pixel", "convolution", "resnet", "yolo", "图像"],
            "Natural Language Processing": ["language", "nlp", "text", "bert", "gpt", "transformer",
                                            "word", "sentence", "token", "embedding", "语言", "文本"],
            "Reinforcement Learning": ["reinforcement", "reward", "policy", "agent", "environment",
                                       "q-learning", "dqn", "ppo", "rl", "强化学习"],
            "Machine Learning": ["machine learning", "classification", "regression", "clustering",
                                "supervised", "unsupervised", "机器学习"],
            "Deep Learning": ["deep learning", "neural network", "layer", "backpropagation",
                             "gradient", "深度学习", "神经网络"]
        }
        
        scores = {}
        for topic in topics:
            if topic in keyword_map:
                keywords = keyword_map[topic]
                score = sum(1 for kw in keywords if kw in text_lower)
                scores[topic] = score
            else:
                scores[topic] = 0
        
        if scores:
            best_topic = max(scores, key=scores.get)
            max_score = scores[best_topic]
            
            if max_score > 0:
                return {
                    "topic": best_topic,
                    "confidence": min(0.9, 0.5 + max_score * 0.1),
                    "reason": f"Keyword matching (score: {max_score})"
                }
        
        return {
            "topic": topics[-1] if topics else "Other",
            "confidence": 0.3,
            "reason": "No strong keyword match"
        }


def get_classifier() -> LLMClassifier:
    """获取分类器实例"""
    return LLMClassifier()