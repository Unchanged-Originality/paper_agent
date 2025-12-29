# 🤖 本地多模态AI智能助手

本项目是一个基于 Python 的本地多模态 AI 智能助手，旨在解决本地大量文献和图像素材管理困难的问题。不同于传统的文件名搜索，本项目利用多模态神经网络技术，实现对内容的**语义搜索**和**自动分类**。



## 核心功能



### 智能文献管理

- **语义搜索**：支持使用自然语言提问，如"Transformer的核心架构是什么？"，系统返回最相关的论文及具体片段

- **自动分类**: 基于LLM自动分析论文内容，将其归类到CV/NLP/RL等目录

- **批量整理**: 支持对现有的混乱文件夹进行“一键整理”，自动扫描所有 PDF，识别主题并归档到相应目录。

- **文件索引**: 持仅返回相关文件列表，方便快速定位所需文献



### 智能图像管理

- **以文搜图**: : 使用CLIP模型，支持自然语言描述搜索图片，如"海边的日落"



## 🔧 技术架构

| 课程编号 |        课程名称        | 开设学期 | 类型 |       课件       |       作业       |      期末资料      |
| :------: | :--------------------: | :------: | :--: | :--------------: | :--------------: | :----------------: |
| M502034B |   [高级软件开发与实践] | 暑假小学期 |  硕  |                  |  |                    |

| 组件 | 技术选型 | 说明 |

| :------: | :----------: | :------: |

| 文本嵌入 | sentence-transformers (all-mpnet-base-v2) | 高质量语义向量 |

| 图像嵌入 | OpenCLIP (ViT-L-14) | 图文多模态匹配 |

| 向量数据库 | ChromaDB | 本地持久化存储 |

| LLM分类 | Ollama + Qwen2 | 本地大模型推理 |

| PDF解析 | PyMuPDF | 高效文本提取 |



## 环境配置



### 系统要求

- Python 3.8+

- NVIDIA GPU (推荐，用于加速)

- 8GB+ RAM



### 安装步骤





# 1. 克隆仓库
```py

git clone https://github.com/Unchanged-Originality/paper_agent.git
cd agent
```


# 2. 创建虚拟环境
```
conda create -n agent python=3.10 -y
conda activate agent
```


# 3. 安装PyTorch 
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```


# 4. 安装依赖
```
pip install -r requirements.txt
```


# 5. 安装Ollama并下载模型
```
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull qwen2:7b
```


##  使用命令

添加论文
```
# 添加单篇论文并自动分类
python main.py add\_paper ./paper.pdf

# 指定分类主题
python main.py add\_paper ./paper.pdf --topics "CV,NLP,RL,Other"

# 只索引不移动文件
python main.py add\_paper ./paper.pdf --no-move
```


搜索论文
```
# 语义搜索
python main.py search\_paper "Transformer的核心架构是什么"

#文件搜索
python main.py search\_paper "attention mechanism" --files-only

# 限制返回数量
python main.py search\_paper "attention mechanism" --top-k 10


# 按主题筛选
python main.py search\_paper "图像分类" --topic "CV"
```


批量整理
```
# 整理整个文件夹
python main.py organize ./messy\_papers/


# 自定义分类主题
python main.py organize ./papers --topics "CV,NLP,RL,ML,Other"
```


图像管理
```
# 索引图片文件夹
python main.py index\_images ./my\_photos/

# 以文搜图
python main.py search\_image "海边的日落"
python main.py search\_image "a cute cat playing" --top-k 10
```


查看统计
```
python main.py stats

```





