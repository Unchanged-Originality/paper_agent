

# 测试论文功能
python main.py add_paper "examples/sample_papers/attention-is-all-you-need.pdf.pdf"
python main.py search_paper "attention mechanism" --files-only
python main.py search_paper "Transformer architecture"
python main.py organize examples/sample_papers/

# 测试图像功能
python main.py index_images examples/sample_images/
python main.py search_image "beautiful sunset"


# 查看统计
python main.py stats