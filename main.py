import sys
import click
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint
from loguru import logger

# 配置日志
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/log1029.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG"
)

console = Console()


@click.group()
@click.version_option(version="1.0.0", prog_name="AI文献助手")
def cli():
    pass




@cli.command("add_paper")
@click.argument("path", type=click.Path(exists=True))
@click.option("--topics", "-t", default=None, 
              help="分类主题，逗号分隔。如: 'CV,NLP,RL'")
@click.option("--no-move", is_flag=True, default=False,
              help="不移动文件到分类目录")
def add_paper(path: str, topics: str, no_move: bool):
    """
    添加单篇论文并自动分类
    
    示例:
        python main.py add_paper ./paper.pdf
        python main.py add_paper ./paper.pdf --topics "CV,NLP,RL"
        python main.py add_paper ./paper.pdf --no-move
    """
    from core.agents.paper_agent import get_paper_agent
    
    console.print(Panel.fit("添加论文", style="bold blue"))
    
    # 解析主题
    topic_list = None
    if topics:
        topic_list = [t.strip() for t in topics.split(",")]
    
    try:
        agent = get_paper_agent()
        result = agent.add_paper(
            pdf_path=path,
            topics=topic_list,
            auto_classify=True,
            move_file=not no_move
        )
        
        # 显示结果
        console.print("\n[green]添加成功![/green]\n")
        
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("属性", style="dim")
        table.add_column("值")
        
        table.add_row("标题", result["title"])
        table.add_row("页数", str(result["pages"]))
        table.add_row("文本块数", str(result["chunks"]))
        table.add_row("分类主题", result["topic"] or "N/A")
        table.add_row("置信度", f"{result['confidence']:.2%}" if result["confidence"] else "N/A")
        table.add_row("存储路径", result["new_path"])
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]错误: {e}[/red]")
        logger.exception("添加论文失败")
        sys.exit(1)



@cli.command("search_paper")
@click.argument("query")
@click.option("--top-k", "-k", default=5, type=int, help="返回结果数量")
@click.option("--topic", "-t", default=None, help="按主题筛选")
@click.option("--files-only", "-f", is_flag=True, default=False, 
              help="仅返回文件列表，不显示详细片段")
def search_paper(query: str, top_k: int, topic: str, files_only: bool):
    """
    语义搜索论文
    
    示例:
        python main.py search_paper "Transformer的核心架构"
        python main.py search_paper "attention" --files-only
        python main.py search_paper "图像分类" -f -k 10
    """
    from core.agents.paper_agent import get_paper_agent
    
    console.print(Panel.fit(f"🔍 搜索论文: {query}", style="bold blue"))
    
    try:
        agent = get_paper_agent()
        
        if files_only:
            # 仅返回文件列表
            results = agent.search_files(query=query, top_k=top_k, topic_filter=topic)
            
            if not results:
                console.print("\n[yellow]未找到相关论文[/yellow]")
                return
            
            console.print(f"\n[green]找到 {len(results)} 个相关文件:[/green]\n")
            
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("#", style="dim", width=4)
            table.add_column("文件名", style="bold")
            table.add_column("标题", max_width=40)
            table.add_column("主题", style="magenta")
            table.add_column("相似度", justify="right")
            
            for i, result in enumerate(results, 1):
                table.add_row(
                    str(i),
                    result["filename"],
                    result["title"][:40] + "..." if len(result["title"]) > 40 else result["title"],
                    result["topic"],
                    f"{result['score']:.1%}"
                )
            
            console.print(table)
            
            # 输出文件路径列表（方便复制）
            console.print("\n[dim]文件路径列表:[/dim]")
            for result in results:
                console.print(f"  {result['file_path']}")
        
        else:
            # 返回详细结果（原有逻辑）
            results = agent.search(query=query, top_k=top_k, topic_filter=topic)
            
            if not results:
                console.print("\n[yellow]未找到相关论文[/yellow]")
                return
            
            console.print(f"\n[green]找到 {len(results)} 个相关结果:[/green]\n")
            
            for i, result in enumerate(results, 1):
                panel_content = f"""
[bold]📄 {result.title}[/bold]

[dim]文件路径:[/dim] {result.file_path}
[dim]相关页码:[/dim] 第 {result.page} 页
[dim]分类主题:[/dim] {result.topic}
[dim]相似度:[/dim] {result.score:.2%}

[dim]相关片段:[/dim]
{result.snippet}
"""
                console.print(Panel(panel_content, title=f"结果 #{i}", border_style="cyan"))
        
    except Exception as e:
        console.print(f"[red] 错误: {e}[/red]")
        logger.exception("搜索论文失败")
        sys.exit(1)



@cli.command("organize")
@click.argument("folder", type=click.Path(exists=True))
@click.option("--topics", "-t", default=None,
              help="分类主题，逗号分隔")
@click.option("--no-move", is_flag=True, default=False,
              help="只分类不移动文件")
def organize(folder: str, topics: str, no_move: bool):
    """
    批量整理文件夹中的论文
    
    示例:
        python main.py organize ./messy_papers/
        python main.py organize ./papers --topics "CV,NLP,RL,Other"
    """
    from core.agents.paper_agent import get_paper_agent
    
    console.print(Panel.fit("批量整理论文", style="bold blue"))
    
    topic_list = None
    if topics:
        topic_list = [t.strip() for t in topics.split(",")]
    
    try:
        agent = get_paper_agent()
        result = agent.batch_organize(
            folder_path=folder,
            topics=topic_list,
            move_files=not no_move
        )
        
        console.print("\n[green]整理完成![/green]\n")
        
        # 总体统计
        table = Table(title="整理统计", show_header=True, header_style="bold cyan")
        table.add_column("统计项", style="dim")
        table.add_column("数量", justify="right")
        
        table.add_row("总计", str(result["total"]))
        table.add_row("成功", f"[green]{result['success']}[/green]")
        table.add_row("失败", f"[red]{result['failed']}[/red]")
        
        console.print(table)
        
        # 分类统计
        console.print("\n[bold]按主题分布:[/bold]")
        topic_table = Table(show_header=True, header_style="bold magenta")
        topic_table.add_column("主题")
        topic_table.add_column("数量", justify="right")
        
        for topic, count in result["by_topic"].items():
            if count > 0:
                topic_table.add_row(topic, str(count))
        
        console.print(topic_table)
        
    except Exception as e:
        console.print(f"[red] 错误: {e}[/red]")
        logger.exception("批量整理失败")
        sys.exit(1)


def index_images(folder: str, recursive: bool):
    """
    索引文件夹中的图像
    
    示例:
        python main.py index_images ./my_photos/
        python main.py index_images ./images --no-recursive
    """
    from core.agents.image_agent import get_image_agent
    
    console.print(Panel.fit("索引图像", style="bold blue"))
    
    try:
        agent = get_image_agent()
        result = agent.index_folder(folder_path=folder, recursive=recursive)
        
        console.print("\n[green]索引完成![/green]\n")
        
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("统计项", style="dim")
        table.add_column("数量", justify="right")
        
        table.add_row("总计", str(result["total"]))
        table.add_row("成功", f"[green]{result['success']}[/green]")
        table.add_row("失败", f"[red]{result['failed']}[/red]")
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]错误: {e}[/red]")
        logger.exception("索引图像失败")
        sys.exit(1)


@cli.command("search_image")
@click.argument("query")
@click.option("--top-k", "-k", default=5, type=int, help="返回结果数量")
def search_image(query: str, top_k: int):
    """
    以文搜图 - 用自然语言搜索图片
    
    示例:
        python main.py search_image "海边的日落"
        python main.py search_image "a cute cat" --top-k 10
    """
    from core.agents.image_agent import get_image_agent
    
    console.print(Panel.fit(f"搜索图像: {query}", style="bold blue"))
    
    try:
        agent = get_image_agent()
        results = agent.search(query=query, top_k=top_k)
        
        if not results:
            console.print("\n[yellow]未找到相关图像[/yellow]")
            return
        
        console.print(f"\n[green]找到 {len(results)} 个相关结果:[/green]\n")
        
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("#", style="dim", width=4)
        table.add_column("文件名")
        table.add_column("相似度", justify="right")
        table.add_column("路径", style="dim")
        
        for i, result in enumerate(results, 1):
            table.add_row(
                str(i),
                result.filename,
                f"{result.score:.2%}",
                result.file_path
            )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]错误: {e}[/red]")
        logger.exception("搜索图像失败")
        sys.exit(1)




@cli.command("stats")
def stats():
    """
    显示系统统计信息
    """
    from core.agents.paper_agent import get_paper_agent
    from core.agents.image_agent import get_image_agent
    
    console.print(Panel.fit("系统统计", style="bold blue"))
    
    try:
        paper_agent = get_paper_agent()
        image_agent = get_image_agent()
        
        paper_stats = paper_agent.get_stats()
        image_stats = image_agent.get_stats()
        
        # 论文统计
        console.print("\n[bold cyan]论文统计:[/bold cyan]")
        table1 = Table(show_header=False)
        table1.add_column("项目", style="dim")
        table1.add_column("数量", justify="right")
        
        table1.add_row("已索引论文", str(paper_stats["total_papers"]))
        table1.add_row("文本块总数", str(paper_stats["total_chunks"]))
        
        console.print(table1)
        
        if paper_stats["by_topic"]:
            console.print("\n[bold]按主题分布:[/bold]")
            for topic, count in paper_stats["by_topic"].items():
                console.print(f"  • {topic}: {count}")
        
        # 图像统计
        console.print("\n[bold cyan]图像统计:[/bold cyan]")
        table2 = Table(show_header=False)
        table2.add_column("项目", style="dim")
        table2.add_column("数量", justify="right")
        
        table2.add_row("已索引图像", str(image_stats["total_images"]))
        
        console.print(table2)
        
    except Exception as e:
        console.print(f"[red]错误: {e}[/red]")
        logger.exception("获取统计失败")
        sys.exit(1)


@cli.command("clear")
@click.option("--papers", is_flag=True, help="清空论文索引")
@click.option("--images", is_flag=True, help="清空图像索引")
@click.option("--all", "clear_all", is_flag=True, help="清空所有索引")
@click.confirmation_option(prompt="确定要清空索引吗?")
def clear(papers: bool, images: bool, clear_all: bool):
    """
    清空索引数据
    """
    from core.database.vector_store import get_paper_store, get_image_store
    
    if clear_all or papers:
        store = get_paper_store()
        store.clear()
        console.print("[green]论文索引已清空[/green]")
    
    if clear_all or images:
        store = get_image_store()
        store.clear()
        console.print("[green]图像索引已清空[/green]")
    
    if not (papers or images or clear_all):
        console.print("[yellow]请指定 --papers, --images 或 --all[/yellow]")


if __name__ == "__main__":
    cli()