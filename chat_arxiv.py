import argparse
import datetime
import logging
import os
import re
import time
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import arxiv  # 导入官方 API 库
import tenacity
import requests

# 确保 logging 已配置 (从 chat_paper.py 导入时会自动配置)
try:
    import logging_config
except ImportError:
    logging.basicConfig(level=logging.INFO)

# 动态导入 Paper 类，以避免循环依赖
# 我们只在类型提示和函数内部导入它
try:
    # (!!!) 导入已修复，从 paper_class 导入 (!!!)
    from paper_class import Paper
except ImportError:
    # 如果单独运行此文件，定义一个临时的 Paper 占位符
    logging.warning("无法导入 'paper_class.Paper'，将使用临时占位符。")
    class Paper:
        def __init__(self, **kwargs):
            logging.info(f"临时 Paper 对象创建: {kwargs.get('title')}")
            pass

# ---------------------------------------------------------------------
# 实用函数
# ---------------------------------------------------------------------

def validateTitle(title: str) -> str:
    """清理标题以用作安全的文件名。"""
    if not title:
        return "untitled"
    title = title[:100] # 限制文件名长度
    rstr = r"[/\\:*?\"<>|]"
    title = re.sub(rstr, "_", title)
    title = re.sub(r"\s+", " ", title)
    title = re.sub(r"_+", "_", title)
    return title.strip(' _') or "untitled"

def _get_save_path(query: str) -> str:
    """
    (!!!) 修正：获取 API 论文保存的根目录 (!!!)
    """
    root_path = os.path.abspath(os.path.dirname(__file__))
    # (!!!) 修正：API下载的论文保存到 'api_downloads' 目录 (!!!)
    # 'myPapers' 目录保留给 --retriever local 模式使用。
    path = os.path.join(root_path, 'api_downloads', validateTitle(query))
    os.makedirs(path, exist_ok=True)
    return path

@tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
                stop=tenacity.stop_after_attempt(5), reraise=True)
def try_download_pdf(pdf_url: str, title: str, query: str) -> str:
    """
    带重试的下载函数。
   
    """
    path = _get_save_path(query) # (!!!) 此函数现在指向 'api_downloads' (!!!)
    base_name = f"{validateTitle(title)}.pdf"
    filename = os.path.join(path, base_name)
    
    if os.path.exists(filename):
        logging.info("文件已存在, 跳过下载: %s", filename)
    else:
        logging.info("正在下载: %s", pdf_url)
        # 添加 User-Agent 模拟浏览器
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
        }
        r = requests.get(pdf_url, headers=headers, timeout=30)
        r.raise_for_status() # 确保请求成功
        with open(filename, 'wb') as f:
            f.write(r.content)
        logging.info("已保存到: %s", filename)
    return filename

# ---------------------------------------------------------------------
# 核心 API 逻辑 (全新)
# ---------------------------------------------------------------------

def search_arxiv_api(query: str, max_fetch_results: int = 200) -> List[arxiv.Result]:
    """
    使用 arXiv API 搜索论文。
    - 重写
    
    我们一次性获取 N 篇最新的论文（按提交日期排序），
    后续步骤再根据 'days' 参数进行过滤。
    """
    logging.info(f"正在使用 arXiv API 搜索: query='{query}', sort_by=SubmittedDate")
    try:
        client = arxiv.Client(
            page_size=max_fetch_results, # 一次获取的页面大小
            delay_seconds=3,  # 遵守 API 速率限制
            num_retries=3
        )
        
        search = arxiv.Search(
            query=query,
            max_results=max_fetch_results, # 获取 N 篇
            sort_by=arxiv.SortCriterion.SubmittedDate # 按最新提交排序
        )
        
        results = list(client.results(search))
        logging.info(f"API 返回了 {len(results)} 篇论文")
        return results
        
    except Exception as e:
        logging.error(f"arXiv API 搜索失败: {e}", exc_info=True)
        return []

def _filter_papers_by_days(results: List[arxiv.Result], days: int) -> List[arxiv.Result]:
    """
    (!!!) 修正：根据 'days' 参数过滤 API 结果 (!!!)
    (移除了错误的 'break' 逻辑)
    """
    if days <= 0:
        return results # days <= 0 表示不过滤

    filtered_list = []
    # 转换为 aware datetime
    today = datetime.datetime.now(datetime.timezone.utc)
    day_limit = today - datetime.timedelta(days=days)
    
    logging.info(f"开始按 {days} 天过滤 (检查 {len(results)} 篇论文)...")
    
    for result in results:
        # result.published 和 result.updated 都是 aware datetime
        submission_time = result.published
        
        if submission_time >= day_limit:
            filtered_list.append(result)
        else:
            # (!!!) 修正：不再 'break' (!!!)
            # API 可能返回相关性排序，我们必须检查所有论文
            logging.debug(f"论文 '{result.title}' (日期: {submission_time.date()}) 已超出 {days} 天范围，已跳过。")
            pass # 继续检查下一篇论文
            
    logging.info(f"经过 'days={days}' 过滤后，剩余 {len(filtered_list)} 篇论文。")
    return filtered_list

def _download_and_create_paper(result: arxiv.Result, query: str) -> Paper:
    """
    (供并发调用) 下载单个 PDF 并创建 Paper 对象。
    - 新增
    """
    try:
        # (!!!) 修正：从 entry_id 派生 PDF URL (!!!)
        pdf_url = result.pdf_url
        entry_id = result.entry_id
        
        if not pdf_url:
            if entry_id and entry_id.startswith('http://arxiv.org/abs/'):
                pdf_url = entry_id.replace('/abs/', '/pdf/') + '.pdf'
            else:
                logging.warning(f"无法为 {entry_id or '未知条目'} 构建 PDF URL。跳过下载。")
                return None
        
        # 1. 下载 PDF
        #
        # (!!!) 修正：移除了 'chat_arxiv.' 前缀 (!!!)
        file_path = try_download_pdf(pdf_url, result.title, query)
        
        # 2. 提取所有元数据 (解决用户痛点)
        arxiv_id = result.get_short_id() # 例如 '2310.06825' (不含版本)
        authors = [str(a) for a in result.authors]
        published_date = result.published.date()
        
        # 3. 创建 Paper 对象
        #
        paper = Paper(
            path=file_path,
            title=result.title,
            url=result.entry_id, # 完整的 URL
            abs=result.summary.replace('\n', ' '), # 摘要
            authers=authors,
            published_date=published_date,
            arxiv_id=arxiv_id
            # citations 默认为 None
        )
        logging.info(f"成功创建 Paper 对象: {arxiv_id}")
        return paper
        
    except Exception as e:
        logging.warning(f"处理论文 {result.entry_id} 时失败: {e}", exc_info=True)
        return None

# ---------------------------------------------------------------------
# 主入口函数 (重写)
# ---------------------------------------------------------------------

def get_arxiv_papers(args) -> List[Paper]:
    """
    重构后的主函数：
    1. 使用 API 搜索 (search_arxiv_api)
    2. 按 'days' 过滤 (_filter_papers_by_days)
    3. 按 'max_results' 截断
    4. 并发下载并创建 Paper 对象
    - 重写
    """
    
    # 1. API 搜索 (获取一个较大的批次)
    # 我们使用 args.page_num * 50 (原默认值 1*50=50)
    # 并设置一个上限，例如 500
    page_num = getattr(args, 'page_num', 1)
    max_fetch = min(page_num * 50, 500) # 限制单次 API 拉取上限
    
    all_results = search_arxiv_api(args.query, max_fetch_results=max_fetch)
    
    # 2. 按 'days' 过滤
    # - 原有的过滤逻辑
    filtered_results = _filter_papers_by_days(all_results, getattr(args, 'days', 30))
    
    # 3. 按 'max_results' 截断
    #
    max_results = getattr(args, 'max_results', 5)
    papers_to_process = filtered_results[:max_results]
    
    if not papers_to_process:
        logging.info("没有找到符合条件的论文。")
        return []
        
    logging.info(f"将开始并发下载 {len(papers_to_process)} 篇论文...")
    
    # 4. 并发下载 (效率提升)
    paper_list = []
    # 设置最大5个并发下载线程
    with ThreadPoolExecutor(max_workers=5) as executor:
        # 提交任务
        futures = {
            # 我们使用 args.key_word (用于保存) 而不是 args.query (用于搜索)
            executor.submit(_download_and_create_paper, result, args.key_word): result 
            for result in papers_to_process
        }
        
        # 收集结果
        start_time = time.time()
        for i, future in enumerate(as_completed(futures), 1):
            paper = future.result()
            if paper: # 仅添加成功处理的论文
                paper_list.append(paper)
            logging.info(f"下载进度: {i}/{len(papers_to_process)}")
            
    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"成功下载并处理了 {len(paper_list)} 篇论文，耗时 {total_time:.2f} 秒。")
    return paper_list