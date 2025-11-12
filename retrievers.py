import logging
import os
import sys
import requests
import time
from typing import List

# --- 路径设置 ---
# 将 'others' 目录添加到 Python 路径中，以便导入 google_scholar_spider
#
current_dir = os.path.dirname(os.path.abspath(__file__))
others_dir = os.path.join(current_dir, 'others')
if others_dir not in sys.path:
    sys.path.append(others_dir)

# --- 导入策略接口和 Paper 类 ---
from retrieval_strategy import RetrieverStrategy
from chat_paper import Paper #

# --- 导入特定策略的辅助模块 ---
import chat_arxiv #
try:
    # 尝试导入 Google Scholar 爬虫所需组件
    #
    from google_scholar_spider import (
        GoogleScholarConfig, 
        create_main_url, 
        fetch_data, 
        process_data
    )
    import pandas as pd
    HAS_SCHOLAR = True
except ImportError as e:
    logging.warning(f"Google Scholar 检索器依赖未能加载: {e}。 'scholar' 策略将不可用。")
    HAS_SCHOLAR = False
except Exception as e:
    logging.error(f"加载 google_scholar_spider 时发生意外错误: {e}")
    HAS_SCHOLAR = False

# ---------------------------------------------------------------------
# 策略一：本地文件检索
# ---------------------------------------------------------------------
class LocalFileRetriever(RetrieverStrategy):
    """
    检索策略：从本地 'myPapers' 目录加载 PDF 文件。
    (此逻辑迁移自原 chat_paper.py)
   
    """
    
    def retrieve(self, args) -> List[Paper]:
        logging.info("从本地目录读取PDF文件：%s", args.pdf_path)
        paper_list = []
        
        # 这段逻辑直接从 chat_paper.py 的 get_local_papers 方法迁移而来
        #
        pdf_path = args.pdf_path
        if not os.path.exists(pdf_path):
            logging.error("路径 %s 不存在", pdf_path)
            return paper_list

        if os.path.isfile(pdf_path):
            if pdf_path.lower().endswith('.pdf'):
                try:
                    # 本地文件模式，citations 默认为 None
                    paper = Paper(path=pdf_path)
                    paper_list.append(paper)
                    logging.info("成功加载PDF文件：%s", os.path.basename(pdf_path))
                except Exception as e:
                    logging.error("处理PDF文件 %s 时出错：%s", pdf_path, e)
        else:
            max_results = getattr(args, 'max_results', 0)
            processed_count = 0
            for root, _, files in os.walk(pdf_path):
                if max_results > 0 and processed_count >= max_results:
                    break # 如果在外层循环检查，可以提前终止 walk
                for file in files:
                    if file.lower().endswith('.pdf'):
                        pdf_file = os.path.join(root, file)
                        try:
                            # 本地文件模式，citations 默认为 None
                            paper = Paper(path=pdf_file)
                            paper_list.append(paper)
                            processed_count += 1
                            logging.info("成功加载PDF文件：%s", file)
                            
                            if max_results > 0 and processed_count >= max_results:
                                logging.info("已达到最大处理数量限制 (%s)，停止加载更多PDF文件", max_results)
                                break # 退出内层循环
                                
                        except Exception as e:
                            logging.error("处理PDF文件 %s 时出错：%s", file, e)
        
        if not paper_list:
            logging.info("在 %s 中未找到有效的PDF文件", pdf_path)
            
        return paper_list

# ---------------------------------------------------------------------
# 策略二：arXiv 最新论文检索
# ---------------------------------------------------------------------
class ArxivRetriever(RetrieverStrategy):
    """
    检索策略：使用重构后的 chat_arxiv 模块 (API + 并发)
    从 arXiv 获取最新论文。
   
    """
    
    def retrieve(self, args) -> List[Paper]:
        logging.info("使用 arXiv 搜索模式 (API + 并发)")
        
        # chat_arxiv.get_arxiv_papers 内部创建 Paper 对象时
        # citations 默认为 None
        #
        paper_list = chat_arxiv.get_arxiv_papers(args)
        
        return paper_list

# ---------------------------------------------------------------------
# 策略三：Google Scholar 高引用论文检索
# ---------------------------------------------------------------------
class GoogleScholarRetriever(RetrieverStrategy):
    """
    检索策略：使用 google_scholar_spider 模块从 Google Scholar 
    获取高引用论文，并尝试下载它们（如果链接指向 arXiv）。
   
    """
    
    def __init__(self):
        if not HAS_SCHOLAR:
            raise ImportError("Google Scholar 依赖 (pandas, requests, bs4) 未安装，或 'others/google_scholar_spider.py' 未找到。")
    
    def retrieve(self, args) -> List[Paper]:
        logging.info("使用 Google Scholar 高引用搜索模式")
        
        # 1. 映射参数到 GoogleScholarConfig
        #
        config = GoogleScholarConfig(
            keyword=args.query,  # 复用 'query' 参数作为 Scholar 的关键词
            nresults=args.max_results,
            sortby=args.sort or "Citations", # 复用 'sort' 参数，默认为 Citations
            start_year=getattr(args, 'start_year', None),
            end_year=getattr(args, 'end_year', time.gmtime().tm_year),
            debug=False,
            csvpath="./export" 
        )
        logging.info(f"Scholar 配置: 关键词={config.keyword}, 数量={config.nresults}, 排序={config.sortby}, 年份={config.start_year}-{config.end_year}")

        # 2. 调用爬虫函数获取数据
        #
        gscholar_main_url = create_main_url(config)
        session = requests.Session()
        # fetch_data 需要一个 pbar 参数，我们传入 None
        data = fetch_data(config, session, gscholar_main_url, pbar=None) 
        
        if data.empty:
            logging.info("Google Scholar 未返回结果。")
            return []
            
        data_ranked = process_data(data, config.end_year, config.sortby)
        
        # 3. 尝试下载 PDF 并创建 Paper 对象
        paper_list = []
        
        for _, row in data_ranked.iterrows():
            if len(paper_list) >= args.max_results:
                break # 达到最大数量限制

            title = row['Title']
            url = row['Source']
            citations = row['Citations'] # (!!!) 获取引用次数
            
            # 关键步骤：目前只尝试下载明确指向 arXiv 的链接
            # 这是一个简化处理，因为 Google Scholar 的源链接格式各异
            if "arxiv.org/abs" in url:
                pdf_url = url.replace("/abs/", "/pdf/") + ".pdf"
                try:
                    # (!!!) 已修正：使用 args.key_word 而不是 args.query (!!!)
                    filename = chat_arxiv.try_download_pdf(pdf_url, title, args.key_word)
                    
                    # (!!!) 将引用次数传递给 Paper 对象 (!!!)
                    paper = Paper(
                        path=filename, 
                        title=title, 
                        url=url, 
                        # Scholar 不提供摘要，我们用引用数代替
                        abs=f"Cited by: {citations}. (摘要未从 Google Scholar 获取)",
                        authers=row['Author'].split(' and '), # 简单的作者分割
                        citations=citations  # <--- (!!!) 新增字段 (!!!)
                    )
                    paper_list.append(paper)
                    logging.info(f"成功下载 (Scholar-ArXiv): {title}")
                except Exception as e:
                    logging.warning(f"下载 {title} ({pdf_url}) 失败: {e}")
            else:
                # (!!!) 优化日志记录 (!!!)
                logging.warning(f"【手动下载提示】(非ArXiv链接): {title} @ {url}")

        return paper_list
