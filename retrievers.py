import logging
import os
import sys
import requests
import time
import datetime 
import tenacity 
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
# (!!!) 导入已修复，不再从 chat_paper 导入 (!!!)
from paper_class import Paper 

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
# 策略三：Google Scholar 高引用论文检索 (爬虫)
# ---------------------------------------------------------------------
class GoogleScholarRetriever(RetrieverStrategy):
    """
    检索策略：使用 google_scholar_spider 模块从 Google Scholar 
    获取高引用论文，并尝试下载它们（如果链接指向 arXiv）。
    (!!!) 警告：此方法依赖网页爬虫，不稳定 (!!!)
   
    """
    
    def __init__(self):
        if not HAS_SCHOLAR:
            raise ImportError("Google Scholar 依赖 (pandas, requests, bs4) 未安装，或 'others/google_scholar_spider.py' 未找到。")
    
    def retrieve(self, args) -> List[Paper]:
        logging.info("使用 Google Scholar 高引用搜索模式 (爬虫)")
        
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
                # (!!!) 增强日志：按用户要求添加引用次数 (!!!)
                logging.warning(
                    f"【手动下载提示】(非ArXiv链接): {title}\n"
                    f"\t  URL: {url}\n"
                    f"\t  Citations: {citations}"
                )

        return paper_list

# ---------------------------------------------------------------------
# (!!!) 策略四：Semantic Scholar 高引用论文检索 (API) (!!!)
# ---------------------------------------------------------------------
class SemanticScholarRetriever(RetrieverStrategy):
    """
    检索策略：使用 Semantic Scholar 官方 API 
    获取高引用论文，并尝试下载开放存取(OA) PDF。
    (!!!) 推荐：此方法稳定且功能强大 (!!!)
    """
    
    API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    
    def _parse_date(self, pub_date_str: str) -> datetime.date | None:
        """健壮地解析 Semantic Scholar 返回的日期字符串"""
        if not pub_date_str:
            return None
        try:
            # 尝试解析完整日期, e.g., "2023-10-27"
            return datetime.date.fromisoformat(pub_date_str)
        except (ValueError, TypeError):
            try:
                # 尝试仅解析年份, e.g., "2023"
                return datetime.date(int(pub_date_str[:4]), 1, 1)
            except (ValueError, TypeError):
                logging.warning(f"无法解析日期字符串: {pub_date_str}")
                return None

    # (!!!) 新增：带重试的 API 调用 (!!!)
    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10), # 自动等待
        stop=tenacity.stop_after_attempt(5), # 最多重试5次
        retry=tenacity.retry_if_exception_type(requests.exceptions.HTTPError), # 仅在 HTTP 错误时重试
        reraise=True # 重试失败后，抛出原始异常
    )
    def _call_api(self, params: dict) -> requests.Response:
        """
        使用 tenacity 重试逻辑包装 API 请求。
        如果 API 返回 429 (Too Many Requests) 或 504 (Gateway Timeout)，
        此函数将自动等待并重试。
        """
        logging.info("正在调用 Semantic Scholar API...")
        response = requests.get(self.API_URL, params=params, timeout=30)
        response.raise_for_status() # 请求失败(如 429)则抛出异常，触发重试
        return response

    def retrieve(self, args) -> List[Paper]:
        logging.info("使用 Semantic Scholar 高引用搜索模式 (API)")
        
        # 1. 构建 API 请求参数
        params = {
            'query': args.query,
            'limit': args.max_results,
            'sort': args.sort or 'citationCount:desc', # 默认按引用数排序
            'fields': 'title,url,abstract,authors,publicationDate,citationCount,openAccessPdf'
        }
        
        logging.info(f"Semantic API 查询: query={params['query']}, limit={params['limit']}, sort={params['sort']}")
        
        paper_list = []
        try:
            # 2. 发起 API 请求 (!!!) (已更新为调用 _call_api) (!!!)
            response = self._call_api(params)
            
            data = response.json()
            
            if not data.get('data') or len(data['data']) == 0:
                logging.info("Semantic Scholar API 未返回结果。")
                return []
                
            # 3. 遍历结果并尝试下载
            for result in data['data']:
                if not result:
                    continue
                
                title = result.get('title')
                sem_url = result.get('url')
                abstract = result.get('abstract')
                citations = result.get('citationCount')
                pub_date_str = result.get('publicationDate')
                authors = [a.get('name') for a in result.get('authors', []) if a.get('name')]
                pub_date = self._parse_date(pub_date_str)
                
                pdf_url = None
                if result.get('openAccessPdf') and result['openAccessPdf'].get('url'):
                    pdf_url = result['openAccessPdf']['url']
                    
                if pdf_url:
                    try:
                        # 4. 复用下载器
                        filename = chat_arxiv.try_download_pdf(pdf_url, title, args.key_word)
                        
                        # 5. 创建 Paper 对象
                        paper = Paper(
                            path=filename,
                            title=title,
                            url=sem_url,
                            abs=abstract or "摘要不可用",
                            authers=authors,
                            published_date=pub_date,
                            citations=citations
                            # arxiv_id 默认为 None
                        )
                        paper_list.append(paper)
                        logging.info(f"成功下载 (Semantic Scholar): {title}")
                        
                    except Exception as e:
                        logging.warning(f"下载 {title} ({pdf_url}) 失败: {e}")
                else:
                    # 6. (!!!) 增强日志：按用户要求添加详细元数据 (!!!)
                    authors_str = ', '.join(authors) if authors else 'N/A'
                    date_str = pub_date.strftime('%Y-%m-%d') if pub_date else 'N/A'
                    logging.warning(
                        f"【手动下载提示】(无PDF): {title}\n"
                        f"\t  URL: {sem_url}\n"
                        f"\t  Citations: {citations} | Authors: {authors_str} | Date: {date_str}"
                    )

        except requests.exceptions.HTTPError as e:
            # (!!!) 捕获 429 错误 (!!!)
            if e.response.status_code == 429:
                logging.error(f"Semantic Scholar API 速率限制：重试 5 次后仍然失败。请等待几分钟后再试。")
            else:
                logging.error(f"Semantic Scholar API 请求失败: {e}", exc_info=True)
            return []
        except Exception as e:
            logging.error(f"处理 Semantic Scholar 结果时出错: {e}", exc_info=True)
            return []
            
        return paper_list