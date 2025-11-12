import fitz  # PyMuPDF
from collections import namedtuple

# import arxiv (不再需要)
import argparse
import configparser
import datetime
import io
import os
import re
import sys
import time
import json
import logging
# import google.generativeai as genai (已移入 llm_client_merged)
import requests
import tenacity
from bs4 import BeautifulSoup
from PIL import Image

# --- 核心模块导入 ---
# chat_arxiv 现在是 API 客户端
import chat_arxiv 
# configures logging (file + console) on import
import logging_config as logging_config 
from paper_enhancer import PaperEnhancer

# (!!!) 导入已修复，从 paper_class 导入 Paper (!!!)
from paper_class import Paper

# --- 新增的策略模式 import ---
from retrieval_strategy import RetrieverStrategy
try:
    from retrievers import (
        LocalFileRetriever, 
        ArxivRetriever, 
        GoogleScholarRetriever, 
        SemanticScholarRetriever, # <--- (!!!) 新增导入 (!!!)
        HAS_SCHOLAR
    )
except ImportError as e:
    logging.error(f"无法导入 retrievers.py: {e}。请确保该文件存在。")
    # 定义一个占位符以便程序能继续（虽然会报错）
    HAS_SCHOLAR = False
# ------------------------------


ArxivParams = namedtuple(
    "ArxivParams",
    [
        "query",
        "key_word",
        "page_num",
        "max_results",
        "days",
        "sort",
        "save_image",
        "file_format",
        "language",
        "pdf_path",
        # "use_arxiv", # 已废弃
        "retriever", # 新增
        "start_year", # 新增
        "end_year", # 新增
        "batch_size",
        "batch_delay",
        "force",
        "llm_client",
    ],
)


# (!!!) Paper 类已移至 paper_class.py (!!!)


# 定义Reader类
class Reader:
    # 初始化方法，设置属性
    def __init__(self, key_word, query,
                 root_path='./',
                 gitee_key='',
                 sort=None, 
                 user_name='defualt', args=None):
        self.user_name = user_name  # 读者姓名
        self.key_word = key_word  # 读者感兴趣的关键词
        self.query = query  # 读者输入的搜索查询
        self.sort = sort  # 读者选择的排序方式
        if args.language == 'en':
            self.language = 'English'
        elif args.language == 'zh':
            self.language = 'Chinese'
        else:
            self.language = 'Chinese'
        self.root_path = root_path
        self.args = args
        # 创建一个ConfigParser对象
        self.config = configparser.ConfigParser()
        
        # 定义可能的编码列表
        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'ascii']
        config_file = 'apikey.ini'
        
        # 尝试不同的编码读取文件
        for encoding in encodings:
            try:
                logging.info("尝试使用 %s 编码读取配置文件...", encoding)
                self.config.read(config_file, encoding=encoding)
                logging.info("成功使用 %s 编码读取配置文件", encoding)
                
                # 如果成功读取，尝试将文件转换为UTF-8编码
                try:
                    with open(config_file, 'r', encoding=encoding) as f:
                        content = f.read()
                    with open(config_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    logging.info("已将配置文件转换为 UTF-8 编码")
                except Exception as e:
                    print(f"转换文件编码时出错：{e}")
                
                break  # 如果成功读取，跳出循环
                
            except Exception as e:
                logging.warning("使用 %s 编码读取失败：%s", encoding, e)
                if encoding == encodings[-1]:  # 如果是最后一个编码
                    logging.error("无法读取配置文件，请确保文件编码正确且内容有效")
                continue  # 尝试下一个编码

        # LLM client centralized initialization (using merged multi-LLM client)
        try:
            from llm_client_merged import make_client
            self.llm = make_client(self.config, args)
            
            # 如果指定了手动客户端，尝试切换
            if hasattr(args, 'llm_client') and args.llm_client:
                client_name = args.llm_client
                if self.llm.switch_client(client_name):
                    logging.info("已手动切换到 LLM 客户端: %s", client_name)
                else:
                    logging.warning("无法切换到指定的客户端 %s，将使用默认客户端", client_name)
                    logging.info("可用客户端: %s", self.llm.get_available_clients())
            
            if self.llm.current_client:
                logging.info("使用 LLM 模型: %s", self.llm.current_model_name())
                logging.info("可用客户端: %s", self.llm.get_available_clients())
            else:
                logging.info("LLM 未初始化或不可用，后续生成将返回备用消息")
        except Exception as e:
            logging.warning("初始化 LLM 客户端失败: %s", e)
            self.llm = None

        self.file_format = args.file_format
        if args.save_image:
            try:
                self.gitee_key = self.config.get('Gitee', 'api')
            except Exception:
                self.gitee_key = ''
        else:
            self.gitee_key = ''
        
        # 初始化PaperEnhancer
        self.paper_enhancer = PaperEnhancer()

    # (get_local_papers 方法已移至 retrievers.py 的 LocalFileRetriever)
    
    def _call_gemini_api(self, prompt):
        """
        调用 LLM API 的包装函数 (委托给 llm_client_merged)
        """
        # Delegate LLM generation to centralized LLM client if available
        if hasattr(self, 'llm') and self.llm:
            try:
                result = self.llm.generate(prompt)
                # 检查返回结果是否为错误消息
                if result and "抱歉" in result and ("未初始化" in result or "失败" in result):
                    logging.error("LLM客户端调用失败：%s", result)
                return result
            except Exception as e:
                logging.error("LLM客户端调用异常：%s", str(e))
                return f"抱歉，LLM客户端调用失败：{str(e)}"

        logging.error("LLM 客户端未初始化，将使用备用响应。")
        return "抱歉，由于 API 初始化问题，我暂时无法生成响应。请检查您的 API 密钥和模型可用性。"

    def summary_with_chat(self, paper_list):
        
        # 加载已处理文件缓存，跳过已处理的论文
        export_dir = os.path.join(self.root_path, "export")
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        processed_file = os.path.join(export_dir, 'processed.json')
        try:
            with open(processed_file, 'r', encoding='utf-8') as pf:
                processed = json.load(pf)
        except Exception:
            processed = {}

        # 批次配置
        batch_size = getattr(self.args, 'batch_size', 0)
        batch_delay = getattr(self.args, 'batch_delay', 60)
        if batch_size and batch_size > 0:
            logging.info("批次处理已启用：每 %s 篇论文休眠 %s 秒", batch_size, batch_delay)

        # 遍历论文列表
        processed_in_batch = 0
        total_to_process = len(paper_list)
        for paper_index, paper in enumerate(paper_list):
            
            # 确保 paper 对象有效
            if not paper or not paper.title:
                logging.warning(f"跳过无效的 paper 对象 (索引 {paper_index})")
                continue
            
            if not paper.path:
                logging.warning(f"跳过缺少路径的 paper 对象: {paper.title}")
                continue
                
            abs_path = os.path.abspath(paper.path)
            
            # 第一步：检查是否应该跳过处理（在开始任何处理之前）
            output_file = self.get_output_filename(paper.title)
            skip_conditions = [
                # 文件已存在且未启用force
                output_file is None,
                # 文件在已处理缓存中且未启用force
                not getattr(self.args, 'force', False) and abs_path in processed
            ]
            
            if any(skip_conditions):
                logging.info("跳过已处理论文 %s：%s", paper.title, paper.path)
                continue
            
            # 确保在总结前，本地文件的 PDF 已被解析
            if not paper.section_text_dict.get('Abstract'):
                logging.info(f"论文 {paper.title} 缺少内容，尝试重新解析...")
                # 重新打开
                try:
                    paper.pdf = fitz.open(paper.path)
                    paper.parse_pdf() # 确保本地文件被解析
                    paper.pdf.close()
                except Exception as e:
                    logging.error(f"总结前重新解析失败 {paper.path}: {e}")
                    paper.section_text_dict['Abstract'] = "PDF 解析失败"

                
            logging.info("正在总结论文 %s/%s: %s", paper_index + 1, len(paper_list), paper.title)
            
            # 收集论文所有部分
            method_content = (paper.section_text_dict.get('Method', '') or 
                          paper.section_text_dict.get('Methods', '') or 
                          paper.section_text_dict.get('Methodology', ''))[:3000]
            conclusion_content = paper.section_text_dict.get('Conclusion', '')[:1500]
            
            # 三轮逐步摘要：借鉴 chat_arxiv.py 的流程，提高分段结构和可解释性
            # 1) 第一轮：基于标题、摘要与引言，给出核心思想（背景/问题/方法/贡献）
            prompt1 = f"""
作为学术论文分析专家，请提供一份简洁的第一轮总结（使用{self.language}）：
将结果按四个部分组织：Background, Problem, Method (high-level), Contribution。

标题: {paper.title}
摘要: {paper.abs}
引言: {paper.section_text_dict.get('Introduction', '')[:3500]}
"""
            summary1 = self._call_gemini_api(prompt1)

            # 2) 第二轮：基于第一轮总结和方法章节，展开方法细节和关键创新
            method_content = method_content[:5000]
            prompt2 = f"""
基于以下初步总结和论文的方法章节，请详细说明该论文的方法细节：
重点描述关键创新、算法/架构细节、关键步骤与整体流程。请使用{self.language}。

初步总结:
{summary1}

方法节内容:
{method_content}
"""
            summary2 = self._call_gemini_api(prompt2)

            # 3) 第三轮：结合前两轮与结论，给出最终评估（总体总结/优点/缺点/应用）
            conclusion_content = conclusion_content[:2500]
            prompt3 = f"""
结合前两轮返回的信息与论文结论部分，请给出最终的综合评估（使用{self.language}），包含：
1) Overall Summary
2) Strengths
3) Weaknesses / Limitations
4) Potential Applications / Implications

初步总结:
{summary1}

方法详述:
{summary2}

结论节:
{conclusion_content}
"""
            summary3 = self._call_gemini_api(prompt3)

            # 整合三轮总结
            full_summary = f"# {paper.title}\n\n"

            # --- (!!!) START: 更新元数据显示 (解决用户痛点) (!!!) ---
            if paper.arxiv_id:
                full_summary += f"**ArXiv ID**: {paper.arxiv_id}\n"
            if paper.url:
                # 优先使用 arXiv 页面链接
                url_to_show = paper.url
                if paper.arxiv_id and "arxiv.org/abs" not in url_to_show:
                    url_to_show = f"http://arxiv.org/abs/{paper.arxiv_id}"
                full_summary += f"**URL**: {url_to_show}\n"
            if paper.published_date:
                full_summary += f"**提交日期**: {paper.published_date.strftime('%Y-%m-%d')}\n"
            if paper.authers:
                # 格式化作者列表
                full_summary += f"**作者**: {'; '.join(paper.authers)}\n"
            
            # --- (!!!) START: Add Citation Count (User Request) (!!!) ---
            if paper.citations is not None:
                # Google Scholar mode: Show the number
                full_summary += f"**引用次数**: {paper.citations}\n"
            else:
                # ArXiv or Local mode: Show NULL (遵照用户要求)
                full_summary += f"**引用次数**: NULL\n"
            # --- (!!!) END: Add Citation Count (!!!) ---
            
            # --- (!!!) END: 更新元数据显示 (!!!) ---
            
            # 添加模型信息（从 LLM 客户端）
            if hasattr(self, 'llm') and self.llm:
                current_model = self.llm.current_model_name()
            else:
                current_model = 'Unknown'
            full_summary += f"使用模型: {current_model}\n\n"

            full_summary += f"## 1. 核心思想总结\n{summary1}\n\n"
            full_summary += f"## 2. 方法详解\n{summary2}\n\n"
            full_summary += f"## 3. 最终评述与分析\n{summary3}\n\n"
            
            # (后续图片处理部分不变)
            # ...
            images_section = ""
            
            if self.args.save_image:
                keyword_dir = os.path.join(self.root_path, "export", chat_arxiv.validateTitle(self.key_word))
                images_dir = os.path.join(keyword_dir, "images")
                if not os.path.exists(images_dir):
                    os.makedirs(images_dir)
                
                logging.info("正在提取论文图片到目录: %s", images_dir)
                saved_images = paper.get_image_path(image_path=images_dir, max_images=10)
                
                if saved_images:
                    images_section = "\n---\n\n# 附录：论文图片\n\n"
                    for i, img_path in enumerate(saved_images, 1):
                        try:
                            if img_path and isinstance(img_path, str):
                                filename = os.path.basename(img_path)
                                rel_path = f"images/{filename}"
                                images_section += f"## 图 {i}\n![Figure {i}]({rel_path})\n\n"
                                logging.info("成功添加图片 %s：%s", i, img_path)
                        except Exception as e:
                            print(f"警告：处理图片 {i} 时出错：{e}")
                            continue
            
            if images_section:
                full_summary += images_section
            
            if self.args.save_image and saved_images:
                try:
                    full_summary = self.paper_enhancer.update_image_links(full_summary, paper.title, self.key_word)
                    logging.info("已更新图片链接")
                except Exception as e:
                    logging.warning("更新图片链接失败: %s", e)
            
            output_file = self.get_output_filename(paper.title)
            
            if output_file is None:
                logging.info("输出文件已存在，跳过论文 %s 的处理", paper.title)
                continue
            
            write_mode = 'w' # 默认覆盖
            self.export_to_markdown(full_summary, output_file, mode=write_mode)
            logging.info("论文《%s》的分析已保存到 %s", paper.title, output_file)

            try:
                processed[abs_path] = {
                    'title': paper.title,
                    'output': os.path.abspath(output_file),
                    'time': str(datetime.datetime.now())
                }
                with open(processed_file, 'w', encoding='utf-8') as pf:
                    json.dump(processed, pf, ensure_ascii=False, indent=2)
            except Exception as e:
                logging.warning("保存已处理缓存时出错：%s", e)

            if batch_size and batch_size > 0:
                processed_in_batch += 1
                if processed_in_batch >= batch_size:
                    processed_in_batch = 0
                    remaining = 0
                    for p in paper_list[paper_index+1:]:
                        if p and p.path and os.path.abspath(p.path) not in processed:
                            remaining += 1
                    if remaining > 0:
                        logging.info("已处理 %s 篇，剩余 %s 篇，等待 %s 秒后继续处理...", batch_size, remaining, batch_delay)
                        time.sleep(batch_delay)
        
        # (生成 Excel 部分不变)
        if paper_list:
            try:
                papers_data = []
                for paper in paper_list:
                    if not paper: continue # 跳过空 paper
                    paper_data = {
                        "title": paper.title,
                        "url": paper.url,
                        "authors": paper.authers,
                        "keyword": self.key_word,
                        "published_date": paper.published_date.strftime("%Y-%m-%d") if paper.published_date else "",
                        "citation_count": paper.citations if paper.citations is not None else 0, # (!!!) 更新 Excel
                        "arxiv_id": paper.arxiv_id,
                        "categories": [],
                        "processed_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    papers_data.append(paper_data)
                
                # 生成汇总Excel表格
                excel_path = self.paper_enhancer.generate_summary_excel(papers_data, self.key_word)
                logging.info("已生成汇总Excel表格: %s", excel_path)
                
            except Exception as e:
                logging.warning("生成汇总Excel表格失败: %s", e)

    def get_output_filename(self, paper_title=None):
        """
        (!!!) 修正此方法以使用 chat_arxiv.validateTitle (!!!)
        获取输出文件的完整路径，如果文件已存在且未启用force，则返回None表示跳过
        """
        # 按照关键词分类存储
        # (!!!) 修正：调用 chat_arxiv.validateTitle (!!!)
        keyword_dir = os.path.join(self.root_path, "export", chat_arxiv.validateTitle(self.key_word))
        if not os.path.exists(keyword_dir):
            os.makedirs(keyword_dir)
        
        # 如果提供了论文标题，使用论文标题作为文件名的一部分（不再包含时间戳）
        if paper_title:
            # (!!!) 修正：调用 chat_arxiv.validateTitle (!!!)
            base = chat_arxiv.validateTitle(paper_title)
        else:
            # (!!!) 修正：调用 chat_arxiv.validateTitle (!!!)
            base = chat_arxiv.validateTitle(self.args.query)

        filename = f"{base}.{self.args.file_format}"
        full = os.path.join(keyword_dir, filename)

        # 如果文件存在且未启用 --force，则返回None表示跳过处理
        if os.path.exists(full) and not getattr(self.args, 'force', False):
            return None

        return full

    def export_to_markdown(self, text, file_name, mode='w'):
        """
        导出文本到markdown文件
        :param text: 要写入的文本内容
        :param file_name: 输出文件名
        :param mode: 写入模式，'w'覆盖，'a'追加
        """
        # 如果是强制模式且文件已存在，先删除旧文件
        if getattr(self.args, 'force', False) and os.path.exists(file_name):
            try:
                os.remove(file_name)
                logging.info("已删除旧文件：%s", file_name)
            except Exception as e:
                logging.warning("删除旧文件时出错：%s", e)
        
        # 打开文件写入内容
        with open(file_name, mode, encoding="utf-8") as f:
            f.write(text)

    # 定义一个方法，打印出读者信息
    def show_info(self):
        """ (!!!) 替换此方法 (!!!) """
        logging.info("=== 运行配置 ===")
        # 根据新的 --retriever 参数显示模式
        if self.args.retriever == 'arxiv':
            logging.info("处理模式: arXiv 最新搜索 (API)")
            logging.info("查询: %s", self.query)
            logging.info("关键词 (用于保存): %s", self.key_word)
            logging.info("排序: %s", self.sort or "SubmittedDate") # API 默认
            logging.info("最近天数: %s", self.args.days)
        elif self.args.retriever == 'scholar':
            logging.info("处理模式: Google Scholar 高引用搜索 (爬虫)")
            logging.info("查询 (关键词): %s", self.query)
            logging.info("关键词 (用于保存): %s", self.key_word)
            logging.info("排序: %s", self.sort or "Citations") # Scholar 排序
            logging.info("年份范围: %s - %s", getattr(self.args, 'start_year', 'None'), getattr(self.args, 'end_year', 'Default'))
        elif self.args.retriever == 'semantic':
            logging.info("处理模式: Semantic Scholar 高引用搜索 (API)")
            logging.info("查询 (关键词): %s", self.query)
            logging.info("关键词 (用于保存): %s", self.key_word)
            logging.info("排序: %s", self.sort or "citationCount:desc") # API 默认
        else: # 默认 'local'
            logging.info("处理模式: 本地PDF文件")
            logging.info("PDF目录: %s", self.args.pdf_path)

        logging.info("最大处理数量: %s", self.args.max_results)
        logging.info("保存图片: %s", '是' if self.args.save_image else '否')
        logging.info("输出语言: %s", '中文' if self.args.language == 'zh' else '英文')
        logging.info("强制重新处理: %s", '是' if getattr(self.args, 'force', False) else '否')
        logging.info("LLM 客户端: %s", getattr(self.args, 'llm_client', '自动'))
        logging.info("%s", "="*20)


def chat_arxiv_main(args):
    """ (!!!) 替换此方法 (!!!) """
    reader1 = Reader(key_word=args.key_word,
                     query=args.query,
                     args=args)
    reader1.show_info() # 显示更新后的配置

    # --- 策略模式实现 ---
    # 1. 定义策略映射
    retrievers = {
        "local": LocalFileRetriever(),
        "arxiv": ArxivRetriever(),
        "semantic": SemanticScholarRetriever() # <--- (!!!) 新增策略 (!!!)
    }
    # 仅当 Scholar 依赖成功加载时才添加它
    if HAS_SCHOLAR:
        retrievers["scholar"] = GoogleScholarRetriever()
    
    # 2. 根据 args.retriever 选择策略
    retriever = retrievers.get(args.retriever)
    
    if not retriever:
        logging.error(f"未知的检索策略: {args.retriever}")
        if args.retriever == 'scholar' and not HAS_SCHOLAR:
            logging.error("Google Scholar 策略不可用，请检查依赖 (pandas, bs4, requests)。")
        return
        
    logging.info(f"正在使用检索策略: {args.retriever}")
    
    # --- 特定策略的预检查 (例如本地目录) ---
    if args.retriever == 'local':
        if not os.path.exists(args.pdf_path):
            try:
                os.makedirs(args.pdf_path)
                logging.info("已创建目录：%s", args.pdf_path)
            except Exception as e:
                logging.error("创建目录失败：%s", e)
                return
        if not os.listdir(args.pdf_path) and not os.path.isfile(args.pdf_path):
            logging.info("提示：%s 目录为空或路径非文件。", args.pdf_path)
            logging.info("请将PDF放入该目录，或使用 --retriever 'arxiv'/'scholar'/'semantic' 切换模式。")
            return
            
    # 3. 执行检索
    try:
        paper_list = retriever.retrieve(args)
    except Exception as e:
        logging.error(f"使用策略 {args.retriever} 检索论文时失败: {e}", exc_info=True)
        paper_list = []

    # 4. 后续处理 (保持不变)
    if paper_list:
        logging.info(f"检索到 {len(paper_list)} 篇论文，开始总结...")
        reader1.summary_with_chat(paper_list=paper_list)
    else:
        logging.info("没有找到要处理的论文，程序退出")


if __name__ == '__main__':
    """ (!!!) 替换此方法 (!!!) """
    # 设置默认的myPapers路径
    default_papers_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'myPapers')
    
    parser = argparse.ArgumentParser(description='ChatPaper: 论文分析工具')
    
    # 模式选择参数组
    mode_group = parser.add_argument_group('运行模式')
    mode_group.add_argument("--pdf_path", type=str, default=default_papers_dir, 
                      help="指定要分析的PDF文件或文件夹的路径 (仅在 --retriever='local' 时生效)")
    mode_group.add_argument("--retriever", type=str, default="local", 
                      choices=["local", "arxiv", "scholar", "semantic"], # <--- (!!!) 新增选项 (!!!)
                      help="选择检索策略: 'local'(本地), 'arxiv'(arXiv最新), 'scholar'(Google Scholar高引用-爬虫), 'semantic'(Semantic Scholar高引用-API)")
    
    # 检索参数组 (arXiv 和 Scholar 共用)
    search_group = parser.add_argument_group('检索选项 (arXiv / Scholar / Semantic)')
    search_group.add_argument("--query", type=str, default='large language model', 
                         help="搜索查询字符串 (arXiv格式 或 Google/Semantic Scholar 关键词)")
    search_group.add_argument("--key_word", type=str, default='LLM', 
                         help="用于分类和保存文件的关键词 (例如 'LLM')")
    search_group.add_argument("--page_num", type=int, default=1, 
                         help="arXiv搜索时的页数 (每页50条, 仅 'arxiv' 模式)")
    search_group.add_argument("--days", type=int, default=30, 
                         help="arXiv搜索时的最近天数限制 (仅 'arxiv' 模式)")
    search_group.add_argument("--sort", type=str, default=None, 
                         help="排序方式: arXiv (例如 'LastUpdatedDate') 或 Scholar (例如 'Citations') 或 Semantic (例如 'citationCount:desc')")
    search_group.add_argument('--start_year', type=int, default=None, 
                         help="Google Scholar 搜索的开始年份 (仅 'scholar' 模式)")
    search_group.add_argument('--end_year', type=int, default=datetime.datetime.now().year, 
                         help="Google Scholar 搜索的结束年份 (仅 'scholar' 模式)")
    
    # 通用选项参数组
    general_group = parser.add_argument_group('通用选项')
    general_group.add_argument("--max_results", type=int, default=2, 
                          help="要处理的最大论文数量")
    general_group.add_argument("--save_image", action='store_true', default=False,
                          help="是否保存论文图片。默认关闭 (False)")
    general_group.add_argument("--file_format", type=str, default='md', 
                          help="导出的文件格式：md（推荐，支持图片）或txt")
    general_group.add_argument("--language", type=str, default='zh', 
                          help="输出语言：zh（中文）或en（英文）")
    # 批次处理选项：batch_size=0 表示不分批，直接处理所有
    general_group.add_argument("--batch-size", type=int, default=0,
                          help="每个批次处理的论文数量，0 表示不分批（默认）")
    general_group.add_argument("--batch-delay", type=int, default=60,
                          help="每个批次之间的等待时间（秒），仅在 --batch-size>0 时生效，默认60秒")
    # 强制重新处理，忽略 export/processed.json 缓存
    general_group.add_argument("--force", action='store_true', default=False,
                          help="启用后忽略已处理缓存，重新处理所有论文（默认: False）")
    # LLM客户端选择
    general_group.add_argument("--llm-client", type=str, default=None,
                          help="手动指定LLM客户端：Gemini, DeepSeek, Kimi, Qwen, Doubao。默认自动选择")

    # 解析命令行参数
    args = parser.parse_args()
    
    # 将 args 命名空间转换为字典，以便与 ArxivParams 兼容
    args_dict = vars(args)
    
    # 过滤掉不在 ArxivParams._fields 中的键
    filtered_args_dict = {k: v for k, v in args_dict.items() if k in ArxivParams._fields}

    # 转换为ArxivParams对象
    arxiv_args = ArxivParams(**filtered_args_dict)

    start_time = time.time()
    chat_arxiv_main(args=arxiv_args)
    logging.info("总运行时间: %.2f seconds", time.time() - start_time)