import fitz  # PyMuPDF
from collections import namedtuple

import arxiv
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
import google.generativeai as genai
import requests
import tenacity
from bs4 import BeautifulSoup
from PIL import Image
import chat_arxiv
import logging_config as logging_config  # configures logging (file + console) on import

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
        "use_arxiv",
            "batch_size",
            "batch_delay",
            "force",
    ],
)


class Paper:
    def __init__(self, path, title='', url='', abs='', authers=None):
        # 初始化函数，根据pdf路径初始化Paper对象                
        self.url = url           # 文章链接
        self.path = path          # pdf路径
        self.section_names = []   # 段落标题
        self.section_texts = {}   # 段落内容
        self.section_text_dict = {}  # 段落内容字典    
        self.abs = abs
        self.title_page = 0
        if title == '':
            self.pdf = fitz.open(self.path) # pdf文档
            self.title = self.get_title()
            self.parse_pdf()            
        else:
            self.title = title
            self.section_text_dict = {'Introduction': '', 'Abstract': self.abs}
        self.authers = authers or []        
        self.roman_num = ["I", "II", 'III', "IV", "V", "VI", "VII", "VIII", "IIX", "IX", "X"]
        self.digit_num = [str(d+1) for d in range(10)]
        self.first_image = ''
        
    def parse_pdf(self):
        try:
            self.pdf = fitz.open(self.path) # pdf文档
            self.text_list = [page.get_text() for page in self.pdf]
            self.all_text = ' '.join(self.text_list)
            self.section_page_dict = self._get_all_page_index() # 段落与页码的对应字典
            logging.debug("section_page_dict %s", self.section_page_dict)
            self.section_text_dict = self._get_all_page() # 段落与内容的对应字典
            if not self.section_text_dict.get('Abstract'):
                self.section_text_dict['Abstract'] = self.abs
            self.section_text_dict.update({"title": self.title})
            self.section_text_dict.update({"paper_info": self.get_paper_info()})
        finally:
            if hasattr(self, 'pdf'):
                self.pdf.close()         
        
    def get_paper_info(self):
        first_page_text = self.pdf[self.title_page].get_text()
        if "Abstract" in self.section_text_dict.keys():
            abstract_text = self.section_text_dict['Abstract']
        else:
            abstract_text = self.abs
        first_page_text = first_page_text.replace(abstract_text, "")
        return first_page_text
        
    def get_image_path(self, image_path='', max_images=3):
        """
        从PDF中提取并保存重要图片
        :param image_path: 图片保存路径
        :param max_images: 最大保存图片数量
        :return: 返回所有保存的图片路径和扩展名的列表 [(path, ext), ...]
        """
        if not os.path.exists(image_path):
            os.makedirs(image_path)

        saved_images = []
        image_info_list = []  # 存储图片信息：(image, size, ext, page_num)

        with fitz.Document(self.path) as pdf_file:
            # 遍历所有页面
            for page_number in range(len(pdf_file)):
                page = pdf_file[page_number]
                for img_idx, image in enumerate(page.get_images()):
                    try:
                        xref = image[0]
                        base_image = pdf_file.extract_image(xref)
                        image_bytes = base_image["image"]
                        ext = base_image["ext"]
                        
                        # 加载图片并计算大小
                        pil_image = Image.open(io.BytesIO(image_bytes))
                        # 转换RGBA为RGB
                        if pil_image.mode == 'RGBA':
                            pil_image = pil_image.convert('RGB')
                        image_size = pil_image.size[0] * pil_image.size[1]
                        
                        # 只保存大于特定大小的图片
                        if image_size > 5000:  # 过滤掉太小的图片
                            image_info_list.append((pil_image, image_size, ext, page_number))
                    except Exception as e:
                        logging.warning("处理页面 %s 的图片 %s 时出错：%s", page_number+1, img_idx+1, e)
                        continue

        # 按图片大小排序
        image_info_list.sort(key=lambda x: x[1], reverse=True)

        # 保存排序后的图片
        for i, (image, size, ext, page_num) in enumerate(image_info_list[:max_images]):
            try:
                # 调整图片大小，保持更好的质量
                max_pix = 1000  # 增加最大像素
                if image.size[0] > image.size[1]:
                    min_pix = int(image.size[1] * (max_pix/image.size[0]))
                    newsize = (max_pix, min_pix)
                else:
                    min_pix = int(image.size[0] * (max_pix/image.size[1]))
                    newsize = (min_pix, max_pix)
                
                resized_image = image.resize(newsize, Image.Resampling.LANCZOS)
                        
                        # 保存图片，包含页码信息
                image_name = f"figure_{i+1}_page{page_num+1}.{ext}"
                im_path = os.path.join(image_path, image_name)
                resized_image.save(im_path, quality=95)  # 使用更高的图片质量
                saved_images.append(im_path)  # 只保存路径
                logging.info("已保存图片 %s/%s：%s", i+1, max_images, im_path)
            except Exception as e:
                logging.warning("保存图片 %s 时出错：%s", i+1, e)
                continue

        # 返回所有保存的图片路径
        return saved_images if saved_images else []    # 定义一个函数，根据字体的大小，识别每个章节名称，并返回一个列表
    def get_chapter_names(self,):
        # # 打开一个pdf文件
        doc = fitz.open(self.path) # pdf文档        
        text_list = [page.get_text() for page in doc]
        all_text = ''
        for text in text_list:
            all_text += text
        # # 创建一个空列表，用于存储章节名称
        chapter_names = []
        for line in all_text.split('\n'):
            line_list = line.split(' ')
            if '.' in line:
                point_split_list = line.split('.')
                space_split_list = line.split(' ')
                if 1 < len(space_split_list) < 5:
                    if 1 < len(point_split_list) < 5 and (point_split_list[0] in self.roman_num or point_split_list[0] in self.digit_num):
                        logging.debug("line: %s", line)
                        chapter_names.append(line)      
                    # 这段代码可能会有新的bug，本意是为了消除"Introduction"的问题的！
                    elif 1 < len(point_split_list) < 5:
                        logging.debug("line: %s", line)
                        chapter_names.append(line)     
        
        return chapter_names
        
    def get_title(self):
        doc = self.pdf # 打开pdf文件
        max_font_size = 0 # 初始化最大字体大小为0
        max_string = "" # 初始化最大字体大小对应的字符串为空
        max_font_sizes = [0]
        for page_index, page in enumerate(doc): # 遍历每一页
            text = page.get_text("dict") # 获取页面上的文本信息
            blocks = text["blocks"] # 获取文本块列表
            for block in blocks: # 遍历每个文本块
                if block["type"] == 0 and len(block['lines']): # 如果是文字类型
                    if len(block["lines"][0]["spans"]):
                        font_size = block["lines"][0]["spans"][0]["size"] # 获取第一行第一段文字的字体大小            
                        max_font_sizes.append(font_size)
                        if font_size > max_font_size: # 如果字体大小大于当前最大值
                            max_font_size = font_size # 更新最大值
                            max_string = block["lines"][0]["spans"][0]["text"] # 更新最大值对应的字符串
        max_font_sizes.sort()
        logging.debug("max_font_sizes %s", max_font_sizes[-10:])
        cur_title = ''
        for page_index, page in enumerate(doc): # 遍历每一页
            text = page.get_text("dict") # 获取页面上的文本信息
            blocks = text["blocks"] # 获取文本块列表
            for block in blocks: # 遍历每个文本块
                if block["type"] == 0 and len(block['lines']): # 如果是文字类型
                    if len(block["lines"][0]["spans"]):
                        cur_string = block["lines"][0]["spans"][0]["text"] # 更新最大值对应的字符串
                        font_flags = block["lines"][0]["spans"][0]["flags"] # 获取第一行第一段文字的字体特征
                        font_size = block["lines"][0]["spans"][0]["size"] # 获取第一行第一段文字的字体大小                         
                        # print(font_size)
                        if abs(font_size - max_font_sizes[-1]) < 0.3 or abs(font_size - max_font_sizes[-2]) < 0.3:                        
                            # print("The string is bold.", max_string, "font_size:", font_size, "font_flags:", font_flags)                            
                            if len(cur_string) > 4 and "arXiv" not in cur_string:                            
                                # print("The string is bold.", max_string, "font_size:", font_size, "font_flags:", font_flags) 
                                if cur_title == ''    :
                                    cur_title += cur_string                       
                                else:
                                    cur_title += ' ' + cur_string     
                            self.title_page = page_index
                            # break
        title = cur_title.replace('\n', ' ')
        logging.debug("detected title: %s", title)
        return title


    def _get_all_page_index(self):
        # 定义需要寻找的章节名称列表
        section_list = ["Abstract", 
                        'Introduction', 'Related Work', 'Background', 
                        "Preliminary", "Problem Formulation",
                        'Methods', 'Methodology', "Method", 'Approach', 'Approaches',
                        # exp
                        "Materials and Methods", "Experiment Settings",
                        'Experiment',  "Experimental Results", "Evaluation", "Experiments",                        
                        "Results", 'Findings', 'Data Analysis',                                                                        
                        "Discussion", "Results and Discussion", "Conclusion",
                        'References']
        # 初始化一个字典来存储找到的章节和它们在文档中出现的页码
        section_page_dict = {}
        # 遍历每一页文档
        for page_index, page in enumerate(self.pdf):
            # 获取当前页面的文本内容
            cur_text = page.get_text()
            # 遍历需要寻找的章节名称列表
            for section_name in section_list:
                # 将章节名称转换成大写形式
                section_name_upper = section_name.upper()
                # 如果当前页面包含"Abstract"这个关键词
                if "Abstract" == section_name and section_name in cur_text:
                    # 将"Abstract"和它所在的页码加入字典中
                    section_page_dict[section_name] = page_index
                # 如果当前页面包含章节名称，则将章节名称和它所在的页码加入字典中
                else:
                    if section_name + '\n' in cur_text:
                        section_page_dict[section_name] = page_index
                    elif section_name_upper + '\n' in cur_text:
                        section_page_dict[section_name] = page_index
        # 返回所有找到的章节名称及它们在文档中出现的页码
        return section_page_dict

    def _get_all_page(self):
        """
        获取PDF文件中每个页面的文本信息，并将文本信息按照章节组织成字典返回。

        Returns:
            section_dict (dict): 每个章节的文本信息字典，key为章节名，value为章节文本。
        """
        text = ''
        text_list = []
        section_dict = {}
        
        # 再处理其他章节：
        text_list = [page.get_text() for page in self.pdf]
        for sec_index, sec_name in enumerate(self.section_page_dict):
            logging.debug("sec_index=%s sec_name=%s page=%s", sec_index, sec_name, self.section_page_dict[sec_name])
            if sec_index <= 0 and self.abs:
                continue
            else:
                # 直接考虑后面的内容：
                start_page = self.section_page_dict[sec_name]
                if sec_index < len(list(self.section_page_dict.keys()))-1:
                    end_page = self.section_page_dict[list(self.section_page_dict.keys())[sec_index+1]]
                else:
                    end_page = len(text_list)
                logging.debug("start_page=%s, end_page=%s", start_page, end_page)
                cur_sec_text = ''
                if end_page - start_page == 0:
                    if sec_index < len(list(self.section_page_dict.keys()))-1:
                        next_sec = list(self.section_page_dict.keys())[sec_index+1]
                        if text_list[start_page].find(sec_name) == -1:
                            start_i = text_list[start_page].find(sec_name.upper())
                        else:
                            start_i = text_list[start_page].find(sec_name)
                        if text_list[start_page].find(next_sec) == -1:
                            end_i = text_list[start_page].find(next_sec.upper())
                        else:
                            end_i = text_list[start_page].find(next_sec)                        
                        cur_sec_text += text_list[start_page][start_i:end_i]
                else:
                    for page_i in range(start_page, end_page):                    
#                         print("page_i:", page_i)
                        if page_i == start_page:
                            if text_list[start_page].find(sec_name) == -1:
                                start_i = text_list[start_page].find(sec_name.upper())
                            else:
                                start_i = text_list[start_page].find(sec_name)
                            cur_sec_text += text_list[page_i][start_i:]
                        elif page_i < end_page:
                            cur_sec_text += text_list[page_i]
                        elif page_i == end_page:
                            if sec_index < len(list(self.section_page_dict.keys()))-1:
                                next_sec = list(self.section_page_dict.keys())[sec_index+1]
                                if text_list[start_page].find(next_sec) == -1:
                                    end_i = text_list[start_page].find(next_sec.upper())
                                else:
                                    end_i = text_list[start_page].find(next_sec)  
                                cur_sec_text += text_list[page_i][:end_i]
                section_dict[sec_name] = cur_sec_text.replace('-\n', '').replace('\n', ' ')
        return section_dict


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
                    raise Exception("无法读取配置文件，请确保文件编码正确且内容有效") from e
                continue  # 尝试下一个编码

        # LLM client centralized initialization (moved to llm_client.py)
        try:
            from llm_client import make_client
            self.llm = make_client(self.config, args)
            if getattr(self.llm, 'enabled', False):
                logging.info("使用 LLM 模型: %s", self.llm.current_model_name())
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

    # arXiv 相关的功能已抽离到 chat_arxiv 模块。下面的方法作为兼容性包装，
    # 将调用 chat_arxiv 中的实现以避免重复代码。
    def get_url(self, keyword, page):
        return chat_arxiv.get_url(keyword, page)

    def get_titles(self, url, days=1):
        return chat_arxiv.get_titles(url, days)

    def get_all_titles_from_web(self, keyword, page_num=1, days=1):
        return chat_arxiv.get_all_titles_from_web(keyword, page_num=page_num, days=days)

    def get_arxiv_web(self, args, page_num=1, days=2):
        # chat_arxiv.get_arxiv_papers 已封装抓取与下载并返回 Paper 列表
        return chat_arxiv.get_arxiv_papers(args)

    def validateTitle(self, title):
        return chat_arxiv.validateTitle(title)

    def download_pdf(self, url, title):
        # 尽量使用 chat_arxiv 提供的带重试的下载器
        try:
            return chat_arxiv.try_download_pdf(url, title, getattr(self.args, 'query', ''))
        except Exception:
            # 作为降级，尝试直接调用内部下载函数（如果存在）
            try:
                return chat_arxiv._download_pdf_to_path(url, title, getattr(self.args, 'query', ''))
            except Exception as e:
                raise

    def try_download_pdf(self, url, title):
        return self.download_pdf(url, title)

    def get_local_papers(self, pdf_path):
        """
        从本地路径读取PDF文件
        :param pdf_path: PDF文件或文件夹的路径
        :return: Paper对象列表
        """
        paper_list = []
        if not os.path.exists(pdf_path):
            logging.error("路径 %s 不存在", pdf_path)
            return paper_list

        if os.path.isfile(pdf_path):
            # 处理单个PDF文件
            if pdf_path.lower().endswith('.pdf'):
                try:
                    paper = Paper(path=pdf_path)
                    paper_list.append(paper)
                    logging.info("成功加载PDF文件：%s", os.path.basename(pdf_path))
                except Exception as e:
                    logging.error("处理PDF文件 %s 时出错：%s", pdf_path, e)
        else:
            # 处理文件夹：解析目录下的所有 PDF（不再受 self.args.max_results 限制）
            for root, _, files in os.walk(pdf_path):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        pdf_file = os.path.join(root, file)
                        try:
                            paper = Paper(path=pdf_file)
                            paper_list.append(paper)
                            logging.info("成功加载PDF文件：%s", file)
                        except Exception as e:
                            logging.error("处理PDF文件 %s 时出错：%s", file, e)

            # 已收集文件夹中所有的 PDF，返回完整列表

        return paper_list

    # API调用限制相关变量
    _min_delay = 31  # 两次调用之间的最小延迟（秒）
    _api_calls = []  # 记录最近的API调用时间
    _call_window = 60  # 时间窗口（秒）
    _max_calls_per_window = 2  # 每个时间窗口内的最大调用次数
    _available_models = None  # 可用模型列表
    _current_model_index = 0  # 当前使用的模型索引
    _retry_count = 0  # 重试计数器
    
    def _wait_for_rate_limit(self):
        """等待API限流时间"""
        current_time = time.time()
        
        # 清理过期的调用记录
        self._api_calls = [t for t in self._api_calls if current_time - t < self._call_window]
        
        # 如果当前窗口内的调用次数达到限制
        if len(self._api_calls) >= self._max_calls_per_window:
            wait_time = self._api_calls[0] + self._call_window - current_time
            if wait_time > 0:
                logging.info("已达到API调用限制，等待 %.1f 秒...", wait_time)
                time.sleep(wait_time)
                # 递归调用以确保等待后仍然符合限制
                self._wait_for_rate_limit()
                return
        
        # 记录新的API调用时间
        self._api_calls.append(current_time)
    
    def _switch_model(self):
        """切换到下一个可用的模型"""
        if not self._available_models:
            try:
                logging.info("正在获取可用模型列表...")
                # 获取所有可用模型
                all_models = [
                    model.name for model in genai.list_models()
                    if 'generateContent' in getattr(model, 'supported_generation_methods', [])
                ]
                
                # 筛选2.5pro和2.5flash版本的模型
                self._available_models = [
                    model for model in all_models
                    if "gemini-2.5-pro" in model or "gemini-2.5-flash" in model
                ]
                
                # 按照优先级排序：flash > pro
                self._available_models.sort(key=lambda x: "flash" in x, reverse=True)
                if not self._available_models:
                    logging.warning("未找到2.5版本的模型，将使用所有可用模型")
                    self._available_models = all_models
                
                logging.info("可用的模型列表: %s", self._available_models)
            except Exception as e:
                logging.warning("获取模型列表失败: %s", e)
                return False
        
        # 尝试切换到下一个模型
        if self._available_models:
            self._current_model_index = (self._current_model_index + 1) % len(self._available_models)
            model_name = self._available_models[self._current_model_index]
            try:
                self.model = genai.GenerativeModel(model_name=model_name)
                logging.info("已切换到新模型: %s", model_name)
                return True
            except Exception as e:
                logging.warning("切换到模型 %s 失败: %s", model_name, e)
        return False

    def _call_gemini_api(self, prompt):
        """
        调用 Gemini API 的包装函数，包含重试逻辑、速率限制处理和模型自动切换
        """
        # Delegate LLM generation to centralized LLM client if available
        if hasattr(self, 'llm') and self.llm:
            return self.llm.generate(prompt)

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
            abs_path = os.path.abspath(paper.path)
            # 如果未设置 --force 并且文件在已处理缓存中，则跳过
            if not getattr(self.args, 'force', False) and abs_path in processed:
                logging.info("跳过已处理论文 %s：%s", paper.title, paper.path)
                continue
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
            full_summary += f"URL: {paper.url}\n\n"
            full_summary += f"作者: {', '.join(paper.authers)}\n\n"
            # 添加模型信息（从 LLM 客户端）
            if hasattr(self, 'llm') and self.llm:
                current_model = getattr(self.llm, 'model_name', 'Unknown')
            else:
                current_model = 'Unknown'
            full_summary += f"使用模型: {current_model}\n\n"

            full_summary += f"## 1. 核心思想总结\n{summary1}\n\n"
            full_summary += f"## 2. 方法详解\n{summary2}\n\n"
            full_summary += f"## 3. 最终评述与分析\n{summary3}\n\n"
            
            # 图片部分（将在最后作为附录添加）
            images_section = ""
            
            # 如果开启了保存图片功能，提取并保存图片
            if self.args.save_image:
                # 创建图片保存目录（不再在目录名中使用时间戳）
                image_dir = os.path.join(self.root_path, "export", f"images_{self.validateTitle(paper.title)}")
                if not os.path.exists(image_dir):
                    os.makedirs(image_dir)
                
                logging.info("正在提取论文图片...")
                saved_images = paper.get_image_path(image_path=image_dir, max_images=10)  # 提取更多图片
                
                if saved_images:  # 如果获取到了图片
                    # 获取输出文件名
                    output_file = self.get_output_filename(paper.title)
                    
                    # 准备图片部分（作为附录）
                    images_section = "\n---\n\n# 附录：论文图片\n\n"
                    for i, img_path in enumerate(saved_images, 1):
                        try:
                            if img_path and isinstance(img_path, str):  # 确保是有效的字符串路径
                                rel_path = os.path.relpath(img_path, os.path.dirname(output_file))
                                images_section += f"## 图 {i}\n![Figure {i}]({rel_path})\n\n"
                                logging.info("成功添加图片 %s：%s", i, img_path)
                        except Exception as e:
                            print(f"警告：处理图片 {i} 时出错：{e}")
                            continue
            
            # full_summary += f"## 论文分析\n{summary}\n\n"
            
            # # 将summary拆分为三个部分
            # summary_parts = summary.split('\n\n', 2) if summary else ['暂无总结', '暂无详解', '暂无评析']
            # if len(summary_parts) < 3:
            #     summary_parts.extend(['暂无详解', '暂无评析'][:3 - len(summary_parts)])
            
            # full_summary += f"## 1. 核心思想总结\n{summary_parts[0]}\n\n"
            # full_summary += f"## 2. 方法详解\n{summary_parts[1]}\n\n"
            # full_summary += f"## 3. 最终评析\n{summary_parts[2]}\n\n"
            
            # 在最后添加图片部分（如果有）
            if images_section:
                full_summary += images_section
            
            # 导出到文件：根据 --force 决定写入模式（强制模式覆盖，否则追加）
            output_file = self.get_output_filename(paper.title)
            
            # 如果输出文件为None，表示文件已存在且未启用force，跳过处理
            if output_file is None:
                logging.info("输出文件已存在，跳过论文 %s 的处理", paper.title)
                continue
            
            write_mode = 'w' if getattr(self.args, 'force', False) else 'a'
            self.export_to_markdown(full_summary, output_file, mode=write_mode)
            logging.info("论文《%s》的分析已保存到 %s", paper.title, output_file)
            # 标记为已处理并保存缓存
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
            # 批次计数与等待
            if batch_size and batch_size > 0:
                processed_in_batch += 1
                # 如果达到批次限制并且不是最后一个待处理论文，则等待
                # 注意：processed_in_batch counts已处理的论文（跳过的不计）
                if processed_in_batch >= batch_size:
                    processed_in_batch = 0
                    # 计算剩余待处理数量（跳过的不会计入）
                    remaining = 0
                    for p in paper_list[paper_index+1:]:
                        if os.path.abspath(p.path) not in processed:
                            remaining += 1
                    if remaining > 0:
                        logging.info("已处理 %s 篇，剩余 %s 篇，等待 %s 秒后继续处理...", batch_size, remaining, batch_delay)
                        time.sleep(batch_delay)

    def get_output_filename(self, paper_title=None):
        """获取输出文件的完整路径，如果文件已存在且未启用force，则返回None表示跳过"""
        export_path = os.path.join(self.root_path, "export")
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        
        # 如果提供了论文标题，使用论文标题作为文件名的一部分（不再包含时间戳）
        if paper_title:
            base = self.validateTitle(paper_title)
        else:
            base = self.validateTitle(self.args.query)

        filename = f"{base}.{self.args.file_format}"
        full = os.path.join(export_path, filename)

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
        logging.info("=== 运行配置 ===")
        if self.args.use_arxiv:
            logging.info("处理模式: arxiv在线搜索")
            logging.info("关键词: %s", self.key_word)
            logging.info("查询: %s", self.query)
            logging.info("排序: %s", self.sort)
            logging.info("最近天数: %s", self.args.days)
        else:
            logging.info("处理模式: 本地PDF文件")
            logging.info("PDF目录: %s", self.args.pdf_path)
        logging.info("最大处理数量: %s", self.args.max_results)
        logging.info("保存图片: %s", '是' if self.args.save_image else '否')
        logging.info("输出语言: %s", '中文' if self.args.language == 'zh' else '英文')
        logging.info("强制重新处理: %s", '是' if getattr(self.args, 'force', False) else '否')
        logging.info("%s", "="*20)


def chat_arxiv_main(args):
    reader1 = Reader(key_word=args.key_word,
                     query=args.query,
                     args=args)

    # 确保myPapers目录存在
    if not os.path.exists(args.pdf_path):
        try:
            os.makedirs(args.pdf_path)
            logging.info("已创建目录：%s", args.pdf_path)
        except Exception as e:
            logging.error("创建目录失败：%s", e)
            return

    reader1.show_info()

    # 根据参数选择处理模式
    if args.use_arxiv:
        logging.info("使用 arXiv 搜索模式（通过 chat_arxiv 模块获取）")
        try:
            import chat_arxiv
            paper_list = chat_arxiv.get_arxiv_papers(args)
        except Exception as e:
            logging.error("从 chat_arxiv 获取论文列表失败: %s", e)
            paper_list = []
    else:
        if not os.listdir(args.pdf_path):
            logging.info("提示：%s 目录为空", args.pdf_path)
            logging.info("请将要分析的PDF文件放入该目录，或使用 --use_arxiv 参数切换到arxiv搜索模式")
            return
            
        logging.info("从本地目录读取PDF文件：%s", args.pdf_path)
        paper_list = reader1.get_local_papers(args.pdf_path)
        if not paper_list:
            logging.info("在 %s 中未找到有效的PDF文件", args.pdf_path)
            logging.info("请确保文件具有.pdf扩展名，或使用 --use_arxiv 参数切换到arxiv搜索模式")
            return

    # 处理论文列表
    if paper_list:
        reader1.summary_with_chat(paper_list=paper_list)
    else:
        logging.info("没有找到要处理的论文，程序退出")


if __name__ == '__main__':
    # 设置默认的myPapers路径
    default_papers_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'myPapers')
    
    parser = argparse.ArgumentParser(description='ChatPaper: 论文分析工具')
    
    # 模式选择参数组
    mode_group = parser.add_argument_group('运行模式')
    mode_group.add_argument("--pdf_path", type=str, default=default_papers_dir, 
                      help="指定要分析的PDF文件或文件夹的路径。默认为myPapers目录")
    mode_group.add_argument("--use_arxiv", action='store_true', default=False,
                      help="是否使用arxiv搜索模式，默认为False（使用本地PDF模式）")
    
    # arxiv搜索参数组
    arxiv_group = parser.add_argument_group('arxiv搜索选项')
    arxiv_group.add_argument("--query", type=str, default='traffic flow prediction', 
                         help="arxiv搜索查询字符串，ti: xx, au: xx, all: xx")
    arxiv_group.add_argument("--key_word", type=str, default='GPT robot', 
                         help="用户研究领域的关键词")
    arxiv_group.add_argument("--page_num", type=int, default=25, 
                         help="arxiv搜索时的最大页数")
    arxiv_group.add_argument("--days", type=int, default=180, 
                         help="arxiv搜索时的最近天数限制")
    arxiv_group.add_argument("--sort", type=str, default="web", 
                         help="arxiv排序方式，可选 LastUpdatedDate")
    
    # 通用选项参数组
    general_group = parser.add_argument_group('通用选项')
    general_group.add_argument("--max_results", type=int, default=2, 
                          help="要处理的最大论文数量")
    general_group.add_argument("--save_image", action='store_true', default=True,
                          help="是否保存论文图片，默认开启。可能需要一两分钟的时间来保存图片")
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

    # 解析命令行参数
    args = parser.parse_args()
    
    # 转换为ArxivParams对象
    arxiv_args = ArxivParams(**vars(args))
    # reader = Reader(key_word='reinforcement learning', query='all: ChatGPT robot')
    # reader.show_info()
    # reader.get_arxiv()

    start_time = time.time()
    chat_arxiv_main(args=arxiv_args)
    logging.info("summary time: %.2f seconds", time.time() - start_time)
