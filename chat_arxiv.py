import fitz # PyMuPDF
from collections import namedtuple

# import arxiv
import argparse
import configparser
import datetime
import io
import os
import re
import sys
import time
# import openai
# import tiktoken
import google.generativeai as genai
import requests
import tenacity
from bs4 import BeautifulSoup
from PIL import Image

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
    ],
)


class Paper:
    def __init__(self, path, title='', url='', abs='', authers=[]):
        # 初始化函数，根据pdf路径初始化Paper对象                
        self.url = url  # 文章链接
        self.path = path  # pdf路径
        self.section_names = []  # 段落标题
        self.section_texts = {}  # 段落内容
        self.abs = abs
        self.title_page = 0
        self.title = title
        self.pdf = fitz.open(self.path)  # pdf文档
        self.parse_pdf()
        self.authers = authers
        self.roman_num = ["I", "II", 'III', "IV", "V", "VI", "VII", "VIII", "IIX", "IX", "X"]
        self.digit_num = [str(d + 1) for d in range(10)]
        self.first_image = ''

    def parse_pdf(self):
        self.pdf = fitz.open(self.path)  # pdf文档
        self.text_list = [page.get_text() for page in self.pdf]
        self.all_text = ' '.join(self.text_list)
        self.section_page_dict = self._get_all_page_index()  # 段落与页码的对应字典
        print("section_page_dict", self.section_page_dict)
        self.section_text_dict = self._get_all_page()  # 段落与内容的对应字典
        self.section_text_dict.update({"title": self.title})
        self.section_text_dict.update({"paper_info": self.get_paper_info()})
        self.pdf.close()

    def get_paper_info(self):
        first_page_text = self.pdf[self.title_page].get_text()
        if "Abstract" in self.section_text_dict.keys():
            abstract_text = self.section_text_dict['Abstract']
        else:
            abstract_text = self.abs
        first_page_text = first_page_text.replace(abstract_text, "")
        return first_page_text

    def get_image_path(self, image_path=''):
        """
        将PDF中的第一张图保存到image.png里面，存到本地目录，返回文件名称，供gitee读取
        :param filename: 图片所在路径，"C:\\Users\\Administrator\\Desktop\\nwd.pdf"
        :param image_path: 图片提取后的保存路径
        :return:
        """
        # open file
        max_size = 0
        image_list = []
        with fitz.Document(self.path) as my_pdf_file:
            # 遍历所有页面
            for page_number in range(1, len(my_pdf_file) + 1):
                # 查看独立页面
                page = my_pdf_file[page_number - 1]
                # 查看当前页所有图片
                images = page.get_images()
                # 遍历当前页面所有图片
                for image_number, image in enumerate(page.get_images(), start=1):
                    # 访问图片xref
                    xref_value = image[0]
                    # 提取图片信息
                    base_image = my_pdf_file.extract_image(xref_value)
                    # 访问图片
                    image_bytes = base_image["image"]
                    # 获取图片扩展名
                    ext = base_image["ext"]
                    # 加载图片
                    image = Image.open(io.BytesIO(image_bytes))
                    image_size = image.size[0] * image.size[1]
                    if image_size > max_size:
                        max_size = image_size
                    image_list.append(image)
        for image in image_list:
            image_size = image.size[0] * image.size[1]
            if image_size == max_size:
                image_name = f"image.{ext}"
                im_path = os.path.join(image_path, image_name)
                print("im_path:", im_path)

                max_pix = 480
                origin_min_pix = min(image.size[0], image.size[1])

                if image.size[0] > image.size[1]:
                    min_pix = int(image.size[1] * (max_pix / image.size[0]))
                    newsize = (max_pix, min_pix)
                else:
                    min_pix = int(image.size[0] * (max_pix / image.size[1]))
                    newsize = (min_pix, max_pix)
                image = image.resize(newsize)

                image.save(open(im_path, "wb"))
                return im_path, ext
        return None, None

    # 定义一个函数，根据字体的大小，识别每个章节名称，并返回一个列表
    def get_chapter_names(self, ):
        # # 打开一个pdf文件
        doc = fitz.open(self.path)  # pdf文档
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
                    if 1 < len(point_split_list) < 5 and (
                            point_split_list[0] in self.roman_num or point_split_list[0] in self.digit_num):
                        print("line:", line)
                        chapter_names.append(line)
                    # 这段代码可能会有新的bug，本意是为了消除"Introduction"的问题的！
                    elif 1 < len(point_split_list) < 5:
                        print("line:", line)
                        chapter_names.append(line)

        return chapter_names

    def get_title(self):
        doc = self.pdf  # 打开pdf文件
        max_font_size = 0  # 初始化最大字体大小为0
        max_string = ""  # 初始化最大字体大小对应的字符串为空
        max_font_sizes = [0]
        for page_index, page in enumerate(doc):  # 遍历每一页
            text = page.get_text("dict")  # 获取页面上的文本信息
            blocks = text["blocks"]  # 获取文本块列表
            for block in blocks:  # 遍历每个文本块
                if block["type"] == 0 and len(block['lines']):  # 如果是文字类型
                    if len(block["lines"][0]["spans"]):
                        font_size = block["lines"][0]["spans"][0]["size"]  # 获取第一行第一段文字的字体大小
                        max_font_sizes.append(font_size)
                        if font_size > max_font_size:  # 如果字体大小大于当前最大值
                            max_font_size = font_size  # 更新最大值
                            max_string = block["lines"][0]["spans"][0]["text"]  # 更新最大值对应的字符串
        max_font_sizes.sort()
        print("max_font_sizes", max_font_sizes[-10:])
        cur_title = ''
        for page_index, page in enumerate(doc):  # 遍历每一页
            text = page.get_text("dict")  # 获取页面上的文本信息
            blocks = text["blocks"]  # 获取文本块列表
            for block in blocks:  # 遍历每个文本块
                if block["type"] == 0 and len(block['lines']):  # 如果是文字类型
                    if len(block["lines"][0]["spans"]):
                        cur_string = block["lines"][0]["spans"][0]["text"]  # 更新最大值对应的字符串
                        font_flags = block["lines"][0]["spans"][0]["flags"]  # 获取第一行第一段文字的字体特征
                        font_size = block["lines"][0]["spans"][0]["size"]  # 获取第一行第一段文字的字体大小
                        # print(font_size)
                        if abs(font_size - max_font_sizes[-1]) < 0.3 or abs(font_size - max_font_sizes[-2]) < 0.3:
                            # print("The string is bold.", max_string, "font_size:", font_size, "font_flags:", font_flags)                            
                            if len(cur_string) > 4 and "arXiv" not in cur_string:
                                # print("The string is bold.", max_string, "font_size:", font_size, "font_flags:", font_flags) 
                                if cur_title == '':
                                    cur_title += cur_string
                                else:
                                    cur_title += ' ' + cur_string
                            self.title_page = page_index
                            # break
        title = cur_title.replace('\n', ' ')
        return title

    def _get_all_page_index(self):
        # 定义需要寻找的章节名称列表
        section_list = ["Abstract",
                        'Introduction', 'Related Work', 'Background',

                        "Introduction and Motivation", "Computation Function", " Routing Function",

                        "Preliminary", "Problem Formulation",
                        'Methods', 'Methodology', "Method", 'Approach', 'Approaches',
                        # exp
                        "Materials and Methods", "Experiment Settings",
                        'Experiment', "Experimental Results", "Evaluation", "Experiments",
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
            print(sec_index, sec_name, self.section_page_dict[sec_name])
            if sec_index <= 0 and self.abs:
                continue
            else:
                # 直接考虑后面的内容：
                start_page = self.section_page_dict[sec_name]
                if sec_index < len(list(self.section_page_dict.keys())) - 1:
                    end_page = self.section_page_dict[list(self.section_page_dict.keys())[sec_index + 1]]
                else:
                    end_page = len(text_list)
                print("start_page, end_page:", start_page, end_page)
                cur_sec_text = ''
                if end_page - start_page == 0:
                    if sec_index < len(list(self.section_page_dict.keys())) - 1:
                        next_sec = list(self.section_page_dict.keys())[sec_index + 1]
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
                            if sec_index < len(list(self.section_page_dict.keys())) - 1:
                                next_sec = list(self.section_page_dict.keys())[sec_index + 1]
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
        self.args = args
        if args.language == 'en':
            self.language = 'English'
        elif args.language == 'zh':
            self.language = 'Chinese'
        else:
            self.language = 'Chinese'
        self.root_path = root_path
        # 创建一个ConfigParser对象
        self.config = configparser.ConfigParser()
        # 读取配置文件
        self.config.read('apikey.ini')

        # --- Gemini and API Key Configuration ---
        try:
            gemini_api_key = self.config.get('Gemini', 'API_KEY')
            if gemini_api_key and gemini_api_key != 'your_gemini_api_key_here':
                genai.configure(api_key=gemini_api_key)
                print("Gemini API Key loaded and configured.")
                # For safety, prevent harmful content generation
                safety_settings = [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ]
                self.model = genai.GenerativeModel('gemini-pro', safety_settings=safety_settings)
            else:
                print("Warning: Gemini API Key not found or not set in apikey.ini.")
                self.model = None
        except (configparser.NoSectionError, configparser.NoOptionError):
            print("Warning: [Gemini] section or API_KEY not found in apikey.ini.")
            self.model = None

        self.file_format = args.file_format
        if args.save_image:
            self.gitee_key = self.config.get('Gitee', 'api')
        else:
            self.gitee_key = ''

    # 定义一个函数，根据关键词和页码生成arxiv搜索链接
    def get_url(self, keyword, page):
        base_url = "https://arxiv.org/search/?"
        params = {
            "query": keyword,
            "searchtype": "all",  # 搜索所有字段
            "abstracts": "show",  # 显示摘要
            "order": "-announced_date_first",  # 按日期降序排序
            "size": 50  # 每页显示50条结果
        }
        if page > 0:
            params["start"] = page * 50  # 设置起始位置
        return base_url + requests.compat.urlencode(params)

    # 定义一个函数，根据链接获取网页内容，并解析出论文标题
    def get_titles(self, url, days=1):
        titles = []
        # 创建一个空列表来存储论文链接
        links = []
        dates = []
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.find_all("li", class_="arxiv-result")  # 找到所有包含论文信息的li标签
        today = datetime.date.today()
        last_days = datetime.timedelta(days=days)
        for article in articles:
            try:
                title = article.find("p", class_="title").text  # 找到每篇论文的标题，并去掉多余的空格和换行符
                title = title.strip()            
                link = article.find("span").find_all("a")[0].get('href')            
                date_text = article.find("p", class_="is-size-7").text
                date_text = date_text.split('\n')[0].split("Submitted ")[-1].split("; ")[0]
                date_text = datetime.datetime.strptime(date_text, "%d %B, %Y").date()
                if today - date_text <= last_days:
                    titles.append(title.strip())
                    links.append(link)
                    dates.append(date_text)
                # print("links:", links)
            except Exception as e:
                print("error:", e)
                print("error_title:", title)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)          
                
        return titles, links, dates

    # 定义一个函数，根据关键词获取所有可用的论文标题，并打印出来
    def get_all_titles_from_web(self, keyword, page_num=1, days=1):
        title_list, link_list, date_list = [], [], []
        for page in range(page_num):
            url = self.get_url(keyword, page)  # 根据关键词和页码生成链接
            titles, links, dates = self.get_titles(url, days)  # 根据链接获取论文标题
            if not titles:  # 如果没有获取到任何标题，说明已经到达最后一页，退出循环
                break
            for title_index, title in enumerate(titles):  # 遍历每个标题，并打印出来
                print(page, title_index, title, links[title_index], dates[title_index])
            title_list.extend(titles)
            link_list.extend(links)
            date_list.extend(dates)
        print("-" * 40)
        return title_list, link_list, date_list

    def get_arxiv_web(self, args, page_num=1, days=2):
        titles, links, dates = self.get_all_titles_from_web(args.query, page_num=page_num, days=days)
        paper_list = []
        for title_index, title in enumerate(titles):
            if title_index + 1 > args.max_results:
                break
            print(title_index, title, links[title_index], dates[title_index])
            url = links[title_index] + ".pdf"  # the link of the pdf document
            filename = self.try_download_pdf(url, title)
            paper = Paper(path=filename,
                          url=links[title_index],
                          title=title,
                          )
            paper_list.append(paper)
        return paper_list

    def validateTitle(self, title):
        # 将论文的乱七八糟的路径格式修正
        rstr = r"[\/\\\:\*\?\"\<\>\|]"  # '/ \ : * ? " < > |'
        new_title = re.sub(rstr, "_", title)  # 替换为下划线
        return new_title

    def download_pdf(self, url, title):
        response = requests.get(url)  # send a GET request to the url
        date_str = str(datetime.datetime.now())[:13].replace(' ', '-')
        path = self.root_path + 'pdf_files/' + self.validateTitle(self.args.query) + '-' + date_str
        try:
            os.makedirs(path)
        except:
            pass
        filename = os.path.join(path, self.validateTitle(title)[:80] + '.pdf')
        with open(filename, "wb") as f:  # open a file with write and binary mode
            f.write(response.content)  # write the content of the response to the file
        return filename

    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
                    stop=tenacity.stop_after_attempt(5),
                    reraise=True)
    def try_download_pdf(self, url, title):
        return self.download_pdf(url, title)

    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
                    stop=tenacity.stop_after_attempt(5),
                    reraise=True)
    def _call_gemini_api(self, prompt):
        """
        A wrapper function to call the Gemini API with retry logic.
        """
        if not self.model:
            raise ValueError("Gemini model is not initialized. Please check your API key in apikey.ini.")
        
        print("\n--- Calling Gemini API... ---")
        try:
            response = self.model.generate_content(prompt)
            time.sleep(1) # Avoid hitting rate limits
            return response.text
        except Exception as e:
            print(f"An error occurred with Gemini API: {e}")
            # Handle potential content blocking or other API errors
            try:
                # If the response object exists, it might contain feedback
                print(f"Prompt feedback: {response.prompt_feedback}")
            except:
                pass
            return f"Error: Could not generate content due to an API error or content restrictions. {e}"

    def summary_with_chat(self, paper_list):
        # 遍历论文列表
        for paper_index, paper in enumerate(paper_list):
            print(f"\nSummarizing paper {paper_index + 1}/{len(paper_list)}: {paper.title}")
            
            # 1. 第一轮总结：根据标题、摘要和引言，总结论文的核心思想
            prompt1 = f"""
As an expert in academic paper analysis, please provide a concise summary of the following paper based on its title, abstract, and introduction.
Your summary must be in {self.language} and structured into these four sections:
1.  **Background**: What is the research context and background?
2.  **Problem**: What specific problem does this paper address?
3.  **Method**: Briefly describe the core methodology or approach proposed.
4.  **Contribution**: What are the key contributions of this work?

**Title**: {paper.title}
**Abstract**: {paper.abs}
**Introduction**: {paper.section_text_dict.get('Introduction', '')[:3500]}
"""
            summary1 = self._call_gemini_api(prompt1)
            
            # 2. 第二轮总结：结合第一轮总结和方法章节，深入理解方法细节
            method_content = (paper.section_text_dict.get('Method', '') or 
                              paper.section_text_dict.get('Methods', '') or 
                              paper.section_text_dict.get('Methodology', ''))[:5000]
            prompt2 = f"""
Based on the initial summary and the "Method" section of the paper, please provide a detailed explanation of the proposed method.
Focus on the key innovations, technical details, and the overall workflow. Please output in {self.language}.

**Initial Summary**:
{summary1}

**Method Section**:
{method_content}
"""
            summary2 = self._call_gemini_api(prompt2)

            # 3. 第三轮总结：结合前两轮总结和结论，进行全文总结和评价
            conclusion_content = paper.section_text_dict.get('Conclusion', '')[:2500]
            prompt3 = f"""
Based on all the provided information, please provide a final, comprehensive review of the paper.
The review must be in {self.language} and include:
1.  **Overall Summary**: A holistic summary of the paper's content and findings.
2.  **Strengths**: What are the main strengths and innovations of this work?
3.  **Weaknesses**: What are the potential weaknesses, limitations, or unanswered questions?
4.  **Potential Applications**: What are the potential applications or implications of this research?

**Initial Summary**:
{summary1}

**Method Details**:
{summary2}

**Conclusion Section**:
{conclusion_content}
"""
            summary3 = self._call_gemini_api(prompt3)

            # 整合所有总结
            full_summary = f"# {paper.title}\n\n"
            full_summary += f"URL: {paper.url}\n\n"
            full_summary += f"Authors: {', '.join(paper.authers)}\n\n"
            full_summary += f"## 1. Core Idea Summary\n{summary1}\n\n"
            full_summary += f"## 2. Method Details\n{summary2}\n\n"
            full_summary += f"## 3. Final Review & Analysis\n{summary3}\n\n"
            
            # 导出到文件
            date_str = str(datetime.datetime.now())[:13].replace(' ', '-')
            export_path = os.path.join(self.root_path, "export")
            if not os.path.exists(export_path):
                os.makedirs(export_path)
            
            file_name = os.path.join(export_path, f"{self.validateTitle(self.args.query)}-{date_str}.{self.file_format}")
            self.export_to_markdown(full_summary, file_name, mode='a')
            print(f"Summary for '{paper.title}' has been saved to {file_name}")

    def export_to_markdown(self, text, file_name, mode='w'):
        # 打开一个文件，以写入模式
        with open(file_name, mode, encoding="utf-8") as f:
            # 将html格式的内容写入文件
            f.write(text)

    # 定义一个方法，打印出读者信息
    def show_info(self):
        print(f"Key word: {self.key_word}")
        print(f"Query: {self.query}")
        print(f"Sort: {self.sort}")


def chat_arxiv_main(args):
    reader1 = Reader(key_word=args.key_word,
                     query=args.query,
                     args=args
                     )
    reader1.show_info()
    paper_list = reader1.get_arxiv_web(args=args, page_num=args.page_num, days=args.days)

    reader1.summary_with_chat(paper_list=paper_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default='traffic flow prediction', help="the query string, ti: xx, au: xx, all: xx,")
    parser.add_argument("--key_word", type=str, default='GPT robot', help="the key word of user research fields")
    parser.add_argument("--page_num", type=int, default=2, help="the maximum number of page")
    parser.add_argument("--max_results", type=int, default=3, help="the maximum number of results")
    parser.add_argument("--days", type=int, default=10, help="the last days of arxiv papers of this query")
    parser.add_argument("--sort", type=str, default="web", help="another is LastUpdatedDate")
    parser.add_argument("--save_image", default=False,
                        help="save image? It takes a minute or two to save a picture! But pretty")
    parser.add_argument("--file_format", type=str, default='md', help="导出的文件格式，如果存图片的话，最好是md，如果不是的话，txt的不会乱")
    parser.add_argument("--language", type=str, default='zh', help="The other output lauguage is English, is en")

    arxiv_args = ArxivParams(**vars(parser.parse_args()))
    import time

    start_time = time.time()
    chat_arxiv_main(args=arxiv_args)
    print("summary time:", time.time() - start_time)
