import fitz  # PyMuPDF
import logging
import os
import re
import io
from PIL import Image
import datetime # 确保导入 datetime

class Paper:
    def __init__(self, path, title='', url='', abs='', authers=None, 
                 published_date=None, arxiv_id=None, citations=None, 
                 manual_download_required=False): # <--- (!!!) 新增 manual_download_required
        
        self.path = path         # pdf路径 (!!!) (现在可以为 None) (!!!)
        self.title = title
        self.url = url           # 文章链接
        self.abs = abs
        self.authers = authers or []
        self.published_date = published_date
        self.arxiv_id = arxiv_id
        self.citations = citations
        self.manual_download_required = manual_download_required # <--- (!!!) 新增标志 (!!!)
        
        self.section_names = []
        self.section_texts = {}
        self.section_text_dict = {}
        self.title_page = 0
        
        # (!!!) 仅在 path 存在时 (本地文件或已下载) 才尝试打开 PDF (!!!)
        if self.path and os.path.exists(self.path):
            if title == '':
                # 本地文件模式：需要解析标题和内容
                try:
                    self.pdf = fitz.open(self.path) # pdf文档
                    self.title = self.get_title()
                    self.parse_pdf()
                except Exception as e:
                    logging.error(f"打开或解析PDF失败: {self.path} - {e}")
                    self.title = self.title or "无法解析的标题"
                    self.section_text_dict = {'Abstract': f"PDF文件打开失败: {e}"}
                finally:
                    if hasattr(self, 'pdf') and self.pdf:
                        self.pdf.close()
            else:
                # API 模式（已下载）：预填充摘要
                self.section_text_dict = {'Introduction': '', 'Abstract': self.abs}
        elif self.manual_download_required:
            # API 模式（未下载）：预填充摘要，不尝试打开 PDF
            self.section_text_dict = {'Introduction': '', 'Abstract': self.abs}
        else:
            if not self.path:
                logging.warning(f"Paper 对象 '{self.title}' 被创建时没有路径且未标记为手动下载。")
            else:
                logging.warning(f"Paper 对象 '{self.title}' 路径不存在: {self.path}")
        
        self.roman_num = ["I", "II", 'III', "IV", "V", "VI", "VII", "VIII", "IIX", "IX", "X"]
        self.digit_num = [str(d+1) for d in range(10)]
        self.first_image = ''
        
    def parse_pdf(self):
        """
        解析 PDF 内容，填充 self.section_text_dict
        """
        if not self.path or not os.path.exists(self.path):
            logging.warning(f"尝试解析一个不存在的 PDF: {self.path}")
            return
            
        try:
            # 确保 PDF 是打开的 (主要为本地模式)
            if not hasattr(self, 'pdf') or not self.pdf or self.pdf.is_closed:
                self.pdf = fitz.open(self.path)
                
            self.text_list = [page.get_text() for page in self.pdf]
            self.all_text = ' '.join(self.text_list)
            self.section_page_dict = self._get_all_page_index() # 段落与页码的对应字典
            logging.debug("section_page_dict %s", self.section_page_dict)
            self.section_text_dict = self._get_all_page() # 段落与内容的对应字典
            
            # 填充缺失的摘要 (如果本地解析能找到)
            if not self.section_text_dict.get('Abstract'):
                self.section_text_dict['Abstract'] = self.abs or self.get_abstract(self.all_text)
            
            self.section_text_dict.update({"title": self.title})
            self.section_text_dict.update({"paper_info": self.get_paper_info()})
        except Exception as e:
            logging.error(f"解析PDF内容失败 {self.path}: {e}")
            # 确保即使解析失败，摘要也能保留
            if 'Abstract' not in self.section_text_dict:
                 self.section_text_dict['Abstract'] = self.abs or "PDF内容解析失败"
        finally:
            if hasattr(self, 'pdf') and self.pdf and not self.pdf.is_closed:
                 self.pdf.close() # 确保在操作后关闭
    
    def get_abstract(self, all_text):
        """备用方法：从全文本中简单提取摘要 (如果API未提供)"""
        abs_match = re.search(r"(?i)\bAbstract\b([\s\S]*?)(?=\b(1\.?|I\.)\s+Introduction\b)", all_text, re.IGNORECASE)
        if abs_match:
            abstract = abs_match.group(1).strip().replace('\n', ' ')
            return abstract[:2000] # 限制长度
        logging.warning(f"无法从文本中解析摘要: {self.path}")
        return "摘要未找到"

    def get_paper_info(self):
        first_page_text = ""
        if not self.path or not os.path.exists(self.path): return "PDF 路径不存在"
        try:
            if not hasattr(self, 'pdf') or self.pdf.is_closed:
                self.pdf = fitz.open(self.path)
            
            if not self.pdf: raise Exception("PDF 对象未初始化")
                 
            first_page_text = self.pdf[self.title_page].get_text()
            
            if "Abstract" in self.section_text_dict:
                abstract_text = self.section_text_dict['Abstract']
            else:
                abstract_text = self.abs
            
            if abstract_text: # 仅当摘要存在时才替换
                first_page_text = first_page_text.replace(abstract_text, "")
        except Exception as e:
            logging.warning(f"获取 paper info 失败 {self.path}: {e}")
            return "Paper info 获取失败"
        finally:
            if hasattr(self, 'pdf') and self.pdf and not self.pdf.is_closed:
                 self.pdf.close() # 确保在操作后关闭
                 
        return first_page_text
        
    def get_image_path(self, image_path='', max_images=3):
        """
        从PDF中提取并保存重要图片
        """
        if not self.path or not os.path.exists(self.path):
            logging.warning(f"无法提取图片：PDF 路径不存在 {self.path}")
            return []
            
        if not os.path.exists(image_path):
            os.makedirs(image_path)

        saved_images = []
        image_info_list = []  # 存储图片信息：(image, size, ext, page_num)

        try:
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
                            
                            pil_image = Image.open(io.BytesIO(image_bytes))
                            if pil_image.mode == 'RGBA':
                                pil_image = pil_image.convert('RGB')
                            image_size = pil_image.size[0] * pil_image.size[1]
                            
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
                    max_pix = 1000
                    if image.size[0] > image.size[1]:
                        min_pix = int(image.size[1] * (max_pix/image.size[0]))
                        newsize = (max_pix, min_pix)
                    else:
                        min_pix = int(image.size[0] * (max_pix/image.size[1]))
                        newsize = (min_pix, max_pix)
                    
                    resized_image = image.resize(newsize, Image.Resampling.LANCZOS)
                            
                    image_name = f"figure_{i+1}_page{page_num+1}.{ext}"
                    im_path = os.path.join(image_path, image_name)
                    resized_image.save(im_path, quality=95)
                    saved_images.append(im_path)
                    logging.info("已保存图片 %s/%s：%s", i+1, max_images, im_path)
                except Exception as e:
                    logging.warning("保存图片 %s 时出错：%s", i+1, e)
                    continue
        except Exception as e:
            logging.error(f"提取图片失败 {self.path}: {e}")
            
        return saved_images if saved_images else []
        
    def get_chapter_names(self,):
        all_text = ''
        if not self.path or not os.path.exists(self.path): return []
        try:
            doc = fitz.open(self.path) # pdf文档        
            text_list = [page.get_text() for page in doc]
            all_text = ''
            for text in text_list:
                all_text += text
            doc.close() # 关闭文档
        except Exception as e:
            logging.warning(f"获取章节名失败: {e}")
            return []
        
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
                    elif 1 < len(point_split_list) < 5:
                        logging.debug("line: %s", line)
                        chapter_names.append(line)     
        
        return chapter_names
        
    def get_title(self):
        doc = self.pdf
        if not doc:
            logging.error("PDF 对象在 get_title 中未初始化")
            return "PDF 未初始化"
            
        max_font_size = 0
        max_string = ""
        max_font_sizes = [0]
        for page_index, page in enumerate(doc): # 遍历每一页
            try:
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
            except Exception:
                continue # 跳过无法解析的页面

        max_font_sizes.sort()
        logging.debug("max_font_sizes %s", max_font_sizes[-10:])
        cur_title = ''
        for page_index, page in enumerate(doc): # 遍历每一页
            try:
                text = page.get_text("dict") # 获取页面上的文本信息
                blocks = text["blocks"] # 获取文本块列表
                for block in blocks: # 遍历每个文本块
                    if block["type"] == 0 and len(block['lines']): # 如果是文字类型
                        if len(block["lines"][0]["spans"]):
                            cur_string = block["lines"][0]["spans"][0]["text"] # 更新最大值对应的字符串
                            font_flags = block["lines"][0]["spans"][0]["flags"] # 获取第一行第一段文字的字体特征
                            font_size = block["lines"][0]["spans"][0]["size"] # 获取第一行第一段文字的字体大小                         
                            if abs(font_size - max_font_sizes[-1]) < 0.3 or abs(font_size - max_font_sizes[-2]) < 0.3:                        
                                if len(cur_string) > 4 and "arXiv" not in cur_string:                            
                                    if cur_title == ''    :
                                        cur_title += cur_string                       
                                    else:
                                        cur_title += ' ' + cur_string     
                                self.title_page = page_index
            except Exception:
                continue # 跳过无法解析的页面
                
        title = cur_title.replace('\n', ' ')
        logging.debug("detected title: %s", title)
        
        if not title: # 备用标题
            title = os.path.basename(self.path).replace('.pdf', '')
            logging.warning(f"无法从内容检测到标题，使用文件名: {title}")
            
        return title


    def _get_all_page_index(self):
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
        section_page_dict = {}
        if not self.path or not os.path.exists(self.path): return section_page_dict
        try:
            if not hasattr(self, 'pdf') or self.pdf.is_closed:
                logging.warning("PDF 在 _get_all_page_index 中关闭，重新打开")
                self.pdf = fitz.open(self.path)
            
            if not self.pdf: raise Exception("PDF 对象未初始化")

            for page_index, page in enumerate(self.pdf):
                cur_text = page.get_text()
                for section_name in section_list:
                    section_name_upper = section_name.upper()
                    if "Abstract" == section_name and section_name in cur_text:
                        section_page_dict[section_name] = page_index
                    else:
                        if section_name + '\n' in cur_text:
                            section_page_dict[section_name] = page_index
                        elif section_name_upper + '\n' in cur_text:
                            section_page_dict[section_name] = page_index
        except Exception as e:
            logging.warning(f"解析页面索引失败: {e}")
        return section_page_dict

    def _get_all_page(self):
        text = ''
        text_list = []
        section_dict = {}
        
        if not self.path or not os.path.exists(self.path):
            return {'Abstract': 'PDF 路径不存在'}

        if not hasattr(self, 'text_list') or not self.text_list:
            if hasattr(self, 'pdf') and self.pdf and not self.pdf.is_closed:
                self.text_list = [page.get_text() for page in self.pdf]
            else:
                try:
                    doc = fitz.open(self.path)
                    self.text_list = [page.get_text() for page in doc]
                    doc.close()
                except Exception as e:
                    logging.error(f"无法在 _get_all_page 中读取PDF: {e}")
                    return {'Abstract': 'PDF读取失败'}
        
        text_list = self.text_list 

        for sec_index, sec_name in enumerate(self.section_page_dict):
            logging.debug("sec_index=%s sec_name=%s page=%s", sec_index, sec_name, self.section_page_dict[sec_name])
            if sec_index <= 0 and self.abs:
                if sec_name == "Abstract":
                    section_dict[sec_name] = self.abs
                    continue
            
            start_page = self.section_page_dict.get(sec_name)
            if start_page is None:
                logging.warning(f"章节 {sec_name} 在 page_dict 中未找到，跳过")
                continue
                
            if sec_index < len(list(self.section_page_dict.keys()))-1:
                next_sec_name = list(self.section_page_dict.keys())[sec_index+1]
                end_page = self.section_page_dict.get(next_sec_name)
            else:
                end_page = len(text_list)
                
            if end_page is None:
                end_page = len(text_list)
                
            logging.debug("start_page=%s, end_page=%s", start_page, end_page)
            cur_sec_text = ''
            
            if start_page >= len(text_list):
                logging.warning(f"起始页面 {start_page} 超出范围 (总页数 {len(text_list)})")
                continue
                
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