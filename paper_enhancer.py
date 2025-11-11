import os
import json
import pandas as pd
import datetime
import re
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Any
import logging
import shutil


class PaperEnhancer:
    """论文信息增强器 - 用于图片链接刷新和文件组织"""
    
    def __init__(self, export_base_path: str = "export"):
        self.export_base_path = export_base_path
    
    def update_image_links(self, markdown_content: str, paper_title: str, 
                          keyword: str) -> str:
        """
        更新Markdown中的图片链接，按文献标题存储图片
        :param markdown_content: Markdown内容
        :param paper_title: 论文标题
        :param keyword: 关键词
        :return: 更新后的Markdown内容
        """
        # 创建关键词目录
        keyword_dir = os.path.join(self.export_base_path, keyword)
        os.makedirs(keyword_dir, exist_ok=True)
        
        # 创建图片目录（直接放在images下，不按文献建子文件夹）
        images_dir = os.path.join(keyword_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        # 更新图片链接
        updated_content = markdown_content
        
        # 查找并更新所有图片链接
        image_pattern = r'(!\[.*?\]\()(.*?)(\))'
        matches = list(re.finditer(image_pattern, markdown_content))
        
        # 首先修复所有错误的图片路径格式（images_xxx\filename -> images/xxx/filename）
        # 处理错误的路径格式：images_论文标题\文件名
        wrong_pattern = r'(!\[.*?\]\()(images_[^\\]+)\\([^)]+)(\))'
        wrong_matches = list(re.finditer(wrong_pattern, markdown_content))
        
        for match in reversed(wrong_matches):
            full_match = match.group(0)
            prefix = match.group(1)  # ![...](
            wrong_dir = match.group(2)  # images_论文标题
            filename = match.group(3)  # 文件名
            suffix = match.group(4)  # )
            
            # 提取论文标题（去掉images_前缀）
            paper_title_from_path = wrong_dir.replace('images_', '')
            paper_safe_title_from_path = self._validate_filename(paper_title_from_path)
            
            # 创建正确的路径
            correct_path = f"images/{paper_safe_title_from_path}/{filename}"
            new_image_link = f"{prefix}{correct_path}{suffix}"
            
            # 替换错误的图片链接
            markdown_content = markdown_content[:match.start()] + new_image_link + markdown_content[match.end():]
        
        # 从后往前替换，避免索引变化影响替换
        for match in reversed(matches):
            full_match = match.group(0)
            prefix = match.group(1)  # ![...](
            old_path = match.group(2)  # 图片路径
            suffix = match.group(3)  # )
            
            # 只处理本地相对路径，不处理网络图片
            if not old_path.startswith('http'):
                # 提取文件名
                filename = os.path.basename(old_path)
                
                # 创建新的相对路径（相对于Markdown文件）
                new_relative_path = f"images/{filename}"
                
                # 构建新的图片链接
                new_image_link = f"{prefix}{new_relative_path}{suffix}"
                
                # 替换整个图片链接
                updated_content = updated_content[:match.start()] + new_image_link + updated_content[match.end():]
                
        return updated_content
    
    def generate_summary_excel(self, papers_data: List[Dict[str, Any]], 
                              keyword: str) -> str:
        """
        生成汇总Excel表格（简化版本）
        :param papers_data: 论文数据列表
        :param keyword: 关键词
        :return: Excel文件路径
        """
        # 创建关键词目录
        keyword_dir = os.path.join(self.export_base_path, keyword)
        os.makedirs(keyword_dir, exist_ok=True)
        
        # 准备数据
        excel_data = []
        for paper in papers_data:
            # 简化版本：直接使用论文数据
            row = {
                "标题": paper.get("title", ""),
                "作者": ", ".join(paper.get("authors", [])),
                "收录时间": "",
                "最后更新": "",
                "引用量": 0,
                "arXiv ID": "",
                "分类": "",
                "DOI": "",
                "URL": paper.get("url", ""),
                "处理时间": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "关键词": keyword
            }
            excel_data.append(row)
        
        # 创建DataFrame
        df = pd.DataFrame(excel_data)
        
        # 保存Excel文件
        excel_path = os.path.join(keyword_dir, f"论文汇总_{keyword}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
        df.to_excel(excel_path, index=False, engine='openpyxl')
        
        logging.info(f"已生成汇总Excel表格: {excel_path}")
        return excel_path
    
    def organize_by_keyword(self, papers_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        按关键词组织论文数据
        :param papers_data: 论文数据列表
        :return: 按关键词分组的字典
        """
        organized_data = {}
        
        for paper in papers_data:
            keyword = paper.get("keyword", "general")
            if keyword not in organized_data:
                organized_data[keyword] = []
            organized_data[keyword].append(paper)
            
        return organized_data
    
    def save_enhanced_paper(self, markdown_content: str, paper_title: str, 
                          keyword: str, paper_data: Dict[str, Any]) -> str:
        """
        保存增强后的论文Markdown文件
        :param markdown_content: Markdown内容
        :param paper_title: 论文标题
        :param keyword: 关键词
        :param paper_data: 论文数据
        :return: 保存的文件路径
        """
        # 创建关键词目录
        keyword_dir = os.path.join(self.export_base_path, keyword)
        os.makedirs(keyword_dir, exist_ok=True)
        
        # 验证标题
        safe_title = self._validate_filename(paper_title)
        
        # 保存Markdown文件
        md_filename = f"{safe_title}.md"
        md_path = os.path.join(keyword_dir, md_filename)
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
            
        logging.info(f"已保存增强版论文: {md_path}")
        return md_path
    
    def _validate_filename(self, filename: str) -> str:
        """验证文件名，移除非法字符"""
        # 移除Windows文件名中的非法字符
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        # 限制文件名长度
        if len(filename) > 100:
            filename = filename[:100]
        return filename