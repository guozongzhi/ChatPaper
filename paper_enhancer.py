import logging
import os
import re
import datetime
import pandas as pd
import openpyxl # 确保 pandas 可以读写 .xlsx

class PaperEnhancer:
    def __init__(self, export_dir='export'):
        self.export_dir = export_dir
        if not os.path.exists(self.export_dir):
            os.makedirs(self.export_dir)

    def validate_filename(self, filename):
        """清理字符串，使其成为有效的文件名"""
        if not filename:
            filename = "untitled"
        filename = re.sub(r'[\\/*?:"<>|]', '_', filename)
        filename = filename.replace(' ', '_')
        return filename[:150] # 限制长度

    def update_image_links(self, markdown_content, paper_title, keyword):
        """
        更新Markdown中的图片链接，使其指向关键词/标题子目录
        （此功能逻辑保持不变）
        """
        
        # ... (此函数的其余部分保持不变) ...
        
        paper_safe_title = self.validate_filename(paper_title)
        keyword_safe = self.validate_filename(keyword)
        
        # 定义图片目录
        images_dir = os.path.join(self.export_dir, keyword_safe, "images", paper_safe_title)
        
        # 查找所有Markdown图片链接
        pattern = r"!\[(.*?)\]\((.*?)\)"
        
        def replace_link(match):
            alt_text = match.group(1)
            original_path = match.group(2)
            
            # 只修改相对路径 (即我们保存的图片)
            if not original_path.startswith("http"):
                filename = os.path.basename(original_path)
                # 新的相对路径
                new_path = f"images/{paper_safe_title}/{filename}"
                return f"![{alt_text}]({new_path})"
            
            # 保持外部链接不变
            return match.group(0)

        # 替换内容
        updated_content = re.sub(pattern, replace_link, markdown_content)
        
        return updated_content


    def generate_summary_excel(self, papers_data, keyword):
        """
        (!!!) 重写此方法 (!!!)
        生成或 *追加* 论文信息到 Excel 表格。
        - 使用静态文件名 (非时间戳)
        - 如果文件已存在，则读取、追加、去重
        - 包含所有元数据列
        
        :param papers_data: (List[dict]) 包含所有论文信息的列表
        :param keyword: (str) 用于文件命名的关键词
        :return: (str) Excel 文件路径
        """
        
        safe_keyword = self.validate_filename(keyword)
        excel_path = os.path.join(self.export_dir, f"{safe_keyword}_summary.xlsx")
        
        # 1. 定义新数据的列顺序
        # (!!!) 确保 'manual_download' 列在其中 (!!!)
        columns_order = [
            'title', 
            'url', 
            'citation_count', 
            'published_date', 
            'authors', 
            'arxiv_id', 
            'manual_download',
            'keyword', 
            'processed_time'
        ]

        # 2. 将新数据转换为 DataFrame
        if not papers_data:
            logging.info("没有新的论文数据可用于生成 Excel。")
            return excel_path

        df_new = pd.DataFrame(papers_data)
        
        # 确保新数据包含所有列 (以防万一)
        for col in columns_order:
            if col not in df_new.columns:
                df_new[col] = None
        df_new = df_new[columns_order] # 排序

        # 3. (!!!) 追加逻辑 (!!!)
        if os.path.exists(excel_path):
            logging.info(f"检测到已存在的 Excel 文件: {excel_path}。正在追加...")
            try:
                df_old = pd.read_excel(excel_path)
                
                # 4. 合并并去重
                df_combined = pd.concat([df_old, df_new], ignore_index=True)
                
                # (!!!) 关键：按标题和 URL 去重，保留最后一次（最新）的记录 (!!!)
                df_final = df_combined.drop_duplicates(subset=['title', 'url'], keep='last')
                
                logging.info(f"合并后: {len(df_final)} 条记录 (新增 {len(df_final) - len(df_old)} 条)")
                
            except Exception as e:
                logging.warning(f"读取旧 Excel 文件失败: {e}。将覆盖原文件。")
                df_final = df_new
        else:
            logging.info(f"未找到旧 Excel 文件。正在创建新文件: {excel_path}")
            df_final = df_new

        # 5. 保存到 Excel
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df_final.to_excel(writer, index=False, sheet_name='Papers')
            
            logging.info(f"成功保存 Excel: {excel_path}")
        except Exception as e:
            logging.error(f"保存 Excel 失败: {e}", exc_info=True)
            # 备用保存
            try:
                backup_path = os.path.join(self.export_dir, f"{safe_keyword}_summary_backup.csv")
                df_final.to_csv(backup_path, index=False, encoding='utf-8-sig')
                logging.warning(f"Excel 保存失败，已保存为 CSV: {backup_path}")
                return backup_path
            except Exception as e_csv:
                logging.error(f"CSV 备用保存也失败: {e_csv}")

        return excel_path