import logging
from abc import ABC, abstractmethod
from typing import List
# (!!!) 导入已修复，不再从 chat_paper 导入 (!!!)
from paper_class import Paper 

class RetrieverStrategy(ABC):
    """
    论文检索策略的抽象基类 (Interface)。
    所有具体的检索方法（如本地、arXiv、Scholar）都应继承此类。
    """
    
    @abstractmethod
    def retrieve(self, args) -> List[Paper]:
        """
        根据提供的参数 (args) 检索论文。

        :param args: 来自 argparse 的配置对象，包含所有命令行参数。
        :return: 一个 Paper 对象的列表。
        """
        pass