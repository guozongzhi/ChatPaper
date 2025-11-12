import logging
from abc import ABC, abstractmethod
from typing import List
# chat_paper.py 定义了 Paper 类，这里我们直接导入它
try:
    from chat_paper import Paper 
except ImportError:
    logging.warning("在 retrieval_strategy.py 中导入 Paper 类失败。")
    # 定义一个占位符
    class Paper:
        pass

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
