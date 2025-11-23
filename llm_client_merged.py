import time
import os
import json
import configparser
import logging
import requests
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import google.generativeai as genai
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class BaseLLMClient(ABC):
    """LLM客户端的抽象基类"""
    
    def __init__(self, config: configparser.ConfigParser, args=None):
        self.config = config
        self.args = args
        self.enabled = False
        self.model_name = "Unknown"
        self._api_calls = []
        self._call_window = 60
        self._max_calls_per_window = 2
        
    @abstractmethod
    def initialize(self) -> bool:
        """初始化客户端"""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, max_retries: int = 3, retry_delay: int = 60) -> str:
        """生成文本"""
        pass
    
    def _wait_for_rate_limit(self):
        """等待API限流时间"""
        current_time = time.time()
        self._api_calls = [t for t in self._api_calls if current_time - t < self._call_window]
        if len(self._api_calls) >= self._max_calls_per_window:
            wait_time = self._api_calls[0] + self._call_window - current_time
            if wait_time > 0:
                logging.info("LLMClient: rate limit reached, sleeping %.1fs", wait_time)
                time.sleep(wait_time)
            self._wait_for_rate_limit()
            return
        self._api_calls.append(time.time())
    
    def current_model_name(self) -> str:
        """获取当前模型名称"""
        return self.model_name


class GeminiClient(BaseLLMClient):
    """Google Gemini客户端"""
    
    def __init__(self, config: configparser.ConfigParser, args=None):
        super().__init__(config, args)
        self.api_keys = []
        self.available_models = []
        self.current_api_key_index = 0
        self.current_model_index = 0
        self.model_name = "Unknown"
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
    
    def initialize(self) -> bool:
        """初始化Gemini客户端，支持多API密钥和多模型"""
        try:
            # 读取主API密钥
            if self.config.has_option('Gemini', 'API_KEY'):
                main_key = self.config.get('Gemini', 'API_KEY')
                if main_key and main_key != 'your_gemini_api_key_here':
                    self.api_keys.append(main_key)
            
            # 读取额外的API密钥 (API_KEY_2, API_KEY_3, ...)
            i = 2
            while self.config.has_option('Gemini', f'API_KEY_{i}'):
                extra_key = self.config.get('Gemini', f'API_KEY_{i}')
                if extra_key:
                    self.api_keys.append(extra_key)
                i += 1

        except Exception as e:
            logging.error("GeminiClient: 读取API密钥时出错: %s", e)

        if not self.api_keys:
            logging.warning("GeminiClient: 未提供任何有效的Gemini API密钥。LLM已禁用。")
            self.enabled = False
            return False
            
        logging.info("GeminiClient: 找到 %d 个API密钥。", len(self.api_keys))

        # 使用第一个API密钥进行初始化和模型发现
        try:
            genai.configure(api_key=self.api_keys[self.current_api_key_index])
            
            # 获取所有可用模型
            all_models = [m.name for m in genai.list_models()]
            
            # 优先级模型列表
            priority_models = ["models/gemini-2.5-flash", "models/gemini-2.5-pro", "models/gemini-pro"]
            
            # 尝试实例化并验证每个优先模型
            for model_name in priority_models:
                if model_name in all_models:
                    try:
                        model_instance = genai.GenerativeModel(model_name)
                        # 进行一次快速测试以确保模型可用
                        model_instance.generate_content("test", request_options={'timeout': 20})
                        self.available_models.append(model_instance)
                        logging.info("GeminiClient: 已成功验证并添加模型: %s", model_name.replace('models/', ''))
                    except Exception as e:
                        logging.warning("GeminiClient: 模型 %s 实例化或测试失败: %s", model_name.replace('models/', ''), e)
            
            if self.available_models:
                self.enabled = True
                self.model_name = self.available_models[0].model_name.replace('models/', '')
                logging.info("GeminiClient: 初始化成功，找到 %d 个可用模型。", len(self.available_models))
                return True
            else:
                logging.warning("GeminiClient: 未找到任何可用的Gemini模型。")
                self.enabled = False
                return False
                
        except Exception as e:
            logging.error("GeminiClient: 使用第一个API密钥初始化时出错: %s", e)
            self.enabled = False
            return False

    def _switch_to_next_model(self) -> bool:
        """切换到下一个可用模型"""
        if not self.available_models:
            return False
        self.current_model_index = (self.current_model_index + 1) % len(self.available_models)
        self.model_name = self.current_model_name()
        logging.info("GeminiClient: 切换到下一个模型: %s", self.model_name)
        return True

    def _switch_to_next_key(self) -> bool:
        """切换到下一个API密钥并重新配置"""
        if not self.api_keys:
            return False
        self.current_api_key_index = (self.current_api_key_index + 1) % len(self.api_keys)
        new_key = self.api_keys[self.current_api_key_index]
        logging.warning("GeminiClient: 切换到下一个API密钥 (索引 %d)。", self.current_api_key_index)
        try:
            genai.configure(api_key=new_key)
            # 切换密钥后，重置模型索引以从第一个模型开始重试
            self.current_model_index = 0
            self.model_name = self.current_model_name()
            logging.info("GeminiClient: 已使用新密钥重新配置，并将模型重置为 %s", self.model_name)
            return True
        except Exception as e:
            logging.error("GeminiClient: 使用新API密钥配置失败: %s", e)
            return False
            
    def generate(self, prompt: str, max_retries: int = 2, retry_delay: int = 30) -> str:
        if not self.enabled or not self.available_models:
            error_msg = "抱歉，Gemini客户端未初始化或没有可用的模型。"
            logging.error("GeminiClient: %s", error_msg)
            return error_msg

        initial_key_index = self.current_api_key_index
        initial_model_index = self.current_model_index
        
        # 外层循环：遍历API密钥
        while True:
            # 中层循环：遍历可用模型
            while True:
                current_model = self.available_models[self.current_model_index]
                logging.info("GeminiClient: 正在使用模型 '%s' 和密钥索引 %d 进行生成...", self.current_model_name(), self.current_api_key_index)
                
                # 内层循环：对当前模型和密钥进行重试
                for attempt in range(max_retries):
                    try:
                        if attempt > 0:
                            logging.info("GeminiClient: 对模型 '%s' 的第 %d 次重试...", self.current_model_name(), attempt + 1)
                        
                        self._wait_for_rate_limit()
                        resp = current_model.generate_content(prompt, safety_settings=self.safety_settings)
                        text = getattr(resp, 'text', None)
                        if text:
                            return text
                        
                        raise Exception('empty response from Gemini API')
                        
                    except Exception as e:
                        err_lower = str(e).lower()
                        logging.warning("GeminiClient: 模型 '%s' 生成失败 (尝试 %d/%d): %s", self.current_model_name(), attempt + 1, max_retries, e)
                        
                        # 如果是流量或配额问题，进行等待后重试
                        if any(k in err_lower for k in ('quota', '429', 'rate', 'limit')):
                            if attempt < max_retries - 1:
                                logging.warning("GeminiClient: 检测到流量限制，等待 %d 秒后重试。", retry_delay)
                                time.sleep(retry_delay)
                                continue # 继续内层循环重试
                            else:
                                # 如果最后一次重试仍然是流量问题，则跳出内层循环，尝试切换模型/密钥
                                logging.warning("GeminiClient: 模型 '%s' 在所有重试后仍遇到流量限制。", self.current_model_name())
                                break 
                        else:
                            # 对于其他类型的错误（如连接错误、无效参数等），立即跳出内层重试循环
                            logging.error("GeminiClient: 遇到非流量限制错误，将立即尝试下一个模型/密钥。")
                            break # 跳出内层循环
                
                # 如果内层循环（所有重试）完成或中断，尝试切换到下一个模型
                self._switch_to_next_model()
                # 如果已经把所有模型都试了一遍，跳出中层循环
                if self.current_model_index == initial_model_index:
                    logging.warning("GeminiClient: 已尝试完当前密钥下的所有可用模型。")
                    break

            # 如果中层循环（所有模型）完成，尝试切换到下一个API密钥
            self._switch_to_next_key()
            # 如果已经把所有密钥都试了一遍，跳出外层循环
            if self.current_api_key_index == initial_key_index:
                error_msg = "抱歉，所有Gemini模型和API密钥均尝试失败。"
                logging.error("GeminiClient: %s", error_msg)
                return error_msg
                
    def current_model_name(self) -> str:
        """获取当前活动模型的名称"""
        if self.enabled and self.available_models:
            # 确保索引在范围内
            idx = self.current_model_index % len(self.available_models)
            model_full_name = self.available_models[idx].model_name
            return model_full_name.replace('models/', '')
        return "Unknown"





class DeepSeekClient(BaseLLMClient):
    """DeepSeek客户端 - 支持直接API和火山引擎两种方式"""
    
    def __init__(self, config: configparser.ConfigParser, args=None):
        super().__init__(config, args)
        self.api_key = None
        self.client = None
        self.base_url = "https://api.deepseek.com/v1"
        self.use_volcengine = False
    
    def initialize(self) -> bool:
        """初始化DeepSeek客户端"""
        if OpenAI is None:
            logging.error("DeepSeekClient: OpenAI SDK not installed. Please run: pip install --upgrade 'openai>=1.0'")
            self.enabled = False
            return False
            
        try:
            self.api_key = self.config.get('DeepSeek', 'API_KEY')
            # 使用get方法的安全版本，提供默认值
            try:
                self.model_name = self.config.get('DeepSeek', 'MODEL_NAME')
            except:
                self.model_name = 'deepseek-chat'
            
            # 检查是否使用火山引擎
            try:
                self.use_volcengine = self.config.getboolean('DeepSeek', 'USE_VOLCENGINE')
            except:
                self.use_volcengine = False
                
            # 如果使用火山引擎，读取火山引擎配置
            if self.use_volcengine:
                try:
                    self.volcengine_base_url = self.config.get('DeepSeek', 'VOLCENGINE_BASE_URL', fallback='https://ark.cn-beijing.volces.com/api/v3')
                    self.volcengine_api_key = self.config.get('DeepSeek', 'VOLCENGINE_API_KEY', fallback='')
                    self.base_url = self.volcengine_base_url
                    self.api_key = self.volcengine_api_key
                    # 火山引擎使用特定的模型名称
                    self.model_name = self.config.get('DeepSeek', 'MODEL_NAME', fallback='')
                except Exception as e:
                    logging.warning("DeepSeekClient: 火山引擎配置读取失败: %s", e)
                    self.use_volcengine = False
                    
            logging.info("DeepSeekClient: API key found: %s", self.api_key[:10] + "..." if self.api_key else "None")
            logging.info("DeepSeekClient: 使用模式: %s", "火山引擎" if self.use_volcengine else "直接API")
            logging.info("DeepSeekClient: 模型名称: %s", self.model_name)
            
        except Exception as e:
            logging.error("DeepSeekClient: error reading config: %s", e)
            self.api_key = None

        # 首先检查火山引擎模式
        if self.use_volcengine:
            # 火山引擎模式下，检查火山引擎API密钥
            if not self.api_key or not self.api_key.strip():
                logging.warning("DeepSeekClient: VolcEngine API key not provided. LLM disabled.")
                self.enabled = False
                return False
        else:
            # 直接API模式下，检查直接API密钥
            if not self.api_key or self.api_key.strip() == 'your_deepseek_api_key_here':
                logging.warning("DeepSeekClient: API key not provided or using placeholder. LLM disabled.")
                self.enabled = False
                return False

        try:
            # 初始化OpenAI客户端
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
            
            # 测试API连接
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=10
            )
            
            if completion.choices[0].message.content:
                self.enabled = True
                logging.info("DeepSeekClient: initialized successfully with model %s", self.model_name)
                return True
            else:
                logging.error("DeepSeekClient: API test failed - no content returned")
                self.enabled = False
                return False
                
        except Exception as e:
            logging.error("DeepSeekClient: error during initialization: %s", e)
            self.enabled = False
            return False
    
    def generate(self, prompt: str, max_retries: int = 3, retry_delay: int = 60) -> str:
        if not self.enabled or not self.client:
            error_msg = "抱歉，DeepSeek客户端未初始化或不可用。"
            logging.error("DeepSeekClient: %s", error_msg)
            return error_msg

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logging.info("DeepSeekClient: retry attempt %s for generation", attempt + 1)
                
                self._wait_for_rate_limit()
                
                # 使用OpenAI SDK调用DeepSeek API
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4000,
                    temperature=0.7
                )
                
                # 获取响应内容
                content = completion.choices[0].message.content
                
                # 如果有推理内容，也一并返回
                if hasattr(completion.choices[0].message, 'reasoning_content'):
                    reasoning_content = completion.choices[0].message.reasoning_content
                    if reasoning_content:
                        content = f"推理过程：{reasoning_content}\n\n最终回答：{content}"
                
                return content
                    
            except Exception as e:
                err = str(e).lower()
                logging.exception("DeepSeekClient: generation error: %s", e)
                
                # 详细的错误分类处理
                if any(k in err for k in ('429', 'rate', 'limit')) and attempt < max_retries - 1:
                    error_type = "频率限制"
                    logging.warning("DeepSeekClient: %s 检测到，等待 %s 秒后重试", error_type, retry_delay)
                    time.sleep(retry_delay)
                    continue
                elif any(k in err for k in ('timeout', 'connection', 'network')) and attempt < max_retries - 1:
                    error_type = "网络连接问题"
                    logging.warning("DeepSeekClient: %s 检测到，等待 %s 秒后重试", error_type, retry_delay)
                    time.sleep(retry_delay)
                    continue
                elif any(k in err for k in ('auth', 'unauthorized', 'invalid', 'key')) and attempt < max_retries - 1:
                    error_type = "认证或API密钥问题"
                    logging.warning("DeepSeekClient: %s 检测到，等待 %s 秒后重试", error_type, retry_delay)
                    time.sleep(retry_delay)
                    continue
                    
                if attempt < max_retries - 1:
                    logging.warning("DeepSeekClient: 通用错误，等待 %s 秒后重试", retry_delay)
                    time.sleep(retry_delay)
                    continue
                
                # 最终失败时返回详细的错误信息
                error_msg = f"抱歉，DeepSeek生成内容时遇到问题：{str(e)}"
                logging.error("DeepSeekClient: 最终失败 - %s", error_msg)
                return error_msg


class KimiClient(BaseLLMClient):
    """Kimi客户端"""
    
    def __init__(self, config: configparser.ConfigParser, args=None):
        super().__init__(config, args)
        self.api_key = None
        self.base_url = "https://api.moonshot.cn/v1"
    
    def initialize(self) -> bool:
        """初始化Kimi客户端"""
        try:
            self.api_key = self.config.get('Kimi', 'API_KEY')
        except Exception:
            self.api_key = None

        if not self.api_key or self.api_key == 'your_kimi_api_key_here':
            logging.warning("KimiClient: API key not provided. LLM disabled.")
            self.enabled = False
            return False

        try:
            # 测试API连接
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "moonshot-v1-8k",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 10
            }
            
            response = requests.post(f"{self.base_url}/chat/completions", 
                                   headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                self.enabled = True
                self.model_name = "moonshot-v1-8k"
                logging.info("KimiClient: initialized successfully")
                return True
            else:
                logging.error("KimiClient: API test failed with status %s", response.status_code)
                self.enabled = False
                return False
                
        except Exception as e:
            logging.error("KimiClient: error during initialization: %s", e)
            self.enabled = False
            return False
    
    def generate(self, prompt: str, max_retries: int = 3, retry_delay: int = 60) -> str:
        if not self.enabled or not self.api_key:
            error_msg = "抱歉，Kimi客户端未初始化或不可用。"
            logging.error("KimiClient: %s", error_msg)
            return error_msg

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logging.info("KimiClient: retry attempt %s for generation", attempt + 1)
                
                self._wait_for_rate_limit()
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": "moonshot-v1-8k",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 4000,
                    "temperature": 0.7
                }
                
                response = requests.post(f"{self.base_url}/chat/completions", 
                                       headers=headers, json=data, timeout=120)
                
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    error_msg = f"API error: {response.status_code}"
                    if response.text:
                        error_msg += f" - {response.text}"
                    raise Exception(error_msg)
                    
            except Exception as e:
                err = str(e).lower()
                logging.exception("KimiClient: generation error: %s", e)
                
                # 详细的错误分类处理
                if any(k in err for k in ('429', 'rate', 'limit')) and attempt < max_retries - 1:
                    error_type = "频率限制"
                    logging.warning("KimiClient: %s 检测到，等待 %s 秒后重试", error_type, retry_delay)
                    time.sleep(retry_delay)
                    continue
                elif any(k in err for k in ('timeout', 'connection', 'network')) and attempt < max_retries - 1:
                    error_type = "网络连接问题"
                    logging.warning("KimiClient: %s 检测到，等待 %s 秒后重试", error_type, retry_delay)
                    time.sleep(retry_delay)
                    continue
                elif any(k in err for k in ('auth', 'unauthorized', 'invalid', 'key')) and attempt < max_retries - 1:
                    error_type = "认证或API密钥问题"
                    logging.warning("KimiClient: %s 检测到，等待 %s 秒后重试", error_type, retry_delay)
                    time.sleep(retry_delay)
                    continue
                    
                if attempt < max_retries - 1:
                    logging.warning("KimiClient: 通用错误，等待 %s 秒后重试", retry_delay)
                    time.sleep(retry_delay)
                    continue
                
                # 最终失败时返回详细的错误信息
                error_msg = f"抱歉，Kimi生成内容时遇到问题：{str(e)}"
                logging.error("KimiClient: 最终失败 - %s", error_msg)
                return error_msg


class QwenClient(BaseLLMClient):
    """千问客户端"""
    
    def __init__(self, config: configparser.ConfigParser, args=None):
        super().__init__(config, args)
        self.api_key = None
        self.base_url = "https://dashscope.aliyuncs.com/api/v1"
    
    def initialize(self) -> bool:
        """初始化千问客户端"""
        try:
            self.api_key = self.config.get('Qwen', 'API_KEY')
        except Exception:
            self.api_key = None

        if not self.api_key or self.api_key == 'your_qwen_api_key_here':
            logging.warning("QwenClient: API key not provided. LLM disabled.")
            self.enabled = False
            return False

        try:
            # 测试API连接
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "qwen-turbo",
                "input": {
                    "messages": [{"role": "user", "content": "test"}]
                },
                "parameters": {
                    "max_tokens": 10
                }
            }
            
            response = requests.post(f"{self.base_url}/services/aigc/text-generation/generation", 
                                   headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                self.enabled = True
                self.model_name = "qwen-turbo"
                logging.info("QwenClient: initialized successfully")
                return True
            else:
                logging.error("QwenClient: API test failed with status %s", response.status_code)
                self.enabled = False
                return False
                
        except Exception as e:
            logging.error("QwenClient: error during initialization: %s", e)
            self.enabled = False
            return False
    
    def generate(self, prompt: str, max_retries: int = 3, retry_delay: int = 60) -> str:
        if not self.enabled or not self.api_key:
            error_msg = "抱歉，千问客户端未初始化或不可用。"
            logging.error("QwenClient: %s", error_msg)
            return error_msg

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logging.info("QwenClient: retry attempt %s for generation", attempt + 1)
                
                self._wait_for_rate_limit()
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": "qwen-turbo",
                    "input": {
                        "messages": [{"role": "user", "content": prompt}]
                    },
                    "parameters": {
                        "max_tokens": 4000,
                        "temperature": 0.7
                    }
                }
                
                response = requests.post(f"{self.base_url}/services/aigc/text-generation/generation", 
                                       headers=headers, json=data, timeout=120)
                
                if response.status_code == 200:
                    result = response.json()
                    return result["output"]["choices"][0]["message"]["content"]
                else:
                    error_msg = f"API error: {response.status_code}"
                    if response.text:
                        error_msg += f" - {response.text}"
                    raise Exception(error_msg)
                    
            except Exception as e:
                err = str(e).lower()
                logging.exception("QwenClient: generation error: %s", e)
                
                # 详细的错误分类处理
                if any(k in err for k in ('429', 'rate', 'limit')) and attempt < max_retries - 1:
                    error_type = "频率限制"
                    logging.warning("QwenClient: %s 检测到，等待 %s 秒后重试", error_type, retry_delay)
                    time.sleep(retry_delay)
                    continue
                elif any(k in err for k in ('timeout', 'connection', 'network')) and attempt < max_retries - 1:
                    error_type = "网络连接问题"
                    logging.warning("QwenClient: %s 检测到，等待 %s 秒后重试", error_type, retry_delay)
                    time.sleep(retry_delay)
                    continue
                elif any(k in err for k in ('auth', 'unauthorized', 'invalid', 'key')) and attempt < max_retries - 1:
                    error_type = "认证或API密钥问题"
                    logging.warning("QwenClient: %s 检测到，等待 %s 秒后重试", error_type, retry_delay)
                    time.sleep(retry_delay)
                    continue
                    
                if attempt < max_retries - 1:
                    logging.warning("QwenClient: 通用错误，等待 %s 秒后重试", retry_delay)
                    time.sleep(retry_delay)
                    continue
                
                # 最终失败时返回详细的错误信息
                error_msg = f"抱歉，千问生成内容时遇到问题：{str(e)}"
                logging.error("QwenClient: 最终失败 - %s", error_msg)
                return error_msg


class DoubaoClient(BaseLLMClient):
    """豆包客户端"""
    
    def __init__(self, config: configparser.ConfigParser, args=None):
        super().__init__(config, args)
        self.api_key = None
        self.client = None
        self.base_url = "https://ark.cn-beijing.volces.com/api/v3"
    
    def initialize(self) -> bool:
        """初始化豆包客户端"""
        if OpenAI is None:
            logging.error("DoubaoClient: OpenAI SDK not installed. Please run: pip install --upgrade 'openai>=1.0'")
            self.enabled = False
            return False
            
        try:
            self.api_key = self.config.get('Doubao', 'API_KEY')
            # 使用get方法的安全版本，提供默认值
            try:
                self.model_name = self.config.get('Doubao', 'MODEL_NAME')
            except:
                self.model_name = 'doubao-seed-1-6-lite-251015'
            logging.info("DoubaoClient: API key found: %s", self.api_key[:10] + "..." if self.api_key else "None")
        except Exception as e:
            logging.error("DoubaoClient: error reading config: %s", e)
            self.api_key = None

        if not self.api_key or self.api_key.strip() == 'your_doubao_api_key_here':
            logging.warning("DoubaoClient: API key not provided or using placeholder. LLM disabled.")
            self.enabled = False
            return False

        try:
            # 初始化OpenAI客户端
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
            
            # 测试API连接
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=10
            )
            
            if completion.choices[0].message.content:
                self.enabled = True
                logging.info("DoubaoClient: initialized successfully with model %s", self.model_name)
                return True
            else:
                logging.error("DoubaoClient: API test failed - no content returned")
                self.enabled = False
                return False
                
        except Exception as e:
            logging.error("DoubaoClient: error during initialization: %s", e)
            self.enabled = False
            return False
    
    def generate(self, prompt: str, max_retries: int = 3, retry_delay: int = 60) -> str:
        if not self.enabled or not self.client:
            error_msg = "抱歉，豆包客户端未初始化或不可用。"
            logging.error("DoubaoClient: %s", error_msg)
            return error_msg

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logging.info("DoubaoClient: retry attempt %s for generation", attempt + 1)
                
                self._wait_for_rate_limit()
                
                # 使用OpenAI SDK调用豆包API
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4000,
                    temperature=0.7
                )
                
                # 获取响应内容
                content = completion.choices[0].message.content
                
                # 如果有推理内容，也一并返回
                if hasattr(completion.choices[0].message, 'reasoning_content'):
                    reasoning_content = completion.choices[0].message.reasoning_content
                    if reasoning_content:
                        content = f"推理过程：{reasoning_content}\n\n最终回答：{content}"
                
                return content
                    
            except Exception as e:
                err = str(e).lower()
                logging.exception("DoubaoClient: generation error: %s", e)
                
                # 详细的错误分类处理
                if any(k in err for k in ('429', 'rate', 'limit')) and attempt < max_retries - 1:
                    error_type = "频率限制"
                    logging.warning("DoubaoClient: %s 检测到，等待 %s 秒后重试", error_type, retry_delay)
                    time.sleep(retry_delay)
                    continue
                elif any(k in err for k in ('timeout', 'connection', 'network')) and attempt < max_retries - 1:
                    error_type = "网络连接问题"
                    logging.warning("DoubaoClient: %s 检测到，等待 %s 秒后重试", error_type, retry_delay)
                    time.sleep(retry_delay)
                    continue
                elif any(k in err for k in ('auth', 'unauthorized', 'invalid', 'key')) and attempt < max_retries - 1:
                    error_type = "认证或API密钥问题"
                    logging.warning("DoubaoClient: %s 检测到，等待 %s 秒后重试", error_type, retry_delay)
                    time.sleep(retry_delay)
                    continue
                    
                if attempt < max_retries - 1:
                    logging.warning("DoubaoClient: 通用错误，等待 %s 秒后重试", retry_delay)
                    time.sleep(retry_delay)
                    continue
                
                # 最终失败时返回详细的错误信息
                error_msg = f"抱歉，豆包生成内容时遇到问题：{str(e)}"
                logging.error("DoubaoClient: 最终失败 - %s", error_msg)
                return error_msg


class LLMClientManager:
    """LLM客户端管理器"""
    
    def __init__(self, config: configparser.ConfigParser, args=None):
        self.config = config
        self.args = args
        self.clients = {}
        self.current_client = None
        
        # 按优先级初始化客户端
        self._initialize_clients()
    
    def _initialize_clients(self):
        """初始化客户端，如果指定了特定客户端则只初始化该客户端"""
        client_classes = [
            ("Gemini", GeminiClient),
            ("DeepSeek", DeepSeekClient),
            ("Kimi", KimiClient),
            ("Qwen", QwenClient),
            ("Doubao", DoubaoClient)
        ]
        
        # 检查是否指定了特定的客户端
        specified_client = None
        if self.args and hasattr(self.args, 'llm_client') and self.args.llm_client:
            specified_client = self.args.llm_client
            logging.info("LLMClientManager: 指定使用客户端: %s", specified_client)
        
        # 如果指定了特定客户端，只初始化该客户端
        if specified_client:
            for client_name, client_class in client_classes:
                if client_name.lower() == specified_client.lower():
                    try:
                        client = client_class(self.config, self.args)
                        if client.initialize():
                            self.clients[client_name] = client
                            self.current_client = client
                            logging.info("LLMClientManager: %s 客户端初始化成功", client_name)
                            return
                        else:
                            logging.error("LLMClientManager: 指定的客户端 %s 初始化失败", client_name)
                    except Exception as e:
                        logging.error("LLMClientManager: 初始化指定客户端 %s 时出错: %s", client_name, e)
            # 如果指定的客户端初始化失败，继续初始化其他客户端作为备用
            logging.warning("LLMClientManager: 指定的客户端 %s 不可用，将尝试其他客户端", specified_client)
        
        # 自动模式：按优先级初始化所有可用的客户端
        for client_name, client_class in client_classes:
            # 如果已经指定了客户端但初始化失败，跳过该客户端
            if specified_client and client_name.lower() == specified_client.lower():
                continue
                
            try:
                client = client_class(self.config, self.args)
                if client.initialize():
                    self.clients[client_name] = client
                    logging.info("LLMClientManager: %s client initialized successfully", client_name)
                    # 设置第一个可用的客户端为当前客户端
                    if self.current_client is None:
                        self.current_client = client
                        logging.info("LLMClientManager: using %s as default client", client_name)
                else:
                    logging.warning("LLMClientManager: %s client initialization failed", client_name)
            except Exception as e:
                logging.error("LLMClientManager: error initializing %s client: %s", client_name, e)
        
        if self.current_client is None:
            logging.warning("LLMClientManager: no LLM client available")
    
    def generate(self, prompt: str, max_retries: int = 3, retry_delay: int = 60) -> str:
        """使用当前客户端生成文本"""
        if self.current_client:
            return self.current_client.generate(prompt, max_retries, retry_delay)
        else:
            return "抱歉，没有可用的LLM客户端。请检查API密钥配置。"
    
    def current_model_name(self) -> str:
        """获取当前模型名称"""
        if self.current_client:
            return self.current_client.current_model_name()
        else:
            return "Unknown"
    
    def switch_client(self, client_name: str) -> bool:
        """切换到指定的客户端"""
        # 处理客户端名称的大小写不匹配问题
        available_clients = self.get_available_clients()
        
        # 精确匹配
        if client_name in available_clients:
            self.current_client = self.clients[client_name]
            logging.info("LLMClientManager: switched to %s client", client_name)
            return True
        
        # 大小写不敏感匹配
        for available_client in available_clients:
            if available_client.lower() == client_name.lower():
                self.current_client = self.clients[available_client]
                logging.info("LLMClientManager: switched to %s client (case-insensitive match)", available_client)
                return True
        
        logging.warning("LLMClientManager: client %s not available. Available clients: %s", client_name, available_clients)
        return False
    
    def get_available_clients(self) -> list:
        """获取可用的客户端列表"""
        return list(self.clients.keys())


# 模块级便利函数
def make_client(config, args=None):
    """创建LLM客户端管理器"""
    return LLMClientManager(config, args)


# 向后兼容的旧版LLMClient类
class LLMClient:
    """向后兼容的LLM客户端包装器"""
    
    def __init__(self, config: configparser.ConfigParser, args=None, priority_models=None):
        self.manager = LLMClientManager(config, args)
        self.enabled = self.manager.current_client is not None
    
    def generate(self, prompt: str, max_retries: int = 3, retry_delay: int = 60) -> str:
        return self.manager.generate(prompt, max_retries, retry_delay)
    
    def current_model_name(self) -> str:
        return self.manager.current_model_name()