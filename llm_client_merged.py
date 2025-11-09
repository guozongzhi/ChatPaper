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
        self.model = None
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
    
    def initialize(self) -> bool:
        """初始化Gemini客户端"""
        try:
            api_key = self.config.get('Gemini', 'API_KEY')
        except Exception:
            api_key = None

        if not api_key or api_key == 'your_gemini_api_key_here':
            logging.warning("GeminiClient: API key not provided. LLM disabled.")
            self.enabled = False
            return False

        try:
            genai.configure(api_key=api_key)
            
            # 获取可用模型
            all_models = [m.name.replace('models/', '') for m in genai.list_models()]
            
            # 优先级模型列表
            priority_models = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-pro"]
            model_queue = []
            for m in priority_models:
                if m in all_models:
                    model_queue.append(m)
            
            # 回退到任何可用模型
            if not model_queue and all_models:
                model_queue.append(all_models[0])

            # 尝试模型
            for candidate in model_queue:
                try:
                    logging.info("GeminiClient: trying model %s", candidate)
                    inst = genai.GenerativeModel(candidate)
                    # 快速测试
                    try:
                        resp = inst.generate_content("Test message")
                        text = getattr(resp, 'text', None)
                        if text:
                            self.model = inst
                            self.model_name = candidate
                            self.enabled = True
                            logging.info("GeminiClient: initialized model %s", candidate)
                            return True
                    except Exception:
                        pass
                except Exception as e:
                    logging.warning("GeminiClient: failed to initialize model %s: %s", candidate, e)

            if not self.enabled:
                logging.warning("GeminiClient: no usable model found")
                return False
                
        except Exception as e:
            logging.error("GeminiClient: error during initialization: %s", e)
            self.enabled = False
            return False
        
        return True
    
    def generate(self, prompt: str, max_retries: int = 3, retry_delay: int = 60) -> str:
        if not self.enabled or not self.model:
            error_msg = "抱歉，Gemini客户端未初始化或不可用。"
            logging.error("GeminiClient: %s", error_msg)
            return error_msg

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logging.info("GeminiClient: retry attempt %s for generation", attempt + 1)
                
                self._wait_for_rate_limit()
                resp = self.model.generate_content(prompt, safety_settings=self.safety_settings)
                text = getattr(resp, 'text', None)
                if text:
                    return text
                
                # 备用获取方式
                if hasattr(resp, 'candidates'):
                    try:
                        c = resp.candidates
                        if len(c) and hasattr(c[0], 'content'):
                            return c[0].content
                    except Exception:
                        pass
                raise Exception('empty response')
                
            except Exception as e:
                err = str(e).lower()
                logging.exception("GeminiClient: generation error: %s", e)
                
                # 详细的错误分类处理
                if any(k in err for k in ('quota', '429', 'rate', 'limit')) and attempt < max_retries - 1:
                    error_type = "配额或频率限制"
                    logging.warning("GeminiClient: %s 检测到，等待 %s 秒后重试", error_type, retry_delay)
                    time.sleep(retry_delay)
                    continue
                elif any(k in err for k in ('timeout', 'connection', 'network')) and attempt < max_retries - 1:
                    error_type = "网络连接问题"
                    logging.warning("GeminiClient: %s 检测到，等待 %s 秒后重试", error_type, retry_delay)
                    time.sleep(retry_delay)
                    continue
                elif any(k in err for k in ('auth', 'unauthorized', 'invalid', 'key')) and attempt < max_retries - 1:
                    error_type = "认证或API密钥问题"
                    logging.warning("GeminiClient: %s 检测到，等待 %s 秒后重试", error_type, retry_delay)
                    time.sleep(retry_delay)
                    continue
                    
                if attempt < max_retries - 1:
                    logging.warning("GeminiClient: 通用错误，等待 %s 秒后重试", retry_delay)
                    time.sleep(retry_delay)
                    continue
                
                # 最终失败时返回详细的错误信息
                error_msg = f"抱歉，Gemini生成内容时遇到问题：{str(e)}"
                logging.error("GeminiClient: 最终失败 - %s", error_msg)
                return error_msg


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
                    self.model_name = 'deepseek-v3-1-terminus'
                except Exception as e:
                    logging.warning("DeepSeekClient: 火山引擎配置读取失败: %s", e)
                    self.use_volcengine = False
                    
            logging.info("DeepSeekClient: API key found: %s", self.api_key[:10] + "..." if self.api_key else "None")
            logging.info("DeepSeekClient: 使用模式: %s", "火山引擎" if self.use_volcengine else "直接API")
            logging.info("DeepSeekClient: 模型名称: %s", self.model_name)
            
        except Exception as e:
            logging.error("DeepSeekClient: error reading config: %s", e)
            self.api_key = None

        # 在火山引擎模式下，跳过直接API密钥的检查
        if not self.use_volcengine and (not self.api_key or self.api_key.strip() == 'your_deepseek_api_key_here'):
            logging.warning("DeepSeekClient: API key not provided or using placeholder. LLM disabled.")
            self.enabled = False
            return False
            
        # 在火山引擎模式下，检查火山引擎API密钥
        if self.use_volcengine and (not self.api_key or not self.api_key.strip()):
            logging.warning("DeepSeekClient: VolcEngine API key not provided. LLM disabled.")
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
        if client_name in self.clients:
            self.current_client = self.clients[client_name]
            logging.info("LLMClientManager: switched to %s client", client_name)
            return True
        else:
            logging.warning("LLMClientManager: client %s not available", client_name)
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