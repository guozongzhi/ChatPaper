import time
import os
import json
import configparser
import google.generativeai as genai


class LLMClient:
    """A small wrapper around google.generativeai to centralize model init,
    retry, rate-limiting and generate() calls.

    Usage:
        client = LLMClient(config, args)
        text = client.generate(prompt)
        model_name = client.current_model_name()
    """

    def __init__(self, config: configparser.ConfigParser, args=None, priority_models=None):
        self.config = config
        self.args = args
        self.enabled = False
        self.model = None
        self.model_name = None
        self._api_calls = []
        self._call_window = 60
        self._max_calls_per_window = 2

        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        if priority_models is None:
            # prefer flash then pro
            priority_models = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-pro"]

        try:
            api_key = self.config.get('Gemini', 'API_KEY')
        except Exception:
            api_key = None

        if not api_key or api_key == 'your_gemini_api_key_here':
            print("LLMClient: Gemini API key not provided. LLM disabled.")
            self.enabled = False
            return

        try:
            genai.configure(api_key=api_key)
            # build model queue
            all_models = [m.name.replace('models/', '') for m in genai.list_models()]
            model_queue = []
            for m in priority_models:
                if m in all_models:
                    model_queue.append(m)
            # fallback to any model
            if not model_queue and all_models:
                model_queue.append(all_models[0])

            # try models with limited retries
            for candidate in model_queue:
                try:
                    print(f"LLMClient: trying model {candidate}")
                    inst = genai.GenerativeModel(candidate)
                    # quick smoke test
                    try:
                        resp = inst.generate_content("Test message")
                        text = getattr(resp, 'text', None)
                        if text:
                            self.model = inst
                            self.model_name = candidate
                            self.enabled = True
                            print(f"LLMClient: initialized model {candidate}")
                            break
                    except Exception:
                        # if smoke test fail, still keep trying
                        pass
                except Exception as e:
                    print(f"LLMClient: failed to initialize model {candidate}: {e}")

            if not self.enabled:
                print("LLMClient: no usable model found, LLM disabled")

        except Exception as e:
            print(f"LLMClient: error during initialization: {e}")
            self.enabled = False

    def _wait_for_rate_limit(self):
        current_time = time.time()
        self._api_calls = [t for t in self._api_calls if current_time - t < self._call_window]
        if len(self._api_calls) >= self._max_calls_per_window:
            wait_time = self._api_calls[0] + self._call_window - current_time
            if wait_time > 0:
                print(f"LLMClient: rate limit reached, sleeping {wait_time:.1f}s")
                time.sleep(wait_time)
                # recursive check
                self._wait_for_rate_limit()
                return
        self._api_calls.append(time.time())

    def current_model_name(self):
        return self.model_name or 'Unknown'

    def generate(self, prompt: str, max_retries: int = 3, retry_delay: int = 60):
        if not self.enabled or not self.model:
            return "抱歉，由于 API 初始化问题，我暂时无法生成响应。请检查您的 API 密钥和模型可用性。"

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    print(f"LLMClient: retry attempt {attempt + 1} for generation")
                # rate limit check
                self._wait_for_rate_limit()
                resp = self.model.generate_content(prompt, safety_settings=self.safety_settings)
                text = getattr(resp, 'text', None)
                if text:
                    return text
                # fallback if property different
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
                print(f"LLMClient: generation error: {e}")
                if any(k in err for k in ('quota', '429', 'rate', 'limit')) and attempt < max_retries - 1:
                    print(f"LLMClient: quota/rate detected, sleeping {retry_delay}s then retrying")
                    time.sleep(retry_delay)
                    continue
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return "抱歉，生成内容时遇到问题，请稍后重试。"


# module-level convenience function
def make_client(config, args=None):
    return LLMClient(config=config, args=args)
