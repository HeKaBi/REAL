import time
import os
import openai
from .base_language_model import BaseLanguageModel
import dotenv
import tiktoken
from openai import OpenAI
dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ['TIKTOKEN_CACHE_DIR'] = './tmp'

OPENAI_MODEL = ['gpt-4', 'gpt-3.5-turbo']

def get_token_limit(model='gpt-4'):
    """Returns the token limitation of provided model"""
    if model in ['gpt-4', 'gpt-4-0613']:
        num_tokens_limit = 8192
    elif model in ['gpt-3.5-turbo-16k', 'gpt-3.5-turbo-16k-0613']:
        num_tokens_limit = 16384
    elif model in ['gpt-3.5-turbo', 'gpt-3.5-turbo-0613', 'text-davinci-003', 'text-davinci-002']:
        num_tokens_limit = 4096
    else:
        return 8192
    return num_tokens_limit

class ChatGPT(BaseLanguageModel):
    
    @staticmethod
    def add_args(parser):
        parser.add_argument('--retry', type=int, help="retry time", default=1)
    
    def __init__(self, args):
        super().__init__(args)
        self.retry = args.retry
        self.model_name = args.model_name
        self.maximun_token = get_token_limit(self.model_name)
        self.redundant_tokens = 150 
        self._client = None

    @property
    def client(self):
        """
        【修复4】懒加载属性：在子进程中首次调用时才创建连接
        """
        if self._client is None:
            api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
            base_url = os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")
            self._client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )
        return self._client  
    
    def tokenize(self, text):
        """Returns the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")            
        num_tokens = len(encoding.encode(text))        
        return num_tokens + self.redundant_tokens
    
    def prepare_for_inference(self, model_kwargs={}):
        '''
        ChatGPT model does not need to prepare for inference
        '''
        pass
    
    def generate_sentence(self, llm_input):
        cur_retry = 0
        num_retry = self.retry
        # Chekc if the input is too long
        input_length = self.tokenize(llm_input)
        if input_length > self.maximun_token:
            print(f"Input lengt {input_length} is too long. The maximum token is {self.maximun_token}.\n Right tuncate the input to {self.maximun_token} tokens.")
            llm_input = llm_input[:self.maximun_token]
        query = [{"role": "user", "content": llm_input}]
        while cur_retry <= num_retry:
            try:
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=query,
                    timeout=60,
                )
                result = response.choices[0].message.content.strip()
                return result
            except Exception as e:
                print("Message: ", llm_input)
                print("Number of token: ", self.tokenize(llm_input))
                print(e)
                time.sleep(30)
                cur_retry += 1
                continue
        return None
