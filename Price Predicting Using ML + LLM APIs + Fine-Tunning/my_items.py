from typing import Optional
from transformers import AutoTokenizer  # 使用 LLaMA 3.1-8B 模型的分词器（是处理文本的工具，用于将原始文本转换为模型可以理解的数字序列）
import re

BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"

MIN_TOKENS = 150 # 少于此 token 数的内容认为不够有用
MAX_TOKENS = 160 # 超过此 token 数则截断，最终提示文本约为 180 个 token

MIN_CHARS = 300  # 内容字符数下限
CEILING_CHARS = MAX_TOKENS * 7  # 字符数上限，基于 token 数的估算

# 定义一个名为 Item 的类：通过类可以创建多个具有相同结构和行为的对象，封装数据和功能
class Item:
    """
    An Item is a cleaned, curated datapoint of a Product with a Price
    """
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    # PREFIX 和 QUESTION: 用于构造提示文本
    PREFIX = "Price is $"
    QUESTION = "How much does this cost to the nearest dollar?"

    # 要从详情中移除的无用文本列表
    REMOVALS = ['"Batteries Included?": "No"', '"Batteries Included?": "Yes"', '"Batteries Required?": "No"', '"Batteries Required?": "Yes"', "By Manufacturer", "Item", "Date First", "Package", ":", "Number of", "Best Sellers", "Number", "Product "]

    # Item 类表示一个产品数据点，包含：
    title: str
    price: float
    category: str
    token_count: int = 0
    details: Optional[str]
    prompt: Optional[str] = None
    include = False

    # 构造函数，初始化对象
    # self: 类的实例自身，指向当前实例，允许方法访问或修改实例的属性
    # data, price: 传入的参数
    def __init__(self, data, price):
        self.title = data['title']  # 为实例设置一个属性 title，用 data['title'] 赋值
        self.price = price          # 为实例设置一个属性 price，用 price 赋值
        self.parse(data)            # 调用类中的 parse 方法，将参数 data 传递给 parse 方法


    # 定义一个方法 scrub_details，参数 self 表示当前实例，作用是清洗实例的 details 属性
    def scrub_details(self):
        # 访问当前实例的 details 属性（产品详情，通常是字符串）
        details = self.details
        # 访问类属性 REMOVALS，一个包含无用文本的列表
        for remove in self.REMOVALS:
            details = details.replace(remove, "")
        return details


    # 定义一个方法 scrub，用于清洗输入文本，去除无用字符、空格，并过滤掉可能无关的词（7+字符且含数字）
    def scrub(self, stuff):  # stuff 是待清洗的字符串
        # 将特殊字符 & 多余空白替换为单个空格
        stuff = re.sub(r'[:\[\]"{}【】\s]+', ' ', stuff).strip()
        # 移除逗号前的空格 & 将多个连续逗号替换为单个逗号
        stuff = stuff.replace(" ,", ",").replace(",,,",",").replace(",,",",")
        # 将字符串按空格分割成单词列表
        words = stuff.split(' ')
        # 保留 长度小于7的单词 or 不含数字的单词
        select = [word for word in words if len(word)<7 or not any(char.isdigit() for char in word)]
        # 将过滤后的单词列表用空格连接成字符串
        return " ".join(select)
    
    # 定义一个方法 parse，用于解析产品数据，检查是否符合 token 范围要求，并设置 include 属性
    def parse(self, data):   # data 是产品数据（字典）
        # 从 data 中提取description，将description列表中的字符串用换行符 \n 连接成一个字符串
        contents = '\n'.join(data['description'])  
        # 如果 contents 不为空（即描述存在），在末尾添加换行符 \n
        if contents:
            contents += '\n'
        # # 从 data 中提取features，将features列表中的字符串用换行符 \n 连接成一个字符串
        features = '\n'.join(data['features'])
        # 如果 features 不为空（即描述存在），在末尾添加换行符 \n
        if features:
            contents += features + '\n'
        # 从 data 中提取 details，赋给实例属性
        self.details = data['details']
        # 如果 self.details 不为空，调用 self.scrub_details() 清洗详情（移除无用文本），将清洗后的详情追加到 contents，并加换行符
        if self.details:
            contents += self.scrub_details() + '\n'
        # 检查 contents 的字符数是否超过最小要求，如果满足，继续处理；否则，方法结束
        if len(contents) > MIN_CHARS:
            contents = contents[:CEILING_CHARS]
            text = f"{self.scrub(self.title)}\n{self.scrub(contents)}"
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) > MIN_TOKENS:
                tokens = tokens[:MAX_TOKENS]
                text = self.tokenizer.decode(tokens)
                self.make_prompt(text)
                self.include = True

    # 定义一个方法make_prompt，参数 self 是当前实例，text 是清洗后的文本
    def make_prompt(self, text):
        # 问题 + 两个换行符 + 文本 + 两个换行符
        self.prompt = f"{self.QUESTION}\n\n{text}\n\n"
        self.prompt += f"{self.PREFIX}{str(round(self.price))}.00"
        self.token_count = len(self.tokenizer.encode(self.prompt, add_special_tokens=False))
    
    # 定义一个方法 test_prompt，参数 self 是当前实例，返回适合测试的提示，去除实际价格
    def test_prompt(self):
        return self.prompt.split(self.PREFIX)[0] + self.PREFIX

    # 定义特殊方法 __repr__，参数 self 是当前实例
    def __repr__(self):
        # 返回对象的字符串表示，通常用于调试
        return f"<{self.title} = ${self.price}>"
    