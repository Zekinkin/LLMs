import os
import io
import requests 
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
import base64
from PIL import Image

load_dotenv(override=True)
api_key = os.getenv('DEEPSEEK_API_KEY')
client_ds = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

gk_api_key = os.getenv('GROK_API_KEY')
client_gk = OpenAI(api_key=gk_api_key, base_url="https://api.x.ai/v1")

openai_api_key = os.getenv('OPENAI_API_KEY')
client_oai = OpenAI(api_key=openai_api_key)


system_prompt = """你是我的私人学习指导老师，回答在保持简洁的同时，尽可能体现结构性。
1. **知识传递原则**
   - 三级知识密度控制：
     [1]核心概念 → [2]关键推导 → [3]扩展边界
   - 自动检测知识盲区时，启动「结构化补全」：
     ``` 
     检测到不熟悉RL → 生成强化学习知识地图（分model-free/model-based）
     ```
2. **数学处理规范**
   - 推导过程必须包含：
     [输入条件] → [变换步骤] → [结论验证]
   - 示例：
     ```math
     \frac{d}{dx}e^x = \lim_{h→0}\frac{e^{x+h}-e^x}{h} 
     = e^x \lim_{h→0}\frac{e^h-1}{h} 
     = e^x \cdot 1 = e^x
     ```
3. **结构化输出工具**
   - 对比类知识自动触发表格：
     | 量化方法 | 精度损失 | 硬件需求 |
     |----------|----------|----------|
     | BNB      | 低       | 低       |
     | AWQ      | 中       | 高       |
   - 分类知识生成思维导图：
     ```mermaid
     graph TD
     A[聚类方法] --> B[基于距离]
     A --> C[基于密度]
     ```
4. **交互控制机制**
   - 当您说"详细推导"时，自动展开所有数学步骤
   - 当您说"对比XX和XX"时，强制生成对比表格
   - 检测到关键词"总结"时，输出知识卡片：
     ```
     【知识卡片】Dropout
     - 作用：防止过拟合
     - 数学形式：Bernoulli掩码
     - 典型值：p=0.5
     ```
特别关注我在探索新领域、新知识时，感觉我对某方面的了解很不充足时：
比如：我让你讲解LLM Quantization，我问你什么是bnb格式，你就可以告诉我除了bnb格式，
常用的还有awq格式，awq适合使用vllm。我很需要这样结构化的方式理解新的领域。但是切忌给我太多的细节，比如告诉我所有存在的模型格式，
并一一介绍他们，你只要告诉我主要的，点到即止即可，我需要讲解我会进一步告诉你的！
在输出任何数学公式、符号时，请使用公式格式（允许使用块级公式，如：$$...$$）输出，核心是确保用户能看到渲染过的，美观易读的公式输出：
"""

def process_image(image_data):
    """统一处理文件路径、文件对象、Base64等多种输入"""
    try:
        # 情况1：收到的是文件对象（Hugging Face Spaces）
        if hasattr(image_data, 'name'):  
            with open(image_data.name, "rb") as f:
                image_bytes = f.read()
            ext = image_data.name.split('.')[-1].lower()
        
        # 情况2：收到的是本地文件路径
        elif isinstance(image_data, str) and os.path.exists(image_data):
            with open(image_data, "rb") as f:
                image_bytes = f.read()
            ext = image_data.split('.')[-1].lower()
        
        # 情况3：收到的是PIL图像对象
        elif hasattr(image_data, 'save'):
            buffered = io.BytesIO()
            image_data.save(buffered, format="PNG")
            image_bytes = buffered.getvalue()
            ext = "png"
        
        # 统一转换为Base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        return f"data:image/{ext};base64,{image_base64}"
    
    except Exception as e:
        print(f"Image processing error: {e}")
        return None

def add_message(history, message):
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
    
    if message["files"]:
        for file in message["files"]:
            # 处理图片文件
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_url = process_image(file)
                if not image_url:
                    continue
                    
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
                        {"type": "text", "text": "提取图片中的内容即可，不要额外添加任何描述。数学公式用LaTeX格式（如$$E=mc^2$$）"}
                    ]
                }]
                
                try:
                    completion = client_gk.chat.completions.create(
                        model="grok-2-vision-latest",
                        messages=messages,
                        temperature=0.01
                    )
                    extracted_text = completion.choices[0].message.content
                    history.append({"role": "user", "content": f"图片内容：\n{extracted_text}"})
                except Exception as e:
                    print(f"Vision API error: {e}")
                    history.append({"role": "user", "content": "图片解析失败"})
            
            # 处理文本文件
            elif file.lower().endswith('.txt'):
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        history.append({"role": "user", "content": f.read()})
                except:
                    history.append({"role": "user", "content": "文本文件读取失败"})
    
    if message["text"]:
        history.append({"role": "user", "content": message["text"]})
    
    return history, gr.MultimodalTextbox(value=None, interactive=False)
    

def oai_4omini_bot(history: list):
    messages = [{"role": "system", "content": system_prompt}] + history
    stream = client_oai.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages,
        stream=True)
    history.append({"role": "assistant", "content": ""})
    full_response = ''
    for chunk in stream:
        full_response += chunk.choices[0].delta.content or ''   
        history[-1]["content"] = full_response 
        yield history

def oai_4o_bot(history: list):
    messages = [{"role": "system", "content": system_prompt}] + history
    stream = client_oai.chat.completions.create(
        model='gpt-4o',
        messages=messages,
        stream=True)
    history.append({"role": "assistant", "content": ""})
    full_response = ''
    for chunk in stream:
        full_response += chunk.choices[0].delta.content or ''   
        history[-1]["content"] = full_response 
        yield history

def ds_v3_bot(history: list):
    messages = [{"role": "system", "content": system_prompt}] + history
    stream = client_ds.chat.completions.create(
        model='deepseek-chat',
        messages=messages,
        stream=True)
    history.append({"role": "assistant", "content": ""})
    full_response = ''
    for chunk in stream:
        full_response += chunk.choices[0].delta.content or ''   
        history[-1]["content"] = full_response 
        yield history

def ds_r1_bot(history: list):
    messages = [{"role": "system", "content": system_prompt}] + history
    stream = client_ds.chat.completions.create(
        model='deepseek-reasoner',
        messages=messages,
        stream=True)
    history.append({"role": "assistant", "content": ""})
    full_response = ''
    for chunk in stream:
        full_response += chunk.choices[0].delta.content or ''   
        history[-1]["content"] = full_response 
        yield history

def bot(history, model_choice):
    # 根据 model_choice 调用不同的 bot 函数
    if model_choice == "GPT 4o-mini":
        yield from oai_4omini_bot(history)  # 使用 yield from 传递生成器结果
    elif model_choice == "GPT 4o":
        yield from oai_4o_bot(history)
    elif model_choice == "DeepSeek V3":
        yield from ds_v3_bot(history)
    elif model_choice == "DeepSeek R1":
        yield from ds_r1_bot(history)

with gr.Blocks() as demo:
    gr.Markdown("""# Welcom Zekin!👨🏻‍🚀
                **How Can I Assist With You~🤺**
    """)
    chatbot = gr.Chatbot(elem_id="Zekin's Personal Mentor", 
                         bubble_full_width=False, 
                         type="messages",
                         height=550,
                         show_copy_button=True
                         )

    model_choice = gr.Dropdown(
                label="选择模型",
                choices=["DeepSeek V3", "DeepSeek R1", "GPT 4o-mini", "GPT 4o"],
                value="DeepSeek V3"  # 默认选择 
            )

    chat_input = gr.MultimodalTextbox(
        interactive=True,
        file_count="multiple",
        placeholder="输入文本或粘贴图片...",
        show_label=False,
        sources=["upload"]  # 明确支持剪贴板
    )
    stop_btn = gr.Button("停止")

    # 回车提交功能
    chat_msg = chat_input.submit(add_message, [chatbot, chat_input], [chatbot, chat_input])
    bot_msg = chat_msg.then(bot, [chatbot, model_choice], chatbot)
    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])

    stop_btn.click(None, cancels=[bot_msg])

demo.launch(share=True)