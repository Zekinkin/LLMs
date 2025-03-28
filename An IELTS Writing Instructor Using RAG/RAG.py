import os
import pandas as pd
from glob import glob
from pathlib import Path
from zipfile import ZipFile
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, CSVLoader, BSHTMLLoader, JSONLoader
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

class RAG:
    def __init__(self):
        self.documents = []
        self.file_loaders = {
            ".md": TextLoader,
            ".txt": TextLoader,
            ".docx": Docx2txtLoader,
            ".csv": CSVLoader,
            ".html": BSHTMLLoader,
            ".json": JSONLoader
        }
        
    # 递归获取所有文件并加载
    def process_folder(self,folder_path):
        for item in glob(os.path.join(folder_path, "**"),recursive=True):
            file_ext = Path(item).suffix.lower()
            if file_ext == ".zip":
                with ZipFile(item, 'r') as zipfile:
                    extract_path = os.path.dirname(item)
                    zipfile.extractall(path=extract_path)
                    zipfile.close()
                    os.remove(item)
                    self.process_folder(extract_path)

            else:
                # 如果是文件，加载并添加元数据
                if file_ext in self.file_loaders:
                    loader = self.file_loaders[file_ext](item)  # 将item传入对应的加载器
                    # 加载对应的文件，返回为Document对象列表
                    docs = loader.load()   
                    # 获取直接父文件夹名称
                    doc_type = os.path.basename(os.path.dirname(item))
                    # 遍历每个Document对象
                    for doc in docs:
                        # 给每个 doc 的 metadata 字典添加键
                        doc.metadata["doc_type"] = doc_type
                        # 将修改后的 doc 添加到外部列表 documents 中
                        self.documents.append(doc)
                        
                elif file_ext == ".xlsx":
                    excel_file = pd.read_excel(item, engine="openpyxl")
                    content = excel_file.to_string()
                    doc = Document(page_content=content,metadata={'source':item})
                    doc_type = os.path.basename(os.path.dirname(item))
                    doc.metadata['doc_type'] = doc_type
                    self.documents.append(doc)
                
                elif file_ext == ".pdf":
                    # 先用 PyPDF2 检查是否有可提取文本
                    reader = PdfReader(item)
                    # 提取第一页
                    text = reader.pages[0].extract_text() or ""
                    # 假设少于10字符视为扫描件
                    is_text_pdf = len(text.strip()) > 10  

                    if is_text_pdf:
                        # 文本 PDF，用 PyPDFLoader
                        loader = PyPDFLoader(item)
                        docs = loader.load()
                        # 获取直接父文件夹名称
                        doc_type = os.path.basename(os.path.dirname(item))
                        # 遍历每个Document对象
                        for doc in docs:
                            # 给每个 doc 的 metadata 字典添加键
                            doc.metadata["doc_type"] = doc_type
                            # 将修改后的 doc 添加到外部列表 documents 中
                            self.documents.append(doc)
                    else:
                        # 设置Tesseract路径
                        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
                        # 将PDF转为图像列表
                        images = convert_from_path(item, poppler_path=r"C:\ProgramData\poppler-0.90.0\bin")
                        # 提取文字并存入Document
                        for image in images:
                            content = pytesseract.image_to_string(image, lang='eng')
                            doc = Document(page_content=content,metadata={'source':item})
                            doc_type = os.path.basename(os.path.dirname(item))
                            doc.metadata['doc_type'] = doc_type
                            self.documents.append(doc)
                elif file_ext in [".png",".jpg"]:
                    content = pytesseract.image_to_string(item, lang='eng')
                    doc = Document(page_content=content,metadata={'source':item})
                    doc_type = os.path.basename(os.path.dirname(item))
                    doc.metadata['doc_type'] = doc_type
                    self.documents.append(doc)

    def create_vectorstore(self,db_name):
        # 将大文档按字符数切分为小块，设置重叠以保留上下文
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  
        # 将每个文档分割成小块，返回新的文档块列表
        chunks = text_splitter.split_documents(self.documents)  

        # use OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()

        # Create our Chroma vectorstore!
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
        print(f"Vectorstore created with {vectorstore._collection.count()} documents")
        return vectorstore   