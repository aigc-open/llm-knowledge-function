import os
import re
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, PyPDFium2Loader, UnstructuredMarkdownLoader, PyPDFLoader, TextLoader
from langchain_core.documents.base import Document
from typing import List, Optional
from langchain_milvus.vectorstores import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, MarkdownTextSplitter, PythonCodeTextSplitter, NLTKTextSplitter, SentenceTransformersTokenTextSplitter, HTMLSectionSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from utils.daily_basic_function import logger_execute_time


class LocalKnowledge:
    @logger_execute_time(doc="加载embedding模型")
    def __init__(self, uri=None, model_name=None, cache_folder=None, local_files_only=True):
        self.uri = uri
        if model_name:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                cache_folder=cache_folder,
                model_kwargs={"local_files_only": local_files_only})
            self.vector_db = Milvus(embedding_function=self.embeddings,
                                    connection_args={"uri": self.uri},
                                    auto_id=True)

    def __load_data_from_file__(self, filename):
        if filename.endswith(".pdf"):
            loader = PyPDFium2Loader(filename)
            Splitter = CharacterTextSplitter
        elif filename.endswith(".doc") or filename.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(filename)
            Splitter = CharacterTextSplitter
        elif filename.endswith(".md") or filename.endswith(".MD"):
            loader = UnstructuredMarkdownLoader(filename)
            Splitter = MarkdownTextSplitter
        elif filename.endswith(".py"):
            loader = TextLoader(filename)
            Splitter = PythonCodeTextSplitter
        elif filename.endswith(".html"):
            loader = TextLoader(filename)
            Splitter = RecursiveCharacterTextSplitter
        else:
            # 全按文本处理
            loader = TextLoader(filename)
            Splitter = CharacterTextSplitter
        docs: List[Document] = loader.load()
        return docs, Splitter

    def pre_docs(self, docs: List[Document], strategys: list = []):
        # 清理文本内容
        def remove_url(text):
            # 匹配 http:// 或 https:// 开头的 URL
            url_pattern = r'https?://\S+'
            return re.sub(url_pattern, '', text)

        def remove_mail_url(text):
            # 匹配标准的邮件地址模式
            mail_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            return re.sub(mail_pattern, '', text)

        def replace_continue_space(text):
            return text.replace("  ", "")

        def replace_continue_n(text):
            return text.replace("\n\n", "")

        if not strategys:
            return

        for doc in docs:
            if hasattr(doc, 'page_content'):
                for strategy in strategys:
                    if strategy == "remove_url":
                        doc.page_content = remove_url(doc.page_content)
                    if strategy == "remove_mail_url":
                        doc.page_content = remove_url(doc.page_content)
                    if strategy == "replace_continue_space":
                        doc.page_content = remove_url(doc.page_content)
                    if strategy == "replace_continue_n":
                        doc.page_content = remove_url(doc.page_content)

    @logger_execute_time(doc="文档拆分")
    def split_documents(self, filename, chunk_size=1000, chunk_overlap=0, namespace="xxxxx", strategys: list = []) -> List[Document]:
        documents, Splitter = self.__load_data_from_file__(filename)
        self.pre_docs(documents, strategys=strategys)
        for doc in documents:
            doc.metadata.update({"namespace": namespace})
        splitter = Splitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_documents(documents)

    @logger_execute_time(doc="文档入库时间")
    def add_documents(self, documents: List[Document]):
        return self.vector_db.add_documents(documents)

    @logger_execute_time(doc="文档入库总时间")
    def filename_to_milvus(self, filename, chunk_size=1000, chunk_overlap=0, namespace="xxxxx", strategys: list = []):
        documents = self.split_documents(
            filename=filename, chunk_size=chunk_size, chunk_overlap=chunk_overlap, namespace=namespace, strategys=strategys)
        return self.add_documents(documents)

    @logger_execute_time(doc="文档查询总时间")
    def similarity_search(self,
                          query,
                          k: int = 4,
                          param: Optional[dict] = None,
                          expr: Optional[str] = None,
                          timeout: Optional[float] = None,):
        return self.vector_db.similarity_search(query, k, param, expr, timeout)
