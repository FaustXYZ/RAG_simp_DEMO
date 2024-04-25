from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings

# from langchain_core.tools import BaseTool


class DocumentService(object):
    def __init__(self, docs_path: list, file_type: str = "PDF", embeddings=OpenAIEmbeddings()):

        self.embeddings = embeddings
        self.type = file_type
        self.docs_path = docs_path

    def init_source_vector(self, add_path=[]):
        if add_path:
            self.docs_path = self.docs_path + add_path
        split_text = self.load_file()
        # 采用embeding模型对文本进行向量化
        vector_store = FAISS.from_documents(split_text, self.embeddings)
        return vector_store

        # self.vector_store.save_local(self.vector_store_path)

    # def load_vector_store(self):
    #     self.vector_store = FAISS.load_local(self.vector_store_path, self.embeddings)

    def load_file(self, **kwargs):
        if self.type == "PDF":
            result = []
            for path in self.docs_path:
                parsed_document = PDFPlumberLoader(path).load()  # 包含了parse功能
                result += RecursiveCharacterTextSplitter(**kwargs).split_documents(parsed_document)
        return result
