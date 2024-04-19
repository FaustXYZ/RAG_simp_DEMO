import os

from langchain_community.tools import BingSearchRun
from langchain_community.utilities import BingSearchAPIWrapper

os.environ["BING_SUBSCRIPTION_KEY"] = "<key>"
os.environ["BING_SEARCH_URL"] = "https://api.bing.microsoft.com/v7.0/search"
search_api = BingSearchAPIWrapper(k=3)
searchtool = BingSearchRun(api_wrapper=search_api)
# searchtool.description ='This is Bing search tool. Useful for searching some real time info, such as news.'

from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

api_wrapper = WikipediaAPIWrapper(top_k_result=1, doc_content_chars_max=100)
wikitool = WikipediaQueryRun(api_wrapper=api_wrapper)
# wikitool.name = 'Wikipedia'
wikitool.description = "A wrapper around Wikipedia. Useful for when you need to answer general question about definition and the description of people, place, facts, history etc."

# from langchain.chains import combine_documents,create_retrieval_chain,RetrievalQA,create_history_aware_retriever,ConversationChain
from vector_module import FileTool, DocumentService
from langchain_community.tools.vectorstore.tool import VectorStoreQATool
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chat_models.openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool

ret = DocumentService(docs_path=["data/FaustHsu_Report.pdf"])
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# chatQA = ConversationalRetrievalChain.from_llm(
#             ChatOpenAI(),
#             ret.init_source_vector().as_retriever(),
#             memory=memory)
# chatQA
fileretriver = VectorStoreQATool(vectorstore=ret.init_source_vector(), name="XYZFILE", description="XYZFILE")
fileretriver = Tool(
    func=RetrievalQA.from_chain_type(
        llm=ChatOpenAI(), chain_type="stuff", retriever=ret.init_source_vector().as_retriever()
    ).run,
    name="Faust report",
    description="Useful when you need to answer questions about Faust report",
)
# fileretriver.invoke(input="What is the definition of percentile?")
