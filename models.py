
from langchain.chat_models.openai import ChatOpenAI
from langchain.schema import(
    SystemMessage,
    HumanMessage
)
from transformers import AutoModel, AutoTokenizer
from langchain.llms.base import LLM
# from langchain.chains import combine_documents,create_retrieval_chain,RetrievalQA,create_history_aware_retriever,ConversationChain

class ContextLLM:
    def __init__(self,model,tokenizer,history=[]):
        self.model=model
        self.tokenizer=tokenizer
        self.historu=history
    
    @classmethod
    def load_model(cls,model:str,need_load:bool=False,**kwargs):
        if need_load:
            device_map = "auto"
            tokenizer = AutoTokenizer.from_pretrained(model,trust_remote_code=True)
            model = AutoModel.from_pretrained(model, trust_remote_code=True,device_map=device_map).half().eval()
            return cls(tokenizer=tokenizer,model=model)
        else:
            model = ChatOpenAI(temperature=0,**kwargs) if model=='ChatOpenAI' else None
            return cls(tokenizer=None,model=model)

    def execute(self,prompt,history=[],**kwargs):
        try:
            messages = [
                SystemMessage(content="You are a helpful assistant."),
            ]
        
            messages.append(SystemMessage(content=history))
            
            messages.append(HumanMessage(content=prompt))
            resp = self.model.invoke(messages)

        # print(resp)
            content = resp.content
            return content
        except Exception as e:
            return str(e)

    # def create_retriver(self,retriver,prompt):
    #     return create_history_aware_retriever(llm=self.model, retriever=retriver,prompt=prompt)

    
    # def create_documents_chain(self,prompt):
    #     return combine_documents.create_stuff_documents_chain(self.model, prompt) 

    # def create_whole_chain(self,retriver,**kwargs):
    #     retriever_chain = self.create_retriver(prompt=kwargs.get('propmts',{}).get('document',''),retriver=retriver.as_retriever())
    #     document_chain=self.create_documents_chain(prompt=kwargs.get('propmts',{}).get('context',''))
        

    #     return create_retrieval_chain(retriever_chain, document_chain)