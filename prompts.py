from langchain_core.prompts import ChatPromptTemplate,PromptTemplate,MessagesPlaceholder


# PROMPT_PLT = '''Today is {today}. Please Answer the following questions as best you can. You have access to the following tools:

# {tool_description}

# These are chat history before:
# {chat_history}

# Use the following format:
# - Question: the input question you must answer
# - Thought: you should always think about what to do
# - Action: the action to take, should be one of [{tool_names}]
# - Action Input: the input to the action
# - Observation: the result of the action
# ... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
# - Thought: I now know the final answer
# - Final Answer: the final answer to the original input question

# Begin!

# Question: {query}
# {agent_scratchpad}
# '''

ORIGINAL_CHAT = ChatPromptTemplate.from_messages([    
  MessagesPlaceholder(variable_name="chat_history"), 
  ("user", "{input}"),    
  ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation.")
])

CONTEXT_CHAT=ChatPromptTemplate.from_messages([    
  ("system", "Answer the user's questions based on the below context:\n\n{context}, if you cannot find the answew, please refer to other methods."),    
  MessagesPlaceholder(variable_name="chat_history"),    
  ("user", "{input}"),
])


from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

PROMPT_PLT = '''
    Today is {today}. Please Answer the following questions as best you can. 
    You have access to the following tools: {tool_description}
    These are chat history before:{chat_history}  
    Use the following format: 
    - Question: the input question you must answer 
    - Thought: you should always think about what to do 
    - Action: the action to take, should be one of {tool_names}
    - Action Input: the input to the action 
    - Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
    - Thought: I now know the final answer
    - Final Answer: the final answer to the original input question

    Begin!

  Question: {input}
  {agent_scratchpad}
  
  '''
# system_message_prompt = SystemMessagePromptTemplate.from_template(template)
# human_template = "{input}"
# human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# chat_prompt.format_messages(input_language="English", output_language="French", text="I love programming.")
