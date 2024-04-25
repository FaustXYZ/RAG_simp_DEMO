import json
import datetime
from tools import searchtool, wikitool, fileretriver
from models import ContextLLM
from prompts import PROMPT_PLT, ORIGINAL_CHAT, CONTEXT_CHAT
from langchain_core.messages import HumanMessage, AIMessage
import re

# 工具列表,并组合其重要信息，用于让AI agent进行工具选择和反思


tools = [wikitool, fileretriver]
tool_names = "or".join([tool.name for tool in tools])
tool_descs = "\n".join(
    [
        "%s: %s,args: %s"
        % (
            tool.name,
            tool.description,
            json.dumps(
                [
                    {"name": name, "description": info.get("description", ""), "type": info.get("type", "")}
                    for name, info in tool.args.items()
                ],
                ensure_ascii=False,
            ),
        )
        for tool in tools
    ]
)
tools_config = {"tool_names": tool_names, "tools": tools, "tool_descs": tool_descs}
model = ContextLLM.load_model("ChatOpenAI")
propmts = {"document": ORIGINAL_CHAT, "context": CONTEXT_CHAT}


# retrieval_chain=model.create_whole_chain(retriver=retriver
# ,propmts=propmts)

# ans_1=retrieval_chain.invoke({"chat_history":[],"input":"percentile值代表什么"})

# chat_history=[HumanMessage(ans_1['input']),AIMessage(ans_1['answer'])]
# retrieval_chain.invoke({"chat_history":[],"input":"我刚才问了什么问题"})
# retrieval_chain.invoke({"chat_history":chat_history,"input":"我刚才问了什么问题"})

# original_prompt=ORIGINAL_CHAT
# retriever_chain = create_history_aware_retriever(llm=model.model, retriever=retriver,prompt=ORIGINAL_CHAT)

# context_chat=CONTEXT_CHAT
# document_chain = combine_documents.create_stuff_documents_chain(llm=model.model, prompt=context_chat)
# retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)


def agent_execute(query, chat_history=[], **kwargs):
    # global  llm_model, tokenizer

    tools = kwargs.get("tools_config", {}).get("tools", [])
    tool_names = kwargs.get("tools_config", {}).get("tool_names", [])
    tool_descs = kwargs.get("tools_config", {}).get("tool_descs", [])
    prompt_tpl = kwargs.get("prompt", "")
    llm_model = kwargs.get("model", "")

    agent_scratchpad = ""  # agent执行过程
    final_answer_i = thought_i = -1
    action_i = 0
    action_input_i = 1
    # tho_pat=r'- Thought:.*'
    # obs_pat=r'- Observation:.*'
    # act_pat=r'- Action:.*'
    # actinp_pat=r'- Action Input:.*'
    # ans_pat=r'- Final Answer:.*'

    while True:
        # 返回final answer，执行完成
        if final_answer_i != -1 and thought_i < final_answer_i:
            final_answer = response[final_answer_i + len("\nFinal Answer:") :].strip()
            # final_answer = response[final_answer_i + len('\nFinal Answer:'):].strip()
            chat_history.append((query, final_answer))
            return True, final_answer, chat_history
        # 解析action
        if not (thought_i < action_i < action_input_i):
            return False, "LLM回复格式异常", chat_history

        # 1）触发llm思考下一步action
        history = "\n".join(["Question:%s\nAnswer:%s" % (his[0], his[1]) for his in chat_history])
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        prompt = prompt_tpl.format(
            today=today,
            chat_history=history,
            tool_description=tool_descs,
            tool_names=tool_names,
            input=query,
            agent_scratchpad=agent_scratchpad,
        )
        print("\033[32m---等待LLM返回... ...\n%s\n\033[0m" % prompt, flush=True)
        response = llm_model.execute(prompt, history, user_stop_words=["Observation:"]).replace("\n", "")

        # response = llm_model.invoke({"chat_history":chat_history,"input":query})
        # print('\033[34m---LLM返回---\n%s\n---\033[34m' % response, flush=True)

        # 2）解析thought+action+action input+observation or thought+final answer

        # response=response.replace('\n','')
        thought_i = response.rfind("Thought:")
        action_i = response.rfind("Action:")
        action_input_i = response.rfind("Action Input:")
        observation_i = response.rfind("Observation:")
        final_answer_i = response.rfind("Final Answer:")

        if observation_i == -1:
            observation_i = len(response)
            response = response + "Observation: "
        thought = response[thought_i + len("Thought:") : action_i].strip("- ")
        action = response[action_i + len("\nAction:") : action_input_i].strip("- ")
        action_input = response[action_input_i + len("\nAction Input:") : observation_i].strip("- ")

        """
        if not re.findall(obs_pat,response):
            response+='Observation: '
        # if observation_i == -1: #找不到'Observation:'
        #     observation_i = len(response)
        #     response = response + 'Observation: '
        thought = re.findall(tho_pat,response)[-1].strip(tho_pat)
        action = re.findall(act_pat,response)[-1].strip(act_pat)
        # response[action_i + len('\nAction:'):action_input_i].strip()
        action_input = re.findall(actinp_pat,response)[-1].strip(actinp_pat)
        """
        # 5）匹配tool
        the_tool = None
        for t in tools:
            if t.name == action:
                the_tool = t
                break
        if the_tool is None:
            observation = "the tool not exist"
            agent_scratchpad = agent_scratchpad + response + observation + "\n"
            continue

            # 6）执行tool
        try:
            action_input = json.loads(action_input)
            tool_ret = the_tool.invoke(input=json.dumps(action_input))
        except Exception as e:
            observation = "the tool has error:{}".format(e)
        else:
            observation = str(tool_ret)
        agent_scratchpad = agent_scratchpad + response + observation + "\n"


def agent_execute_with_retry(query, chat_history=[], **kwargs):
    is_success, result, chat_history = agent_execute(query, chat_history=chat_history, **kwargs)
    if is_success:
        return {"status": is_success, "result": result, "chat_history": chat_history}

    return {"status": "FAILED"}


# INPUT: In Faust report what will Persons with very high scores on the Conscientiousness scale do?
# THEN it will find Faust report TOOL and do RetrievalQA.invoke
my_history = []
while True:  # True, final_answer, chat_history
    query = input("query:")

    result = agent_execute_with_retry(
        query=query, chat_history=my_history, tools_config=tools_config, prompt=PROMPT_PLT, model=model
    )
    if result.get("status") != "FAILED":
        my_history = result["chat_history"][-10:]  # 保留最新的10次记录
    else:
        print("FAILED, PLEASE RETRY")

