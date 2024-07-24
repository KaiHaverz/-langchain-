import os
import json
from langchain_community.chat_models import ChatZhipuAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import (initialize_agent, AgentType)
from langchain.agents import Tool

os.environ["ZHIPUAI_API_KEY"] = "ed6fbd504f7c288c2184de79f8fe5d34.RhC4WOlJt8MocUbk"
os.environ["SERPAPI_API_KEY"] = "d2de951e94b9cb687f86da940c5152002568b15ef1d199f5341c42c6ff903a98"
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0"

zhipuai_chat_model = ChatZhipuAI()
chat_model = zhipuai_chat_model

# 文本分词器
text_splitter = RecursiveCharacterTextSplitter()

# 数据库传来用户文件地址 base_dir, 读取文件
def load_documents_from_base_dir(base_dir):
    txt_documents = []
    for filename in os.listdir(base_dir):
        file_path = os.path.join(base_dir, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif filename.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
        txt_documents.extend(loader.load())
    return txt_documents

# 初始化向量数据库
def initialize_vector_store(documents):
    embeddings = HuggingFaceEmbeddings(model_name="./m3e-base", model_kwargs={'device': "cpu"})
    chunked_documents = text_splitter.split_documents(documents=documents)
    vector_store = FAISS.from_documents(documents=chunked_documents, embedding=embeddings)
    return vector_store

# 初始化历史记录
chat_history = []

# 提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are an intelligent question answering robot, 
    and when I ask a question, you will provide me with an answer by calling different tools in the agent
    and referencing chat_history.
    You will prioritize using tools and then use your knowledge base if information cannot be retrieved.
    Answer the user's questions based on the below context:
    {context}
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])

# 历史检索链和工具初始化
def create_retriever_chain(vector_store):
    retriever = vector_store.as_retriever()
    retriever_chain = create_history_aware_retriever(chat_model, retriever, prompt)
    document_chain = create_stuff_documents_chain(chat_model, prompt)
    retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
    return retrieval_chain

def create_tool(name, description, retrieval_chain):
    def retrieval_tool(query):
        return retrieval_chain.invoke({
            "chat_history": chat_history,
            "input": query
        })["answer"]

    return Tool(name=name, description=description, func=retrieval_tool)

# 将用户文档存储到向量数据库中
def initialize_agent_tools(base_dir):
    documents = load_documents_from_base_dir(base_dir)
    vector_store = initialize_vector_store(documents)

    # 创建检索链
    retrieval_chain = create_retriever_chain(vector_store)

    # 加载工具
    tools = load_tools(tool_names=["serpapi", "llm-math"], llm=chat_model)
    retrieval_tool_instance = create_tool("retrieval_tool", "This tool handles web page retrieval.", retrieval_chain)
    tools.append(retrieval_tool_instance)

    return tools, vector_store

def load_history_from_array(history_array):
    history = []
    for i in range(0, len(history_array), 2):
        human_message = HumanMessage(content=history_array[i])
        ai_message = AIMessage(content=history_array[i + 1]) if i + 1 < len(history_array) else None
        history.append(human_message)
        if ai_message:
            history.append(ai_message)
    return history

def isFirst(history, base_dir, question):
    global chat_history

    # 初始化 agent
    tools, vector_store = initialize_agent_tools(base_dir)
    agent = initialize_agent(
        tools=tools,
        llm=chat_model,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )

    # 保存初始历史记录
    chat_history.extend(history)

    # 调用代理进行问答
    response = agent.invoke({
        "input": question,
        "chat_history": chat_history,
    })

    ai_message = response["output"]
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=ai_message))

    return ai_message

def notFirst(question):
    global chat_history

    # 调用代理进行问答
    response = agent.invoke({
        "input": question,
        "chat_history": chat_history,
    })

    ai_message = response["output"]
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=ai_message))

    return ai_message

def main():
    print("Chat Begins !")
    first = True
    history_array = input("历史记录数组（JSON格式）：")
    history = load_history_from_array(json.loads(history_array))
    base_dir = input("文档所在文件夹路径：")

    while True:
        human_message = input("请输入问题（输入 '结束' 结束）：")
        if human_message == "结束":
            break
        if first:
            ai_message = isFirst(history, base_dir, human_message)
            first = False
        else:
            ai_message = notFirst(human_message)

        print("回答：", ai_message)

    print("对话已结束")
    chat_history.clear()

if __name__ == "__main__":
    main()
