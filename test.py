import os
from langchain_community.chat_models import ChatZhipuAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
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
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

# 1 管理历史对话记录
class ChatHistory:
    # 初始化空列表，用来存储历史记录
    def __init__(self):
        self.history = []

    # 填入历史记录，填入每次的 human_message ai_message
    def add_message(self, human_message, ai_message):
        """
        将用户和 AI 的消息添加到历史记录中
        :param human_message: 用户输入的消息
        :param ai_message: AI 返回的消息
        """
        self.history.append(HumanMessage(content=human_message))
        self.history.append(AIMessage(content=ai_message))

     # 获取返回所有的历史记录列表
    def get_history(self):
        """
        获取所有历史记录
        :return: 历史记录列表
        """
        return self.history


# 2 加载知识库文件的类
class KnowledgeBase:
    # 初始化为文档目录路径
    def __init__(self, base_dir):
        self.base_dir = base_dir

    # 遍历所有文件
    def load_documents(self):
        text_splitter = RecursiveCharacterTextSplitter()
        txt_documents = []
        for filename in os.listdir(self.base_dir):
            file_path = os.path.join(self.base_dir, filename)
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                txt_documents.extend(loader.load())
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
                txt_documents.extend(loader.load())
            elif filename.endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
                txt_documents.extend(loader.load())
        txt_chunked_documents = text_splitter.split_documents(documents=txt_documents)
        embeddings = HuggingFaceEmbeddings(model_name="./m3e-base", model_kwargs={'device': "cpu"})
        return FAISS.from_documents(documents=txt_chunked_documents, embedding=embeddings)

# 3 封装用户的问题
class UserQuery:
    def __init__(self, query):
        self.query = query

    def get_query(self):
        return self.query

# 4 存储和返回大模型回答
class Answer:
    def __init__(self):
        self.answer = ""

    def set_answer(self, answer):
        self.answer = answer

    def get_answer(self):
        return self.answer

# 5 Agent，管理整个对话流程和调用大模型
class ChatAgent:
    def __init__(self, api_keys, web_path, doc_base_dir):
        self.api_keys = api_keys
        self.web_path = web_path
        self.doc_base_dir = doc_base_dir
        self.chat_history = ChatHistory()
        self.knowledge_base = KnowledgeBase(doc_base_dir)
        self.answer = Answer()

        # 初始化 API 密钥
        os.environ["ZHIPUAI_API_KEY"] = self.api_keys["zhipuai"]
        os.environ["SERPAPI_API_KEY"] = self.api_keys["serpapi"]
        os.environ["USER_AGENT"] = self.api_keys["user_agent"]

        # 初始化聊天模型
        self.chat_model = ChatZhipuAI()

        # 加载文档并创建向量存储
        self.vector = self._create_vector_store(web_path)
        self.txt_vector = self.knowledge_base.load_documents()

        # 创建检索链
        self.retriever_chain = self._create_retriever_chain(self.vector)
        self.txt_retriever_chain = self._create_retriever_chain(self.txt_vector)

        # 初始化工具和代理
        self.tools = self._load_tools()
        self.agent = self._initialize_agent()

    # 创建网页向量存储
    def _create_vector_store(self, web_path):
        loader = WebBaseLoader(web_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(documents=docs)
        embeddings = HuggingFaceEmbeddings(model_name="./m3e-base", model_kwargs={'device': "cpu"})
        return FAISS.from_documents(documents=documents, embedding=embeddings)

    # 创建检索链
    def _create_retriever_chain(self, vector_store):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an intelligent question answering robot, 
            and when I ask a question, you will provide me with an answer by calling different tools in the agent
            and referencing chat_history.
            You will prioritize using tools and then use your knowledge base if information cannot be retrieved
            Answer the user's questions based on the below context:
            {context}
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
        retriever = vector_store.as_retriever()
        retriever_chain = create_history_aware_retriever(self.chat_model, retriever, prompt)
        document_chain = create_stuff_documents_chain(self.chat_model, prompt)
        return create_retrieval_chain(retriever_chain, document_chain)

    # 加载工具
    def _load_tools(self):
        tools = load_tools(tool_names=["serpapi", "llm-math"], llm=self.chat_model)

        def retrieval_tool(query):
            return self.retriever_chain.invoke({
                "chat_history": self.chat_history.get_history(),
                "input": query
            })["answer"]

        retrieval_tool_instance = Tool(
            name="retrieval_tool",
            description="This tool handles web page retrieval\n "
                        "Questions answering based on chat_history and retrieval_chain.",
            func=retrieval_tool,
        )
        tools.append(retrieval_tool_instance)

        def txt_retrieval_tool(query):
            return self.txt_retriever_chain.invoke({
                "chat_history": self.chat_history.get_history(),
                "input": query
            })["answer"]

        txt_retrieval_tool_instance = Tool(
            name="txt_retrieval_tool",
            description="This tool handles documents retrieval\n"
                        "Questions answering based on chat_history and txt_retrieval_chain.",
            func=txt_retrieval_tool,
        )
        tools.append(txt_retrieval_tool_instance)

        return tools

    # 初始化代理
    def _initialize_agent(self):
        return initialize_agent(
            tools=self.tools,
            llm=self.chat_model,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
        )

    # 处理用户问题
    def ask_question(self, user_query):
        query = user_query.get_query()
        response = self.agent.invoke({
            "input": query,
            "chat_history": self.chat_history.get_history(),
        })
        ai_message = response["output"]
        self.chat_history.add_message(query, ai_message)
        self.answer.set_answer(ai_message)
        return ai_message

    # 启动对话
    def start_conversation(self):
        print("开始与大模型对话")
        while True:
            human_message = input("请输入问题（输入 '结束' 结束）：")
            if human_message == "结束":
                break
            user_query = UserQuery(human_message)
            response = self.ask_question(user_query)
            print("回答：", response)
        print("对话已结束")


if __name__ == "__main__":
    # API 密钥
    api_keys = {
        "zhipuai": "ed6fbd504f7c288c2184de79f8fe5d34.RhC4WOlJt8MocUbk",
        "serpapi": "d2de951e94b9cb687f86da940c5152002568b15ef1d199f5341c42c6ff903a98",
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0"
    }
    # 网页路径
    web_path = "https://baike.baidu.com/item/%E5%94%90%E5%B1%B1%E5%B8%82/8404217"
    # 文档目录路径
    doc_base_dir = "./mydocuments"

    # 创建并启动 ChatAgent 实例
    chat_agent = ChatAgent(api_keys, web_path, doc_base_dir)
    chat_agent.start_conversation()
