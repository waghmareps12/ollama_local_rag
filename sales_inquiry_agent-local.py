import streamlit as st
# from streamlit_extras.add_vertical_space import add_vertical_space
from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_openai import AzureOpenAI, AzureOpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import initialize_agent, AgentType, Tool
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
import ollama

load_dotenv() # read local .env file
OPENAI_API_TYPE = os.getenv('OPENAI_API_TYPE')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_API_BASE = os.getenv('OPENAI_API_BASE')
OPENAI_API_VERSION = os.getenv('OPENAI_API_VERSION')
# # Create embeddingsclear
embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=False)

# Set up conversational memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Load the saved index
loaded_vectorstore = FAISS.load_local("my_faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = loaded_vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# db = Chroma(persist_directory="./db-sales-enquiry",
#             embedding_function=embeddings)

# # # Create retriever
# retriever = db.as_retriever(
#     search_type="similarity",
#     search_kwargs= {"k": 5}
# )

# Create retriever

# index_name = "faiss_index"
# if os.path.exists(f"{index_name}.faiss") and os.path.exists(f"{index_name}.pkl"):
#     vectorstore = FAISS.load_local(index_name, embeddings)
#     print("Loaded existing FAISS index.")

# # Create Ollama language model - Gemma 2
# local_llm = 'llama3.2:latest'

# llm = ChatOllama(model=local_llm,
#                  keep_alive="3h", 
#                  max_tokens=512,  
#                  temperature=0)

# llm = AzureChatOpenAI(openai_api_base=OPENAI_API_BASE,
#                         openai_api_version=OPENAI_API_VERSION,
#                         openai_api_key=OPENAI_API_KEY,
#                         openai_api_type=OPENAI_API_TYPE,
#                         #deployment_name = 'genai-gpt-4-32k',
#                         #model_name = 'gpt-4-32k')
#                         deployment_name = 'genai-gpt-35-turbo',
#                         model_name = 'gpt-4o',
#                         temperature = 0)






# # Create prompt template
template = """<bos><start_of_turn>user
You are a sales inquiry assistant specializing in rebates, market share, promotions, and payouts. Answer the question based only on the following context and provide a meaningful response. 
Write in full sentences with correct spelling and punctuation. Use bullet points or numbered lists when appropriate.
If the context doesn't contain the answer, politely state that you don't have the specific information and suggest contacting the sales department for more details.

CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}

<end_of_turn>
<start_of_turn>model
ANSWER:"""
prompt = ChatPromptTemplate.from_template(template)
# Retrieve chat history
chat_history = memory.load_memory_variables({}).get('chat_history', "")

# If chat_history is a list, convert to a string or another format that can be handled
if isinstance(chat_history, list):
    chat_history = "\n".join(chat_history)
# print(chat_history)
# # Modify the rag_chain to return the result instead of streaming

df = pd.read_csv(r'./dummy_CAR.csv').rename(columns=str.lower)
df_columns =df.columns
# qa_chain = RetrievalQA.from_chain_type(
#     llm, retriever=retriever, chain_type_kwargs={"prompt": prompt}
# )

# # Streamlit UI

def check_sap_in_roster(df: pd.DataFrame, sap_value: int) -> pd.DataFrame:
    """Check if a particular SAP value exists in the roster. and say if it exist also provide the name and address along with rfp_name"""
    return df[df['sap'] == sap_value][['sap','rfp_name_random',
       'rfp_location', 'name', 'address',]]
tools = [
    Tool(
    name="check_sap_in_roster",
    func=lambda x: check_sap_in_roster(df, int(x)),
    description="""
    Useful for checking if a specific SAP value exists in the roster dataframe.
    Input: A numeric SAP value (e.g., 100241).
    Output: If the SAP exists, it returns relevant details such as rfp_name, name, and address.
    If the SAP doesn't exist, it returns an empty DataFrame.
    include and summarise all the information returned by the tool
    """
),
    Tool(
        name="FAQ for sales Inquiries",
        func=retriever.get_relevant_documents,
       description="Use this tool for general sales inquiries, questions about the tracker, or when you can't find specific information about an account. Input should be the full question or description of the issue."
    ),


    
]
# Create custom agent prompt with friendly HR assistant context
agent_kwargs = {
    'prefix': '''You are a sales inquiry assistant specializing in rebates,
      market share, promotions, and payouts. Your primary tasks are:
1. Checking if SAP numbers exist in the roster
2. Answering general sales inquiries
3. Helping with issues related to finding accounts in the tracker

Follow these steps for each query:
1. Determine which tool is most appropriate for the query.
2. Use the selected tool and analyze its output.
3. If the tool doesn't provide a satisfactory answer, consider using the other tool or stating that you don't have the information.
4. Provide a clear and concise answer based on the tool's output.

Remember:
- For SAP number checks, use the "check_sap_in_roster" tool.
- For all other inquiries, including tracker issues, use the "FAQ_for_sales_inquiries" tool.
- If you can't find an answer, politely state that and suggest contacting the sales department for further assistance.

You have access to the following tools:''',
    'format_instructions': '''Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question''',
    'suffix': '''Begin!

Question: {input}
Thought: Let's approach this step-by-step:'''
}



# Custom function to handle agent execution
def execute_agent(agent, query):
    try:
        result = agent.run(query)
        return result
    except Exception as e:
        error_msg = str(e)
        if "Could not parse LLM output" in error_msg:
            # Extract the actual response from the error message
            actual_response = error_msg.split("Could not parse LLM output: `")[1].rsplit("`", 1)[0]
            # stream_handler.text += f"\n\nFinal Answer: {actual_response}"
            # stream_handler.container.markdown(stream_handler.text)
            return actual_response
        else:
            error_response = f"An error occurred: {error_msg}"
            # stream_handler.text += f"\n\n{error_response}"
            # stream_handler.container.markdown(stream_handler.text)
            return error_response
        
def main():
    # Streamlit configuration
    st.set_page_config(page_title="Sales Inquiry Assistant", page_icon="💬")
    # Sidebar contents
    with st.sidebar:
        st.title('🤗💬 LLM Chat App')
        st.markdown('''
        ## About
        This app is an LLM-powered chatbot built using:
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        - [Ollama](https://ollama.com/) for Embeddings
    
        ''')
        ollama_models = [l['name'] for l in ollama.list()['models']]
        local_llm = st.selectbox("Select Ollama Model", ollama_models)
        # add_vertical_space(5)
        # st.write('Made with ❤️')
        # st.write('Made with ❤️ by [Pranay]()')

        

    llm = ChatOllama(model=local_llm,
                 keep_alive="3h", 
                 max_tokens=512,  
                 temperature=0)
    
    rag_chain = (
                        {"context": retriever, "question": RunnablePassthrough(),'chat_history':RunnablePassthrough()}
                        | prompt
                        | llm
                        | StrOutputParser()
                    )
    
    # Initialize the agent
    agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            agent_kwargs=agent_kwargs,
            handle_parsing_errors=True,
            early_stopping_method="generate",
            max_iterations=3
        )
    st.title("Sales Inquiry Assistant")
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    if len(msgs.messages) == 0:
        msgs.add_ai_message("How can I help you?")
    # view_messages = st.expander("View the message contents in session state")

    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)
    # user_question = st.text_input("Ask a question about sales, rebates, market share, promotions, or payouts:")
    # for chunk in :
        # print(chunk.content, end="", flush=True)
    # response = rag_chain.invoke(user_question)
    # st.write("Answer:")
    # st.write(response)
    # user_question = st.chat_input("Ask a question about sales, rebates, market share, promotions, or payouts:")
    # if user_question:
    #     st.session_state.chat_history.append({"role": "user", "content": user_question})

    # with st.chat_message("user"):
    #     st.markdown(user_question)
    if user_question := st.chat_input():
        msgs.add_user_message(user_question)
        st.chat_message("human").write(user_question)
        # Note: new messages are saved to history automatically by Langchain during run
        config = {"configurable": {"session_id": "any"}}
        # response = rag_chain.invoke(user_question)
        # msgs.add_ai_message(response)
        # st.chat_message("ai").write(response)

        ai_message = st.chat_message("ai")
        response_placeholder = ai_message.empty()  # Keeps the message box ready for real-time updates

        # config = {"configurable": {"session_id": "any"}}
        
        # Stream the response from the RAG chain
        partial_response = ""
        
        # for chunk in rag_chain.stream(user_question):
        
        # for chunk in execute_agent(agent, user_question):
        #     partial_response += chunk
        #     response_placeholder.write(partial_response)
        with st.spinner("Thinking..."):
            assistant_response = execute_agent(agent, user_question) 
        # assistant_response = StrOutputParser.parse(assistant_response)
        response_placeholder.write(assistant_response)
        msgs.add_ai_message(assistant_response)
        memory.save_context({"input":user_question}, {"output": partial_response})
    # with view_messages:
    #     """
    #     Message History initialized with:
    #     ```python
    #     msgs = StreamlitChatMessageHistory(key="langchain_messages")
    #     ```

    #     Contents of `st.session_state.langchain_messages`:
    #     """
    #     view_messages.json(st.session_state.langchain_messages)
    # with st.chat_message("assistant"):
    #     response = rag_chain.invoke(user_question)
    #     assistant_response =  response 
    #     st.markdown(assistant_response)
        # st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

# Remove the command-line interface
if __name__ == "__main__": 
    main()
