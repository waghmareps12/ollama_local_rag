import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain.chat_models import AzureChatOpenAI
from dotenv import load_dotenv
import os
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
import pandas as pd

load_dotenv()  # Load environment variables
OPENAI_API_TYPE = os.getenv('OPENAI_API_TYPE')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_API_BASE = os.getenv('OPENAI_API_BASE')
OPENAI_API_VERSION = os.getenv('OPENAI_API_VERSION')

df = pd.read_csv(r'./car_dummy.csv')
df_columns = df.columns


llm = AzureChatOpenAI(openai_api_base=OPENAI_API_BASE,
                        openai_api_version=OPENAI_API_VERSION,
                        openai_api_key=OPENAI_API_KEY,
                        openai_api_type=OPENAI_API_TYPE,
                        #deployment_name = 'genai-gpt-4-32k',
                        #model_name = 'gpt-4-32k')
                        deployment_name = 'genai-gpt-35-turbo',
                        model_name = 'gpt-35-turbo')

df_agent = create_pandas_dataframe_agent(llm, df, verbose=True,allow_dangerous_code=True,max_iterations = 3)

# Create embeddings for the vector store
embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=False)

# Load the saved index
loaded_vectorstore = FAISS.load_local("my_faiss_index", embeddings, allow_dangerous_deserialization=True)

# Create retriever
retriever = loaded_vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# Create prompt template
template = """<bos><start_of_turn>user
You are a sales inquiry assistant specializing in rebates, market share, promotions, and payouts. Answer the question based on the following context, including retrieved documents and relevant employee data when applicable. 
Write in full sentences with correct spelling and punctuation. Use bullet points or numbered lists when appropriate.
If the context doesn't contain the specific information, politely state that you don't have the answer and suggest contacting the sales department for further details.

CONTEXT: {context}

DATAFRAME INFO: {dataframe_info} 

dataframe columns = {df_columns}

QUESTION: {question}

<end_of_turn>
<start_of_turn>model
ANSWER:"""
prompt = ChatPromptTemplate.from_template(template)

# Function to check if a question is related to structured data in the DataFrame
def is_dataframe_question(query):
    related_keywords = ['rebate', 'market share', 'promotion', 'payout']
    return any(keyword in query.lower() for keyword in related_keywords)

# Main chatbot logic
def chatbot(query):
    # Retrieve relevant documents from the vector store (always needed)
    context = retriever.get_relevant_documents(query)
    context_str = "\n".join([doc.page_content for doc in context])

    # Check if the query is related to the pandas DataFrame
    if is_dataframe_question(query):
        # Query the pandas DataFrame agent
        dataframe_info = df_agent.run(query)
    else:
        # If it's not related to the DataFrame, leave dataframe_info empty
        dataframe_info = "No relevant structured data found."

    # Create input for the LLM with both context and DataFrame info
    final_input = template.format(context=context_str, dataframe_info=dataframe_info, question=query)

    # Get response from the LLM
    response = llm(final_input)
    return response

# Streamlit app interface
st.title("Sales Inquiry Assistant")

query = st.text_input("Ask your question:")
if query:
    response = chatbot(query)
    st.write(response)
