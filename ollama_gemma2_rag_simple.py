import streamlit as st
# from streamlit_extras.add_vertical_space import add_vertical_space
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# # Create embeddingsclear
embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=False)

db = Chroma(persist_directory="./db-sales-enquiry",
            embedding_function=embeddings)

# # Create retriever
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs= {"k": 5}
)

# # Create Ollama language model - Gemma 2
local_llm = 'llama3.2:latest'

llm = ChatOllama(model=local_llm,
                 keep_alive="3h", 
                 max_tokens=512,  
                 temperature=0)

# # Create prompt template
template = """<bos><start_of_turn>user
You are a sales inquiry assistant specializing in rebates, market share, promotions, and payouts. Answer the question based only on the following context and provide a meaningful response. 
Write in full sentences with correct spelling and punctuation. Use bullet points or numbered lists when appropriate.
If the context doesn't contain the answer, politely state that you don't have the specific information and suggest contacting the sales department for more details.

CONTEXT: {context}

QUESTION: {question}

<end_of_turn>
<start_of_turn>model
ANSWER:"""
prompt = ChatPromptTemplate.from_template(template )

# # Modify the rag_chain to return the result instead of streaming
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# qa_chain = RetrievalQA.from_chain_type(
#     llm, retriever=retriever, chain_type_kwargs={"prompt": prompt}
# )

# # Streamlit UI


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
        - [Ollama](https://ollama.com/) LLM model
    
        ''')
        # add_vertical_space(5)
        st.write('Made with ❤️')
        # st.write('Made with ❤️ by [Pranay](https://youtube.com/@engineerprompt)')
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
        
        for chunk in rag_chain.stream(user_question):
            partial_response += chunk
            response_placeholder.write(partial_response)
        msgs.add_ai_message(partial_response)
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
