import pandas as pd
import os
from langchain.llms import AzureOpenAI
from langchain.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from dotenv import load_dotenv
load_dotenv() # read local .env file
OPENAI_API_BASE = os.getenv('OPENAI_API_BASE')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_API_BASE = os.getenv('OPENAI_API_BASE')
OPENAI_API_VERSION = os.getenv('OPENAI_API_VERSION')
# # Create embeddingsclear
# embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=False)

# db = Chroma(persist_directory="./db-sales-enquiry",
#             embedding_function=embeddings)

embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-large",
    # dimensions: Optional[int] = None, # Can specify dimensions with new text-embedding-3 models
    azure_endpoint=OPENAI_API_BASE, #"https://<your-endpoint>.openai.azure.com/", If not provided, will read env variable AZURE_OPENAI_ENDPOINT
    api_key=OPENAI_API_KEY, # Can provide an API key directly. If missing read env variable AZURE_OPENAI_API_KEY
    openai_api_version=OPENAI_API_VERSION, # If not provided, will read env variable AZURE_OPENAI_API_VERSION
)

index_name = "faiss_index"
# Load documents from a directory
loader = DirectoryLoader("./input-data", glob="**/*.txt")

print("dir loaded loader")

documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n","\n"," ",""],
    chunk_size=1500,
    chunk_overlap=300,
    add_start_index=True,
)

# # Split documents into chunks
texts = text_splitter.split_documents(documents)
print(len(documents))
if os.path.exists(f"{index_name}.faiss") and os.path.exists(f"{index_name}.pkl"):
    vectorstore = FAISS.load_local(index_name, embeddings)
    print("Loaded existing FAISS index.")
else:
    # texts = df.to_string().split("\n")
    vectorstore = FAISS.from_texts(texts, embeddings)
    vectorstore.save_local(index_name)
    print("Created and saved new FAISS index.")