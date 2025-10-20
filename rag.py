import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI
from langchain.chains import RetrievalQA

# Set your Azure OpenAI credentials (replace with your actual values)
endpoint = os.getenv("PLAYGROUND_AOAI_ENDPOINT")
api_token = os.getenv("PLAYGROUND_LLM_API_TOKEN")

# Step 1: Load the PDF and extract text
loader = PDFPlumberLoader("./20250910_AI_Report.pdf")  # Replace with your PDF file path
documents = loader.load()

# Step 2: Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Chunk size in characters (adjust as needed)
    chunk_overlap=200  # Overlap to maintain context
)
chunks = text_splitter.split_documents(documents)

# Step 3: Embed the chunks
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Step 4: Create a vector store from the embedded chunks
vectorstore = FAISS.from_documents(chunks, embeddings)

# Step 5: Set up the LLM and the retrieval chain
llm = AzureChatOpenAI(
    openai_api_key='no-need-api-key-but-never-remove-this-line',
    azure_endpoint=endpoint,
    azure_deployment="gpt-4o-mini",
    api_version="2024-10-21",
    default_headers={
        "Authorization": f"Basic {api_token}"
    }
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Simple chain type that stuffs context into the prompt
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 relevant chunks
)

# Step 6: Perform a simple LLM query
query = "What is the main topic of the document? Response in Korean."  # Replace with your actual query
result = qa_chain.invoke({"query": query})

# Output the result
print("Query:", query)
print("Response:", result["result"])