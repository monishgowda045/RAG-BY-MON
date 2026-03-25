import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Load environment variables
load_dotenv()

# Initialize components
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chromadb_data")

def initialize_rag():
    """Initialize RAG system with embeddings and vector store."""
    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY
    )
    
    vectorstore = Chroma(
        collection_name="rag_documents",
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    
    return vectorstore, embeddings


def load_documents(file_path):
    """Load PDF documents."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    
    return chunks


def add_documents_to_vectorstore(vectorstore, documents):
    """Add documents to vector store."""
    vectorstore.add_documents(documents)
    vectorstore.persist()
    print(f"Added {len(documents)} document chunks to vector store")


def query_rag(vectorstore, question):
    """Query the RAG system."""
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-3.5-turbo",
        temperature=0.3,
    )

    retriever = vectorstore.as_retriever()
    
    # Retrieve relevant documents
    docs = retriever.invoke(question)
    
    # Format documents into context
    context = "\n".join([doc.page_content for doc in docs])
    
    # Create the prompt
    prompt_text = f"""Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"""
    
    # Get the answer from LLM
    response = llm.invoke(prompt_text)
    
    return response.content if hasattr(response, 'content') else str(response)


def main():
    """Main RAG application."""
    print("🚀 RAG System Initializing...\n")
    vectorstore, embeddings = initialize_rag()
    
    # Auto-ingest PDFs from data folder
    data_dir = "data"
    if os.path.exists(data_dir):
        pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
        if pdf_files:
            print(f"📁 Found {len(pdf_files)} PDF(s) in {data_dir}/\n")
            for pdf_file in pdf_files:
                pdf_path = os.path.join(data_dir, pdf_file)
                try:
                    print(f"📖 Ingesting: {pdf_file}...")
                    documents = load_documents(pdf_path)
                    add_documents_to_vectorstore(vectorstore, documents)
                    print(f"✅ Successfully ingested {len(documents)} chunks\n")
                except Exception as e:
                    print(f"❌ Error loading {pdf_file}: {e}\n")
        else:
            print(f"⚠️  No PDFs found in {data_dir}/ folder\n")
    else:
        print(f"⚠️  {data_dir}/ folder not found\n")
    
    # Interactive Q&A loop
    print("=" * 60)
    print("💬 Interactive RAG Q&A Mode (type 'exit' to quit)")
    print("=" * 60 + "\n")
    
    while True:
        try:
            question = input("📝 Your question: ").strip()
            if question.lower() in ["exit", "quit", "q"]:
                print("\n👋 Exiting RAG system...")
                break
            if not question:
                print("⚠️  Please enter a question\n")
                continue
            
            print("\n🔍 Searching documents...")
            answer = query_rag(vectorstore, question)
            print(f"\n✨ Answer:\n{answer}\n")
            print("-" * 60 + "\n")
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Exiting...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
