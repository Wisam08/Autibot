from langchain.tools import Tool
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import  DuckDuckGoSearchRun
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.tools.tavily_search import TavilySearchResults
# TAVILY_API_KEY="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"
# search = TavilySearchResults(max_results=2, tavily_api_key=TAVILY_API_KEY)
# Wikipedia Search
wikipedia = WikipediaAPIWrapper()
wiki_tool = Tool(
    name="Wikipedia Search",
    func=wikipedia.run,
    description="Use this tool for general autism-related knowledge and definitions."
)

# ArXiv Research
arxiv = ArxivAPIWrapper()
arxiv_tool = Tool(
    name="ArXiv Research",
    func=arxiv.run,
    description="Use this tool to find the latest autism research studies."
)

# DuckDuckGoSearchRun
search=DuckDuckGoSearchRun(name="Search")# 
# search_tool = Tool(
#     name="search",
#     func=search.run,
#     description="Use this tool for real-time autism news and verified sources."
# )

# # PDF Retrieval (for academic papers)
# def query_pdfs(query):
#     loader = PyPDFLoader("autism_research.pdf")  # Load the autism research PDF
#     docs = loader.load()

#     # Convert PDF text into searchable format
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vectorstore = FAISS.from_documents(docs, embeddings)

#     results = vectorstore.similarity_search(query)  # Search for query in PDF
#     return results[0].page_content if results else "No relevant information found."

# pdf_tool = Tool(
#     name="Autism Research PDF",
#     func=query_pdfs,
#     description="Use this tool to extract information from autism-related research PDFs."
# )

# List of tools for chatbot to use
tools = [wiki_tool, arxiv_tool,search]
