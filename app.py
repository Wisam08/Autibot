from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
#from unsloth import FastLanguageModel
from langchain.agents import initialize_agent, AgentType
from langchain_groq import ChatGroq
from tools import tools
import torch
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import os
from dotenv import load_dotenv
load_dotenv()
import torch
# Detect GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
# ✅ Use a sentence embedding model for FAISS
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Initialize FAISS with an empty document store
dummy_doc = Document(page_content="")
memory_store = FAISS.from_documents([dummy_doc], embedding_model)

# ✅ Initialize LangChain memory
memory = ConversationBufferMemory(return_messages=True)

app = FastAPI()

# Load Fine-Tuned TinyLlama Model
model_path = "auti_model2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu")#FastLanguageModel.from_pretrained(model_path, max_seq_length=512, dtype=None, load_in_4bit=True)

# Define Request Model
class ChatRequest(BaseModel):
    user_message: str

# Define Expanded Prompt Template
PROMPT_TEMPLATE = """
<s>[INST]You are an AI autism expert designed to provide accurate, research-backed, and empathetic responses related to autism spectrum disorder (ASD). Your goal is to assist users by offering insights into autism diagnosis, early signs, intervention strategies, therapies, and general support for individuals, caregivers, and professionals. You must respond in a friendly, patient, and non-judgmental manner, ensuring clarity and simplicity in your explanations.

Your responses should be backed by the latest scientific research and practical guidance. If a question is outside the domain of autism, politely decline to answer. Avoid giving medical diagnoses and instead encourage consulting a healthcare professional when necessary.

Your tone should be warm and supportive, making the user feel comfortable and understood. Use positive and empowering language when discussing autism and neurodiversity. Adapt responses based on whether the user is a parent, teacher, healthcare provider, or an autistic individual seeking information.
If you don’t know something, say: 'I do not have enough verified information to answer this.'  

If a question is not related to autism, say: 'I am here to discuss autism-related topics.'

Question: {question} [/INST]
"""


# # Initialize LangChain Agent
# llm = ChatGroq(model_name="gpt-4-turbo", temperature=0)

## load the groq api
# groq_api_key=os.getenv("GROQ_API_KEY")
# os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="llama3-8b-8192",groq_api_key="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")

agent = initialize_agent(
    tools=tools,  
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True  
)

def check_memory(query):
    """Retrieve relevant past conversations for context."""
    if len(memory_store.index_to_docstore_id) <= 1:
        return "No relevant memory found."

    docs = memory_store.similarity_search(query, k=3)
    seen = set()
    unique_docs = []
    for doc in docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc.page_content)

    return "\n".join([doc.page_content for doc in docs]) if docs else "No relevant memory found."

# ✅ Function to Store Chat History
def save_to_memory(query, response):
    """Only store responses related to autism in FAISS memory."""
    if not query or not response:
        return  

    # ✅ Prevent storing irrelevant responses
    if "I am a musician" in response or "I am a beginner in Python" in response:
        return  

    new_doc = Document(page_content=f"User: {query}\nBot: {response}")
    memory_store.add_documents([new_doc])

    print(f"Saved: {query} → {response}")  # Debugging log


@app.post("/chat")
async def chat(request: ChatRequest):
    query = request.user_message.strip()  # Remove extra spaces

    # ✅ Retrieve relevant memory
    context = check_memory(query)

    # ✅ Format input to avoid irrelevant memory
    if "No relevant memory found." in context:
        full_query = f"{query}"  # Only user query if no relevant memory
    else:
        full_query = f"Context: {context}\n\nUser: {query}"

    # ✅ Tokenize and move to device
    input_ids = tokenizer(full_query, return_tensors="pt").input_ids.to(device)
    model.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids, 
            max_new_tokens=100, 
            do_sample=True, 
            temperature=0.7,  # ✅ Encourages diverse responses
            top_k=50,  # ✅ Limits repeated low-probability tokens
            top_p=0.9,  # ✅ Nucleus sampling for diversity
            repetition_penalty=1.2  # ✅ Penalizes repeated phrases
        )

    # ✅ Extract bot response properly
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True).split("User:")[-1].strip()

    # ✅ If response is off-topic, force it back to autism
    if "I am a musician" in response or "I am a beginner in Python" in response:
        response = "I am here to discuss autism-related topics."

    # ✅ Store conversation in memory
    save_to_memory(query, response)

    # ✅ If response is unclear, use LangChain Agent
    if "I do not have enough verified information" in response:
        #I do not have enough verified information to answer this
        response = agent.run(query)

    return {"response": response}


# @app.post("/chat")
# async def chat(request: ChatRequest):
#     query = PROMPT_TEMPLATE.format(question=request.user_message)

#     # Step 1: Try LLM response
#     input_ids = tokenizer(query, return_tensors="pt").input_ids.to("cuda")
#     model.to(device)
#     with torch.no_grad():
#         output_ids = model.generate(input_ids, max_length=200, do_sample=True)
#     response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

#     # Step 2: If response is unclear, use tools
#     if "I do not have enough verified information" in response:
#         response = agent.run(request.user_message)

#     return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# from fastapi import FastAPI
# from pydantic import BaseModel
# from workflow import chatbot  # Import the LangGraph chatbot workflow

# app = FastAPI()

# # Define Request Model
# class ChatRequest(BaseModel):
#     user_message: str

# @app.post("/chat")
# async def chat(request: ChatRequest):
#     response = chatbot.invoke({"query": request.user_message})
#     return {"response": response["response"]}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

