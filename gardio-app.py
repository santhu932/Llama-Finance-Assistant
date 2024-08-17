import gradio as gr
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextStreamer, BitsAndBytesConfig
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, set_global_service_context
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.callbacks import TokenCountingHandler, CallbackManager
import shutil
import glob

# Setting up environment variables for Hugging Face API tokens
os.environ["HF_TOKEN"] = ""
os.environ["HUGGINGFACEHUB_API_TOKEN"]  = ""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
current_index = None
upload_folder = './data'


# Function to delete all files in a given directory
def delete_all_files_in_directory(directory_path):
    files = glob.glob(os.path.join(directory_path, '*'))
    for file in files:
        try:
            os.remove(file)
        except Exception as e:
            print(f"Error deleting {file}: {e}")

# Function to load the tokenizer and model with specified configurations
def get_model_and_tokenizer():
    # Configuration for 4-bit quantization to reduce model size and improve inference speed
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load the tokenizer from the Hugging Face model repository
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./model/')
    tokenizer.pad_token = tokenizer.eos_token

    # Load the model with quantization configuration for efficient inference
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, cache_dir='./model/', quantization_config=bnb_config)
    
    return tokenizer, model

# Define a system prompt that sets the behavior and tone of the model's responses
system_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a highly knowledgeable and experienced financial advisor. 
    Your role is to provide insightful, accurate, and actionable financial advice. 
    Always answer as helpfully and thoroughly as possible, while maintaining a strong emphasis 
    on safety and reliability. Your responses should be free from any harmful, unethical, 
    racist, sexist, toxic, dangerous, or illegal content, and should reflect a deep 
    understanding of financial markets, company performance, and investment strategies.
    
    Ensure that your advice is socially unbiased and positive in nature. If a question is 
    unclear or lacks factual coherence, clarify the misunderstanding rather than providing 
    an incorrect answer. If you don't know the answer 
    to a question, please don't share false information
    
    Your primary goal is to deliver expert guidance on financial performance, investment 
    opportunities, and other related financial queries, supporting users in making 
    informed decisions.<|eot_id|>
    """

# Wrapper prompt that will be used for querying the model
query_wrapper_prompt = SimpleInputPrompt("<|begin_of_text|>{query_str}")

# Load the tokenizer and model using the get_model_and_tokenizer function
tokenizer, model = get_model_and_tokenizer()

# Initialize the HuggingFaceLLM with the loaded model, tokenizer, and prompt settings
llm = HuggingFaceLLM(context_window=4096,
                    max_new_tokens=256,
                    system_prompt=system_prompt,
                    query_wrapper_prompt=query_wrapper_prompt,
                    model=model,
                    tokenizer=tokenizer)

# Set up the embedding model using a pre-trained model from Hugging Face
embedding_model=LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

# Define transformations for processing the text data
transformations = [SentenceSplitter(chunk_size=1024, chunk_overlap=50)]

# Apply settings for LLM, embedding model, and text transformations
Settings.llm = llm
Settings.embed_model = embedding_model
Settings.transformations = transformations

# Function to generate a response based on a user-provided prompt and uploaded documents
def generate_response(prompt, files):
    global current_index
    
    if files is not None:
        # Create the upload folder if it doesn't exist
        if not os.path.exists(upload_folder):
            os.mkdir(upload_folder)

        # Copy uploaded files to the upload folder
        for file in files:
            shutil.copy(file, upload_folder)

        # Load data from the uploaded files and create an index
        documents = SimpleDirectoryReader(upload_folder).load_data()
        current_index = VectorStoreIndex.from_documents(documents)

        # Delete the uploaded files after processing
        delete_all_files_in_directory(upload_folder)
        
    if current_index is None:
        return "No document has been uploaded. Please upload a document to start querying."

    # Query the index and generate a response using the LLM
    query_engine = current_index.as_query_engine()
    response = query_engine.query(prompt)
    
    return response.response

# Set up the Gradio interface for user interaction
interface = gr.Interface(
    fn=generate_response,
    inputs=[gr.Textbox(label="Input your prompt here"), gr.Files(label="Upload a PDF documents")],
    outputs=["text"],
    title="🦙 Llamoney",
    description="Input your prompt and get a response."
)

# Launch the Gradio interface, allowing sharing via the internet
interface.launch(share=True)




    


