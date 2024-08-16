
# "🦙 Llamoney" – a Financial Advisor AI!

A financial advisory tool powered by state-of-the-art AI models. The tool aims to provide accurate, reliable, and actionable financial advice by leveraging some powerful technologies.

### 🔍 **Key Highlights:**
- **Use Cases**: From answering complex financial queries to analyzing market trends, this tool serves as a virtual financial advisor capable of assisting with investment strategies, market analysis, and company performance evaluations.
- **Libraries & Models**: 
  - Utilized **Hugging Face Transformers** to load and deploy the **Meta-Llama-3.1-8B-Instruct** model.
  - Integrated **LlamaIndex** for efficient document indexing and querying.
  - Employed **Gradio** for creating an interactive user interface.
  - Used **LangChain** for embedding-based document search.
- **Techniques**: 
  - Implemented 4-bit quantization using **BitsAndBytesConfig** to optimize model inference on GPU/CPU.
  - Leveraged **SentenceSplitter** to handle large documents with intelligent text chunking for better context management.

### 🛠️ **Future Work:**
- Fine-tuning the model specifically on financial data and news articles to enhance the reliability and relevance of the generated content.
- Expanding the tool's capabilities to include real-time financial news analysis, portfolio management recommendations, and personalized investment advice.
