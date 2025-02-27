# KarryBot - A Personal AI Assistant

KarryBot is a personal AI assistant built using Streamlit, OpenAI, ChromaDB, and LangChain. It allows users to interact via text and voice, retrieve relevant information from a knowledge base, and generate AI-powered responses.

![image](https://github.com/user-attachments/assets/3ed65c98-1936-4385-90fd-9306ba09baf8)

Watch the Working Video at here - https://drive.google.com/file/d/190MCRHqQPdA5Tjyz_gYWhwp9IcpKwdAn/view?usp=drivesdk

## Features
- **Conversational AI**: KarryBot answers queries based on a stored knowledge base.
- **Voice Input & Output**: Users can record voice input and receive AI-generated audio responses.
- **Knowledge Base Retrieval**: Uses ChromaDB for semantic search and retrieval.
- **Streamlit UI**: Provides a clean and interactive chat interface with sample questions.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.12.7
- Streamlit
- OpenAI SDK
- ChromaDB
- LangChain
- Audio Recorder for Streamlit
- Librosa (for audio duration calculation)

### Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/karrybot.git
   cd karrybot
   ```
2. Install dependencies:
   ```sh
   pip install streamlit openai chromadb langchain-openai audio-recorder-streamlit librosa
   ```
3. Set up your OpenAI API key in `.streamlit/secrets.toml`:
   ```toml
   [secrets]
   openai_key = "your_openai_api_key"
   ```

## Usage
1. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```
2. Interact with KarryBot by typing a message or using voice input.
3. View AI-generated responses in the chat interface.

## File Structure
```
karrybot/
├── app.py                 # Main Streamlit app
├── knowledge_base/        # Stores the ChromaDB knowledge base
├── requirements.txt       # List of dependencies
├── .streamlit/secrets.toml # OpenAI API key configuration
```

## How It Works
- **Text Input Handling**: Users enter text queries, which are processed by OpenAI's GPT-3.5 Turbo model.
- **Voice Input Handling**: Users can record their voice, which is transcribed using OpenAI's Whisper model.
- **Context Retrieval**: Queries are searched in ChromaDB to find relevant information.
- **AI Response Generation**: The chatbot generates responses based on retrieved context and predefined instructions.
- **Voice Output**: The assistant can convert text responses into speech using OpenAI's text-to-speech API.

## Customization
- Modify the `fetch_ai_response()` function to customize bot responses.
- Update the knowledge base stored in ChromaDB for improved responses.
- Adjust UI elements in `app.py` to enhance user experience.


