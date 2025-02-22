import streamlit as st
import openai
import chromadb
from langchain_openai import OpenAIEmbeddings
from audio_recorder_streamlit import audio_recorder
import time
import librosa
import os

chroma_client = chromadb.PersistentClient(path="./knowledge_base")
collection_name = "FAQs"
collection = chroma_client.get_or_create_collection(collection_name)

def setup_openai(api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        client.models.list()
        return client
    except Exception as e:
        st.error(f"Invalid API Key: {e}")
        st.stop()

def retriever(query):
    embedding_model = OpenAIEmbeddings(api_key=api_key)
    query_embedding = embedding_model.embed_query(query)
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=10
    )
    return [doc for doc in results["documents"][0]] if results["documents"] else []

def fetch_ai_response(query):
    try:
        retrieved_results = retriever(query)
        context = "\n".join(retrieved_results) if retrieved_results else "No relevant information found."
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """You are Karthik Rajan’s personal AI assistant and your name is Karthik.If a user greets you (e.g., 'Hello', 'Hi', 'Hey'), respond politely.
                If a user asks who you are (e.g., 'Who are you?', 'Tell me about yourself', 'Introduce yourself'), respond with:
                'I am Karthik Rajan’s personal assistant. I can answer questions about him based on the information I have.'
                If a user asks what you can do (e.g., 'What can you do?', 'How can you help me?', 'What is your purpose?'), respond with:
                'I can provide information about Karthik Rajan. Feel free to ask about him.'Your purpose is to provide information about Karthik based on the given knowledge base.
                If a user asks about Karthik, respond using the available information."""},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"

def transcribe_audio(audio_path):
    with open(audio_path, "rb") as f:
        transcript = client.audio.transcriptions.create(model="whisper-1", file=f)
        return transcript.text

def text_to_audio(client, text, audio_path):
    response = client.audio.speech.create(model="tts-1", voice="echo", input=text)
    response.stream_to_file(audio_path)
    if os.path.exists(audio_path):
        audio_duration = librosa.get_duration(path=audio_path)
    else:
        audio_duration = 0 
    return audio_duration

def auto_play_audio(audio_file):
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()
    st.audio(audio_bytes, format="audio/mp3", autoplay=True)

st.set_page_config(page_title="AI Chatbot")


st.markdown("""
    <style>
        .chat-container {
        border: 2px solid #2196F3;
        border-radius: 10px;
        padding: 15px;
        max-width: 800px;
        margin: auto;
        background-color: #f9f9f9;
    }
     
    .message-container {
        display: flex;
        align-items: flex-start;
        margin: 10px 0;
        gap: 10px;
    }
    .message {
        padding: 10px 15px;
        border-radius: 15px;
        max-width: 70%;
    }
    .user-message {
        background-color: #DBE2E9;
        margin-left: auto;
    }
    .assistant-message {
        background-color:#E3F2FD;
    }
    .user-container {
        flex-direction: row-reverse;
    }
    .avatar {
        width: 35px;
        height: 35px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        flex-shrink: 0;
    }
    .user-avatar {
        background-color: #FF7F50;
        color: white;
    }
    .assistant-avatar {
        background-color: #2196F3;
        color: white;
    }
    .input-container {
        position: fixed;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 800px;
        background-color: white;
        padding: 10px;
        border-top: 1px solid #ddd;
        display: flex;
        align-items: center;
    }
    .stTextInput input {
        border-radius: 20px !important;
    }
    </style>
""", unsafe_allow_html=True)



st.title("🤖 KarryBot - A Personal Assistant")

api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

if not api_key:
    st.error("No API key provided. Please enter it in the sidebar.")
    st.stop()

client = setup_openai(api_key)

chat_container = st.container()

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.show_sample_questions = True 

sample_questions = [
    "Who are you?",
    "What is the age of karthik?",
    "What skills does Karthik have?",
    "What certifications or courses has Karthik completed?",
]    

if st.session_state.show_sample_questions and not st.session_state.messages:
    for question in sample_questions:
        if st.button(question):
            st.session_state.messages.append({"role": "user", "content": question})
            response = fetch_ai_response(question)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.show_sample_questions = False 
            st.rerun()

with chat_container:
    for msg in st.session_state.messages:
        message_class = "user-message" if msg["role"] == "user" else "assistant-message"
        container_class = "user-container" if msg["role"] == "user" else ""
        avatar_class = "user-avatar" if msg["role"] == "user" else "assistant-avatar"
        avatar_content = "👤" if msg["role"] == "user" else "🤖"
        
        st.markdown(
            f'''
            <div class="message-container {container_class}">
                <div class="avatar {avatar_class}">{avatar_content}</div>
                <div class="message {message_class}">{msg["content"]}</div>
            </div>
            ''',
            unsafe_allow_html=True
        )

st.markdown('<div class="input-container">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([0.12, 0.75, 0.13])

with col1:
    recorded_audio = audio_recorder(text="", pause_threshold=10.0)
with col2:
    user_input = st.text_input("Input Message: ", "", label_visibility="collapsed", placeholder="Enter your query here")
with col3:
    send_clicked = st.button("➤", help="Send", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)


if send_clicked:
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        response = fetch_ai_response(user_input)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

if recorded_audio:
    with st.spinner("Speaking...", show_time=True):
        audio_path = "input_audio.mp3"
        with open(audio_path, "wb") as f:
            f.write(recorded_audio)
        transcibed_text = transcribe_audio(audio_path)
        response_audio = fetch_ai_response(transcibed_text)
        response_audio_file = "audio_response.mp3"
        audio_duration=text_to_audio(client, response_audio, response_audio_file)
        auto_play_audio(response_audio_file)
        time.sleep(audio_duration)
        os.remove(audio_path)
        os.remove(response_audio_file)

        
    