# Interactive Full-Stack AI Chatbot (Streamlit + LangChain)

A browser-based chat app with **streaming token output**, **chat message history**, and **session-state conversation memory**.

## Features
- `st.chat_message` / `st.chat_input` chat UI
- Streaming responses via `st.write_stream` and LangChain `chain.stream`
- Conversation memory in `st.session_state`, fed back to the model each turn
- Switchable provider: OpenAI, Anthropic, or local Ollama
- Optional context file upload (`.txt` / `.md`) to chat about a document
- Adjustable model, temperature, and system prompt; one-click clear chat

## Architecture
```
st.chat_input -> session_state.messages -> ChatPromptTemplate(system + history + question)
              -> LLM (streaming) -> StrOutputParser -> st.write_stream -> session_state
```
`build_chain()` in `app.py` is the single place to swap in your own RAG retriever or agent from earlier tasks.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # add your API key (or paste it in the sidebar)
streamlit run app.py
```
Open http://localhost:8501.

For a fully local setup, install [Ollama](https://ollama.com), run `ollama pull llama3.1`, and pick "Ollama (local)" in the sidebar.

## Project structure
```
app.py            # Streamlit app
requirements.txt
.env.example
.gitignore
```
