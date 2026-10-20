"""Interactive Full-Stack AI Chatbot with Streamlit + LangChain.

Features
- Streaming token output (st.write_stream)
- Chat message history rendered with st.chat_message / st.chat_input
- Conversation memory via st.session_state (history is fed back to the model)
- Pluggable LLM provider (OpenAI, Anthropic, or local Ollama)
- Optional document context: upload a .txt/.md file and chat about it
"""

import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

st.set_page_config(page_title="AI Chatbot", page_icon="💬", layout="centered")

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, concise AI assistant. Use the conversation history "
    "to stay consistent, and say so when you are unsure."
)

PROVIDERS = {
    # Free-tier providers (OpenAI-compatible APIs)
    "Groq (free)": {
        "default_model": "openai/gpt-oss-120b",
        "env_key": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
    },
    "Google Gemini (free)": {
        "default_model": "gemini-2.5-flash",
        "env_key": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
    "OpenAI": {"default_model": "gpt-4o-mini", "env_key": "OPENAI_API_KEY"},
    "Anthropic": {"default_model": "claude-sonnet-4-5", "env_key": "ANTHROPIC_API_KEY"},
    "Ollama (local)": {"default_model": "llama3.1", "env_key": None},
}


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
def init_state() -> None:
    """Create all session_state keys once per browser session."""
    st.session_state.setdefault("messages", [])  # list[{"role", "content"}]
    st.session_state.setdefault("doc_text", "")
    st.session_state.setdefault("doc_name", "")


def clear_chat() -> None:
    st.session_state.messages = []


# --------------------------------------------------------------------------
# LLM + chain
# --------------------------------------------------------------------------
def get_llm(
    provider: str,
    model: str,
    temperature: float,
    api_key: str | None,
    base_url: str | None = None,
):
    """Return a streaming LangChain chat model for the chosen provider."""
    if provider in ("OpenAI", "Groq (free)", "Google Gemini (free)"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            temperature=temperature,
            streaming=True,
            api_key=api_key or None,
            base_url=base_url,
        )
    if provider == "Anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model, temperature=temperature, streaming=True, api_key=api_key or None
        )
    from langchain_ollama import ChatOllama

    return ChatOllama(model=model, temperature=temperature)


def build_chain(llm, system_prompt: str, doc_text: str):
    """prompt | llm | parser. Swap this for your RAG/agent pipeline if desired."""
    system = system_prompt
    if doc_text:
        system += (
            "\n\nUse the following document as context when relevant. "
            "If the answer is not in it, say so.\n\n"
            "<document>\n{document}\n</document>"
        )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ]
    )
    return prompt | llm | StrOutputParser()


def to_lc_history(messages: list[dict]):
    """Convert stored dict messages to LangChain message objects."""
    out = []
    for m in messages:
        cls = HumanMessage if m["role"] == "user" else AIMessage
        out.append(cls(content=m["content"]))
    return out


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
init_state()

with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("Provider", list(PROVIDERS))
    cfg = PROVIDERS[provider]
    model = st.text_input("Model", value=cfg["default_model"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)

    api_key = None
    if cfg["env_key"]:
        env_val = os.getenv(cfg["env_key"], "")
        api_key = st.text_input(
            f"{cfg['env_key']}",
            value="",
            type="password",
            placeholder="Loaded from .env" if env_val else "Paste your key",
        ) or env_val

    system_prompt = st.text_area("System prompt", DEFAULT_SYSTEM_PROMPT, height=120)

    uploaded = st.file_uploader("Optional context (.txt / .md)", type=["txt", "md"])
    if uploaded is not None and uploaded.name != st.session_state.doc_name:
        st.session_state.doc_text = uploaded.read().decode("utf-8", errors="ignore")
        st.session_state.doc_name = uploaded.name
    elif uploaded is None and st.session_state.doc_name:
        st.session_state.doc_text = ""
        st.session_state.doc_name = ""

    st.button("🗑️ Clear conversation", on_click=clear_chat, use_container_width=True)
    st.caption(f"{len(st.session_state.messages)} messages in memory")

st.title("💬 AI Chatbot")
st.caption("Streaming responses · conversation memory · LangChain + Streamlit")

# Replay history on every rerun
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask me anything..."):
    # Snapshot history BEFORE adding the new turn (the prompt adds it itself)
    history = to_lc_history(st.session_state.messages)

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            llm = get_llm(provider, model, temperature, api_key, cfg.get("base_url"))
            chain = build_chain(llm, system_prompt, st.session_state.doc_text)
            inputs = {"history": history, "question": question}
            if st.session_state.doc_text:
                inputs["document"] = st.session_state.doc_text
            # write_stream renders tokens live and returns the full text
            answer = st.write_stream(chain.stream(inputs))
        except Exception as e:  # missing key, bad model name, network, etc.
            answer = f"⚠️ Something went wrong: `{e}`"
            st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})