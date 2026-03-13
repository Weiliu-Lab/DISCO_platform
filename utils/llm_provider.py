import os

from langchain_openai import ChatOpenAI

ACTIVE_LLM_PROVIDER = "gemini"
ACTIVE_LLM_MODEL = "gemini-2.5-pro"

DEFAULT_DEEPSEEK_API_KEY = "sk-618adc94f39d445aa32d5ebe9ca78cf8"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_GEMINI_API_KEY = "AIzaSyCYVyfVnVzCC4Xs0WMYRfuB8R-hmkkPJcs"
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def get_active_model_name() -> str:
    return os.environ.get("ACTIVE_LLM_MODEL", ACTIVE_LLM_MODEL)


def get_langchain_llm(temperature: float = 0.1):
    provider = (os.environ.get("ACTIVE_LLM_PROVIDER") or ACTIVE_LLM_PROVIDER).strip().lower()
    model = os.environ.get("ACTIVE_LLM_MODEL") or ACTIVE_LLM_MODEL

    if provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY") or DEFAULT_DEEPSEEK_API_KEY
        os.environ["DEEPSEEK_API_KEY"] = api_key
        return ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            openai_api_base=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL),
            temperature=temperature,
        )

    api_key = os.environ.get("GEMINI_API_KEY") or DEFAULT_GEMINI_API_KEY
    os.environ["GEMINI_API_KEY"] = api_key
    return ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        openai_api_base=os.environ.get("GEMINI_BASE_URL", DEFAULT_GEMINI_BASE_URL),
        temperature=temperature,
    )