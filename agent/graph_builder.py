"""
Полный файл с авто-регистрацией всех инструментов из каталога tools/.
"""

import os, importlib, pkgutil
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI

# ── ключ ────────────────────────────────────────────────────────────────
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY не найден! Проверь .env или переменные окружения.")

# ── авто-импорт инструментов ────────────────────────────────────────────
from tools import registry, __path__ as tools_path  # noqa: E402  (после установки PYTHONPATH)

for mod in pkgutil.iter_modules(tools_path):
    importlib.import_module(f"tools.{mod.name}")

# ── LLM ─────────────────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
)

# ── узлы графа ──────────────────────────────────────────────────────────
def planner(state: dict) -> dict:
    return {"action": "say_hello", "args": state["user_input"]}


def executor(state: dict) -> dict:
    tool_fn = registry.get(state["action"])
    if not tool_fn:
        return {"tool_output": f"⚠️ Неизвестный инструмент: {state['action']}"}
    return {"tool_output": tool_fn(state["args"])}


def responder(state: dict) -> dict:
    return {"response": state["tool_output"]}


# ── сборка графа ────────────────────────────────────────────────────────
def build_graph():
    g = StateGraph(input=dict, output=dict)
    g.add_node("planner", planner)
    g.add_node("executor", executor)
    g.add_node("responder", responder)

    g.set_entry_point("planner")
    g.add_edge("planner", "executor")
    g.add_edge("executor", "responder")
    g.add_edge("responder", END)
    return g.compile()
