import os
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from schemas import AgentState
from tools import TOOLS


load_dotenv()


def create_model() -> Any:
    """Crea el LLM configurado mediante variables de entorno."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "openai":
        model_name = os.getenv("OPENAI_MODEL")

        if not model_name:
            raise ValueError(
                "OPENAI_MODEL no está configurado."
            )

        return ChatOpenAI(
            model=model_name,
            temperature=0,
        )

    if provider == "anthropic":
        model_name = os.getenv("ANTHROPIC_MODEL")

        if not model_name:
            raise ValueError(
                "ANTHROPIC_MODEL no está configurado."
            )

        return ChatAnthropic(
            model=model_name,
            temperature=0,
        )

    raise ValueError(
        "LLM_PROVIDER debe ser 'openai' o 'anthropic'."
    )


model = create_model()

model_with_tools = model.bind_tools(TOOLS)


async def call_model(
    state: AgentState,
) -> dict[str, list[Any]]:
    """
    Ejecuta el modelo con el historial actual y devuelve
    el nuevo mensaje generado por el agente.
    """
    response = await model_with_tools.ainvoke(
        state["messages"]
    )

    return {"messages": [response]}


tool_node = ToolNode(TOOLS)


workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "tools",
        END: END,
    },
)

workflow.add_edge("tools", "agent")


@asynccontextmanager
async def create_app():
    """
    Crea el grafo compilado utilizando SQLite como
    almacenamiento persistente de checkpoints.
    """
    async with AsyncSqliteSaver.from_conn_string(
        "agent_memory.sqlite"
    ) as memory:
        app = workflow.compile(
            checkpointer=memory
        )

        yield app