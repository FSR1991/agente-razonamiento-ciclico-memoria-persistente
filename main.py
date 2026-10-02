import asyncio
import json
from typing import Any

from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from schemas import AgentState
from tools import TOOLS


class OfflineToolCallingModel(FakeMessagesListChatModel):
    """
    Modelo determinista utilizado exclusivamente para validar
    el ciclo ReAct sin consumir una API externa.

    Las respuestas contienen tool_calls predefinidos para demostrar:

    1. búsqueda inicial sin resultados;
    2. búsqueda refinada;
    3. consulta del detalle técnico;
    4. respuesta final;
    5. respuesta de una segunda interacción con el mismo thread_id.
    """

    def bind_tools(
        self,
        tools: Any,
        **kwargs: Any,
    ) -> "OfflineToolCallingModel":
        """Acepta las herramientas; los tool_calls ya están preparados."""
        return self


def build_test_graph() -> Any:
    """
    Construye una copia de prueba del StateGraph utilizando
    un modelo determinista y las mismas herramientas reales
    del agente.
    """

    model = OfflineToolCallingModel(
        responses=[
            AIMessage(
                content="Necesito buscar información técnica.",
                tool_calls=[
                    {
                        "id": "call-search-1",
                        "name": "search_knowledge_base",
                        "args": {
                            "query": "problema desconocido",
                            "limit": 3,
                        },
                    }
                ],
            ),
            AIMessage(
                content="La primera búsqueda no fue suficiente. Refinaré la consulta.",
                tool_calls=[
                    {
                        "id": "call-search-2",
                        "name": "search_knowledge_base",
                        "args": {
                            "query": "timeout conexiones PostgreSQL",
                            "limit": 3,
                        },
                    }
                ],
            ),
            AIMessage(
                content="Encontré un documento relevante. Consultaré su detalle.",
                tool_calls=[
                    {
                        "id": "call-detail-1",
                        "name": "get_knowledge_detail",
                        "args": {
                            "document_id": "KB-001",
                        },
                    }
                ],
            ),
            AIMessage(
                content=(
                    "El diagnóstico apunta a un posible agotamiento "
                    "del pool de conexiones PostgreSQL. Se recomienda "
                    "revisar el tamaño del pool, el tiempo de espera "
                    "y la cantidad de conexiones concurrentes."
                ),
            ),
            AIMessage(
                content=(
                    "Sí. La información anterior pertenece al mismo "
                    "hilo de conversación y permanece disponible "
                    "gracias al checkpoint persistente."
                ),
            ),
        ]
    )

    async def call_model(
        state: AgentState,
    ) -> dict[str, list[Any]]:
        """Ejecuta el modelo de prueba de forma asíncrona."""
        response = await model.ainvoke(
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

    return workflow


def serialize_message(message: Any) -> dict[str, Any]:
    """
    Convierte un mensaje de LangChain en una estructura JSON
    con información observable del ciclo ReAct.
    """

    data: dict[str, Any] = {
        "type": message.type,
        "content": message.content,
    }

    if getattr(message, "tool_calls", None):
        data["tool_calls"] = [
            {
                "id": tool_call["id"],
                "name": tool_call["name"],
                "args": tool_call["args"],
            }
            for tool_call in message.tool_calls
        ]

    if hasattr(message, "name") and message.name:
        data["name"] = message.name

    if hasattr(message, "tool_call_id") and message.tool_call_id:
        data["tool_call_id"] = message.tool_call_id

    return data


async def main() -> None:
    """
    Ejecuta dos interacciones utilizando el mismo thread_id
    para demostrar memoria persistente.
    """

    workflow = build_test_graph()

    thread_id = "support-thread-001"

    config = {
        "configurable": {
            "thread_id": thread_id,
        },
        "recursion_limit": 10,
    }

    async with AsyncSqliteSaver.from_conn_string(
        "agent_memory.sqlite"
    ) as memory:

        app = workflow.compile(
            checkpointer=memory
        )

        print("=" * 70)
        print("PRE-ENTREGA 5 - AGENTE DE RAZONAMIENTO CÍCLICO")
        print("=" * 70)

        print("\n[1] Primera interacción")
        print("-" * 70)

        first_input = {
            "messages": [
                HumanMessage(
                    content=(
                        "Tengo errores de timeout en conexiones "
                        "PostgreSQL. ¿Cómo puedo diagnosticarlos?"
                    )
                )
            ]
        }

        first_result = await app.ainvoke(
            first_input,
            config,
        )

        first_messages = first_result["messages"]

        print(
            "Mensajes acumulados:",
            len(first_messages),
        )

        print(
            "Última respuesta:",
            first_messages[-1].content,
        )

        print("\n[2] Segunda interacción con el mismo thread_id")
        print("-" * 70)

        second_input = {
            "messages": [
                HumanMessage(
                    content=(
                        "¿Y qué aspectos debería revisar "
                        "específicamente?"
                    )
                )
            ]
        }

        second_result = await app.ainvoke(
            second_input,
            config,
        )

        second_messages = second_result["messages"]

        print(
            "Mensajes acumulados después de la segunda interacción:",
            len(second_messages),
        )

        print(
            "Última respuesta:",
            second_messages[-1].content,
        )

        trace = {
            "project": (
                "Agente de razonamiento cíclico "
                "con memoria persistente"
            ),
            "thread_id": thread_id,
            "recursion_limit": 10,
            "persistence": "AsyncSqliteSaver",
            "database": "agent_memory.sqlite",
            "execution": {
                "first_interaction": {
                    "input": first_input["messages"][0].content,
                    "messages_after_execution": len(
                        first_messages
                    ),
                },
                "second_interaction": {
                    "input": second_input["messages"][0].content,
                    "messages_after_execution": len(
                        second_messages
                    ),
                },
            },
            "observable_react_trace": [
                serialize_message(message)
                for message in second_messages
            ],
        }

        with open(
            "trace.json",
            "w",
            encoding="utf-8",
        ) as trace_file:
            json.dump(
                trace_file if False else trace,
                trace_file,
                ensure_ascii=False,
                indent=2,
            )

        print("\n[3] Persistencia")
        print("-" * 70)
        print(
            "Thread utilizado:",
            thread_id,
        )
        print(
            "Trace guardado en: trace.json"
        )
        print(
            "SQLite utilizado en: agent_memory.sqlite"
        )

        print("\n" + "=" * 70)
        print("PRUEBA COMPLETADA")
        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())