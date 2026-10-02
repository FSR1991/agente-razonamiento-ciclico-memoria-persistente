# Agente de razonamiento cíclico con memoria persistente

Agente técnico construido con Python 3.12, LangGraph y herramientas personalizadas. El sistema implementa un ciclo de razonamiento basado en mensajes y herramientas, con recuperación de información, memoria persistente mediante SQLite y ejecución asíncrona.

## Objetivo

Construir un agente capaz de:

- interpretar una consulta técnica;
- decidir cuándo utilizar herramientas;
- ejecutar búsquedas técnicas;
- refinar una búsqueda cuando la información inicial es insuficiente;
- consultar el detalle de un documento técnico;
- regresar al nodo del agente después de cada herramienta;
- conservar el historial de conversación mediante un `thread_id`;
- persistir checkpoints localmente en SQLite;
- limitar la profundidad del ciclo para evitar loops infinitos.

## Tecnologías

- Python 3.12
- LangGraph
- LangChain
- LangChain OpenAI
- LangChain Anthropic
- Pydantic
- python-dotenv
- SQLite
- aiosqlite
- AsyncSqliteSaver
- asyncio

## Arquitectura

```mermaid
flowchart TD
    A[Usuario] --> B[AgentState / MessagesState]
    B --> C[Agent Node]
    C --> D{tools_condition}
    D -->|tool call| E[ToolNode]
    D -->|respuesta final| F[END]
    E --> G[Resultado de herramienta]
    G --> C
    C -. checkpoint .-> H[(SQLite)]
    H -. thread_id .-> B