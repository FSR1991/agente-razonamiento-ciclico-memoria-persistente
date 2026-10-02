from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class SearchKnowledgeInput(BaseModel):
    query: str = Field(
        description=(
            "Consulta técnica específica que se desea buscar "
            "en la base de conocimiento."
        )
    )
    limit: int = Field(
        default=3,
        ge=1,
        le=10,
        description=(
            "Cantidad máxima de resultados técnicos que se deben "
            "devolver."
        ),
    )


class GetKnowledgeDetailInput(BaseModel):
    document_id: str = Field(
        description=(
            "Identificador exacto del documento técnico cuyo "
            "contenido detallado se necesita consultar."
        )
    )


KNOWLEDGE_BASE: dict[str, dict[str, Any]] = {
    "KB-001": {
        "title": "Timeout en conexiones PostgreSQL",
        "content": (
            "Los errores de timeout pueden producirse cuando el "
            "pool de conexiones alcanza su límite. Se recomienda "
            "revisar el tamaño del pool, el tiempo de espera y "
            "la cantidad de conexiones concurrentes."
        ),
        "topic": "postgresql",
    },
    "KB-002": {
        "title": "Redis y pérdida de rendimiento",
        "content": (
            "Una configuración inadecuada de Redis puede generar "
            "latencia elevada. Se recomienda revisar memoria, "
            "políticas de expulsión y cantidad de conexiones."
        ),
        "topic": "redis",
    },
    "KB-003": {
        "title": "FastAPI y operaciones asíncronas",
        "content": (
            "Las operaciones de I/O bloqueantes dentro de endpoints "
            "asíncronos pueden degradar el rendimiento. Se recomienda "
            "utilizar operaciones compatibles con async/await."
        ),
        "topic": "fastapi",
    },
}


@tool(args_schema=SearchKnowledgeInput)
async def search_knowledge_base(
    query: str,
    limit: int = 3,
) -> list[dict[str, str]]:
    """
    Busca información técnica relevante en una base de conocimiento
    simulada.

    Utiliza esta herramienta cuando el usuario solicite diagnóstico,
    explicación o resolución de un problema relacionado con
    infraestructura, bases de datos, APIs, rendimiento o arquitectura
    de software.

    La consulta debe describir claramente el problema técnico.
    Devuelve identificadores de documentos que pueden consultarse
    posteriormente para obtener información detallada.

    No inventes resultados. Si la búsqueda no encuentra información
    suficientemente específica, realiza otra búsqueda con una consulta
    más precisa antes de responder al usuario.
    """
    normalized_query = query.lower()

    results: list[dict[str, str]] = []

    for document_id, document in KNOWLEDGE_BASE.items():
        searchable_text = (
            f"{document['title']} "
            f"{document['content']} "
            f"{document['topic']}"
        ).lower()

        if any(
            term in searchable_text
            for term in normalized_query.split()
            if len(term) > 2
        ):
            results.append(
                {
                    "document_id": document_id,
                    "title": document["title"],
                }
            )

        if len(results) >= limit:
            break

    return results


@tool(args_schema=GetKnowledgeDetailInput)
async def get_knowledge_detail(
    document_id: str,
) -> dict[str, str]:
    """
    Recupera el contenido técnico completo de un documento
    previamente identificado mediante search_knowledge_base.

    Utiliza esta herramienta cuando ya exista un document_id y sea
    necesario consultar sus detalles para construir una respuesta
    técnica fundamentada.

    Si el identificador no existe, devuelve un error estructurado
    indicando que el documento no fue encontrado.
    """
    document = KNOWLEDGE_BASE.get(document_id)

    if document is None:
        return {
            "error": (
                f"No se encontró el documento técnico "
                f"{document_id}."
            )
        }

    return {
        "document_id": document_id,
        "title": document["title"],
        "content": document["content"],
    }


TOOLS = [
    search_knowledge_base,
    get_knowledge_detail,
]