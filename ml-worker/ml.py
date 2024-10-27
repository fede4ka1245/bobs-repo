from rag import get_context
from saiga import get_answer


def get_result(question: str, collection_name: str) -> str:
    context = get_context(question, collection_name=collection_name)
    return get_answer(question, context)
