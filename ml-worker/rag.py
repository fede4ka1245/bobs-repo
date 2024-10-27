import torch
import os

from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from transformers import BertForMaskedLM, AutoTokenizer


class CustomEmbedding(Embeddings):
    def __init__(self, directory: str = "models"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = BertForMaskedLM.from_pretrained(directory)
        self.tokenizer = AutoTokenizer.from_pretrained(directory)
        self.model.to(self.device)

    def embed_query(self, text: str) -> list[float]:
        return extract_features(text, self.model, self.tokenizer).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return list(map(lambda text: self.embed_query(text), texts))


def extract_features(text, model, tokenizer):
    model.eval()

    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=512,
        chunk_overlap=64
    )

    chunks = text_splitter.split_text(text)

    features = []

    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]

        attention_mask = inputs['attention_mask']
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
        mean_embeddings = sum_embeddings / torch.clamp(mask_expanded.sum(1), min=1e-9)

        features.append(mean_embeddings)

    return torch.stack(features).mean(dim=0).squeeze(0).cpu()


def get_context(question: str, k: int = 10, collection_name: str = os.getenv("QDRANT_COLLECTION_NAME")) -> list[str]:
    client = QdrantClient(url=os.getenv("QDRANT_URL"))
    embedding = CustomEmbedding()

    qdrant = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding,
    )

    return [i.page_content for i in qdrant.similarity_search(question, k=k)]
