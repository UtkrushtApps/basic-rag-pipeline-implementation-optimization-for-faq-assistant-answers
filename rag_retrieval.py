import time
from typing import List, Dict, Any, Optional
import tiktoken
from openai import OpenAIEmbeddings  # Assumed to be initialized elsewhere or replaced with actual usage

# Assume chromadb_client is injected or initialized elsewhere and passed to this module
# and FAQ embeddings with metadata are already inserted into ChromaDB as collection 'faq_chunks'.

# Dummy LLM function (to be replaced with actual OpenAI/LLM call)
def run_llm(prompt: str) -> str:
    # Placeholder for actual LLM API call
    # Should return a string answer truncated reasonably
    return "[LLM answer based on supplied prompt and context chunks]"

# Few-shot examples (could be loaded from a config or file)
FEW_SHOT_EXAMPLES = [
    {"question": "How can I reset my password?", "answer": "To reset your password, go to the login page and click 'Forgot Password'. Follow the instructions sent to your email. [1]"},
    {"question": "Where can I find the API documentation?", "answer": "You can find the API documentation at https://docs.example.com/api. [2]"},
]

SYSTEM_INSTRUCTIONS = """You are an FAQ assistant. Use the provided context to answer the user's question as accurately and concisely as possible. Always cite your information sources using bracketed numbers (e.g., [1], [2]) that refer to the provided context chunks. Do not use any information not present in the context."""

# Tokenizer setup (using tiktoken for OpenAI models like gpt-3.5-turbo)
tokenizer = tiktoken.get_encoding("cl100k_base")
MAX_PROMPT_TOKENS = 3500  # For prompt (context window + system prompt + few-shot + question)


def num_tokens_from_string(text: str) -> int:
    return len(tokenizer.encode(text))


def semantic_retrieve(
    chromadb_client,
    query: str,
    top_k: int = 6,
    category: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Embeds the input and does Chroma similarity search; applies category filter if given."""
    # Embed the query
    embed_model = OpenAIEmbeddings()  # Should be initialized per environment specs
    query_embedding = embed_model.embed_query(query)

    # Perform similarity search in ChromaDB
    collection = chromadb_client.get_collection("faq_chunks")
    # Chroma uses "where" for metadata filtering; here category
    where = {"category": category} if category else {}
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k * 2,  # get some headroom for later token filtering
        where=where
    )
    # Chroma returns:
    # {'ids': [[...]], 'documents': [[...]], 'metadatas': [[...]], 'distances': [[...]]}
    retrieved = []
    for idx in range(len(results['ids'][0])):
        retrieved.append({
            'id': results['ids'][0][idx],
            'content': results['documents'][0][idx],
            'metadata': results['metadatas'][0][idx],
            'distance': results['distances'][0][idx],
        })
    # Sort by distance ascending (higher similarity), just in case
    retrieved = sorted(retrieved, key=lambda x: x['distance'])
    return retrieved


def build_context_window(
    retrieved_chunks: List[Dict[str, Any]],
    prompt_budget: int,
) -> (str, List[Dict[str, Any]], List[int]):
    """Assembles context window string and per-chunk citation indices, obeying max token budget."""
    context_chunks = []
    token_count = 0
    citations = []  # Each will be (citation_idx, chunk)
    context_parts = []
    for idx, chunk in enumerate(retrieved_chunks):
        # Each chunk gets its own citation index (starting from 1)
        citation_idx = idx + 1
        context_str = f"[{citation_idx}] {chunk['content']}"  # Citation marker
        context_tokens = num_tokens_from_string(context_str)
        if token_count + context_tokens > prompt_budget:
            break  # Stop before exceeding budget
        token_count += context_tokens
        context_parts.append(context_str)
        citations.append({'citation_idx': citation_idx, 'chunk': chunk})
        context_chunks.append(chunk)
    context_window = "\n".join(context_parts)
    return context_window, citations, context_chunks


def build_fewshot_qna_examples() -> str:
    """Format few-shot Q&A examples for prompt."""
    formatted = []
    for ex in FEW_SHOT_EXAMPLES:
        formatted.append(f"Q: {ex['question']}\nA: {ex['answer']}")
    return "\n".join(formatted)


def assemble_prompt(
    user_query: str,
    context_window: str,
    few_shot_block: str,
) -> str:
    prompt_parts = [SYSTEM_INSTRUCTIONS]
    if context_window:
        prompt_parts.append("Context:\n" + context_window)
    if few_shot_block:
        prompt_parts.append(few_shot_block)
    prompt_parts.append(f"Q: {user_query}\nA:")
    prompt = "\n\n".join(prompt_parts)
    return prompt


def extract_citation_indices(answer: str) -> List[int]:
    """Extract citation indices (as ints) from LLM answer."""
    import re
    found = set(int(m) for m in re.findall(r'\[(\d+)\]', answer))
    return sorted(found)


def process_query(
    chromadb_client,
    user_query: str,
    top_k: int = 6,
    category: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main RAG pipeline: retrieve, context build, prompt, run LLM, citations, latency/token stats.
    Returns: {
        'answer': ...,
        'citations': [ ... ],
        'retrieval_latency': float,
        'prompt_tokens': int,
    }
    """
    timings = {}
    t0 = time.time()
    retrieved_chunks = semantic_retrieve(chromadb_client, user_query, top_k=top_k, category=category)
    t1 = time.time()
    timings['retrieval'] = t1 - t0

    # Budget prompt (system + few-shot + user question = X tokens)
    fewshot_block = build_fewshot_qna_examples()
    static_parts = [
        SYSTEM_INSTRUCTIONS,
        build_fewshot_qna_examples(),
        f"Q: {user_query}\nA:"
    ]
    static_token_budget = sum(num_tokens_from_string(p) for p in static_parts)
    context_token_budget = MAX_PROMPT_TOKENS - static_token_budget
    # Build context respecting budget
    context_window, citations, selected_chunks = build_context_window(retrieved_chunks, context_token_budget)

    # Assemble prompt
    prompt = assemble_prompt(user_query, context_window, fewshot_block)
    prompt_tokens = num_tokens_from_string(prompt)

    # Call LLM
    answer = run_llm(prompt)
    # Optionally, can post-process and truncate answer to output token budget if desired

    # Gather citation details
    cited_indices = extract_citation_indices(answer)
    cited_chunks = [  # Filter only cited
        {
            'citation_idx': c['citation_idx'],
            'id': c['chunk']['id'],
            'content': c['chunk']['content'],
            'metadata': c['chunk']['metadata'],
        }
        for c in citations
        if c['citation_idx'] in cited_indices
    ]
    result = {
        'answer': answer,
        'citations': cited_chunks,
        'retrieval_latency': timings['retrieval'],
        'prompt_tokens': prompt_tokens,
    }
    return result
