# Solution Steps

1. 1. Import required Python libraries: time, typing for types, tiktoken for token counting, OpenAI embeddings utility, and assume ChromaDB client is passed in.

2. 2. Set up system prompt and few-shot Q&A examples to provide instruction and format for the LLM. Use a concise system instruction and at least two representative few-shot Q&A pairs with citation markers.

3. 3. Set the maximum prompt token budget (3500 tokens). Initialize the OpenAI tokenizer for token counting.

4. 4. Write a utility function to count tokens in a text using tiktoken encoding.

5. 5. Implement the `semantic_retrieve` function: embed the user's query, do a vector similarity search against the ChromaDB 'faq_chunks' collection, apply optional 'category' filter in Chroma's metadata via 'where', retrieve extra chunks for budget cutoff, and return top results sorted by similarity.

6. 6. Implement `build_context_window`: iterate the top results, assign citation indices, construct context text for each chunk (with numbered marker), stop before exceeding the context token budget, and accumulate the used chunks and their citation indices.

7. 7. Write a function to format the few-shot Q&A examples as 'Q: …\nA: …' for prompt assembly.

8. 8. Implement `assemble_prompt`: put together the system instruction, the context window, the few-shot example block, and the user query into a single prompt string.

9. 9. Implement `extract_citation_indices` to parse the LLM response and list all citation indices (e.g., [1], [2]) used in the answer.

10. 10. Implement `process_query`: measure retrieval latency, run semantic retrieval and context window construction, assemble the prompt, call the LLM (placeholder function), extract citations, and return a dictionary with answer, cited chunk details, retrieval latency, and prompt token count.

11. 11. Return the fully implemented Python file `rag_retrieval.py` with the main pipeline and all helper functions, prepared for import and function calls by the infrastructure.

