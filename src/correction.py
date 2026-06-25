import os
import json
import math
import string
import ollama

def load_jsonl_corpus(data_dir):
    """Loads all text documents from .jsonl files in the specified directory."""
    corpus = []
    if not os.path.exists(data_dir):
        return corpus
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".jsonl"):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if "text" in data:
                                corpus.append(data["text"])
                        except json.JSONDecodeError:
                            continue
    return corpus

def tokenize(text):
    """Tokenizes and normalizes text by lowercasing and removing punctuation."""
    text = text.lower()
    # Remove punctuation
    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator)
    return text.split()

class TFIDFRetriever:
    def __init__(self, corpus):
        self.corpus = corpus
        self.doc_tokens = [tokenize(doc) for doc in corpus]
        self.num_docs = len(corpus)
        
        # Calculate Term Frequency (TF) and Document Frequency (DF)
        self.df = {}
        for tokens in self.doc_tokens:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.df[token] = self.df.get(token, 0) + 1
        
        # Calculate Inverse Document Frequency (IDF)
        self.idf = {}
        for token, count in self.df.items():
            self.idf[token] = math.log((1 + self.num_docs) / (1 + count)) + 1
            
        # Calculate TF-IDF vectors for documents
        self.doc_vectors = []
        for tokens in self.doc_tokens:
            vector = self._vectorize(tokens)
            self.doc_vectors.append(vector)

    def _vectorize(self, tokens):
        """Helper to vectorize tokens into a TF-IDF dictionary."""
        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        
        vector = {}
        for token, count in tf.items():
            if token in self.idf:
                vector[token] = count * self.idf[token]
        return vector

    def _cosine_similarity(self, vec1, vec2):
        """Computes cosine similarity between two sparse TF-IDF vectors."""
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum(vec1[x] * vec2[x] for x in intersection)
        
        sum1 = sum(val ** 2 for val in vec1.values())
        sum2 = sum(val ** 2 for val in vec2.values())
        denominator = math.sqrt(sum1) * math.sqrt(sum2)
        
        if not denominator:
            return 0.0
        return numerator / denominator

    def retrieve(self, query, top_k=3):
        """Retrieves top_k documents most similar to the query."""
        query_tokens = tokenize(query)
        query_vector = self._vectorize(query_tokens)
        
        if not query_vector:
            # Fallback if no words match: return first few documents
            return [(self.corpus[i], 0.0) for i in range(min(top_k, self.num_docs))]
            
        scores = []
        for i, doc_vector in enumerate(self.doc_vectors):
            score = self._cosine_similarity(query_vector, doc_vector)
            scores.append((self.corpus[i], score))
            
        # Sort by similarity score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

def correct_transcript(raw_transcript, corpus_dir="data", model="qwen3.5:0.8b"):
    """Corrects ASR transcript using RAG LLM."""
    # 1. Load corpus
    corpus = load_jsonl_corpus(corpus_dir)
    if not corpus:
        print("Warning: Empty or missing corpus. Proceeding without RAG context.")
        context_str = "No reference texts available."
    else:
        # 2. Retrieve top matching documents
        retriever = TFIDFRetriever(corpus)
        results = retriever.retrieve(raw_transcript, top_k=3)
        context_str = "\n".join([f"- {doc}" for doc, score in results if score > 0.0])
        if not context_str:
            context_str = "\n".join([f"- {doc}" for doc, _ in results[:2]])

    # 3. Formulate Prompt
    prompt = f"""You are an ASR transcript error correction assistant.
Correct misspelt words, typos, and phonetic errors in the RAW TRANSCRIPT using the REFERENCE TEXTS context.

REFERENCE TEXTS:
{context_str}

RAW TRANSCRIPT TO CORRECT:
{raw_transcript}

Corrected transcript:"""

    # 4. Generate correction via local LLM
    response = ollama.generate(
        model=model,
        prompt=prompt
    )
    
    return response.response.strip(), context_str

if __name__ == "__main__":
    # Test case: Cebuano transcript with typical ASR phonetic confusion/typo
    # "Maayong bontag sa inyong tanan." -> Should match "Maayong buntag sa inyong tanan."
    test_raw = "Maayong bontag sa inyong tanan."
    print(f"Raw transcript: {test_raw}\n")
    
    corrected, retrieved_context = correct_transcript(test_raw, corpus_dir="data")
    
    print("Retrieved Context:")
    print(retrieved_context)
    print("\nCorrected transcript:")
    print(corrected)