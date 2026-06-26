import os
import sys
import json
import numpy as np
import ollama

# ── Tuneable constants ──────────────────────────────────────────────────────
CHUNK_SIZE    = 200   # words per chunk
CHUNK_OVERLAP = 40    # words of overlap between adjacent chunks
TOP_K         = 3     # number of retrieved chunks to inject as context
EMBED_MODEL   = "snowflake-arctic-embed2"   # ollama embedding model tag
LLM_MODEL     = "gemma4:12b"               # ollama generation model tag
# ────────────────────────────────────────────────────────────────────────────


# ── Stage 1: Corpus loading & chunking ──────────────────────────────────────

def load_jsonl_corpus(data_path: str) -> list[str]:
    """
    Loads all text documents from .jsonl files in the specified directory
    or a single file.  Each JSON object is expected to have a ``"text"`` key.
    """
    corpus: list[str] = []
    if not os.path.exists(data_path):
        return corpus

    files = (
        [data_path]
        if os.path.isfile(data_path)
        else [
            os.path.join(data_path, f)
            for f in os.listdir(data_path)
            if f.endswith(".jsonl")
        ]
    )

    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "text" in data:
                        corpus.append(data["text"])
                except json.JSONDecodeError:
                    continue
    return corpus


def chunk_corpus(
    corpus: list[str],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Splits each document into overlapping word-window chunks.

    Word-level windows are used deliberately — sentence-boundary heuristics
    tend to break on Cebuano/Waray text.  Overlap ensures that details
    spanning a chunk boundary are not missed during retrieval.
    """
    chunks: list[str] = []
    for doc in corpus:
        words = doc.split()
        if len(words) <= chunk_size:
            chunks.append(doc)
            continue
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start += chunk_size - overlap   # slide forward by (chunk_size - overlap)
    return chunks


# ── Stage 2: Embedding index ─────────────────────────────────────────────────

class EmbeddingRetriever:
    """
    Dense-embedding RAG retriever backed by a local Ollama embedding model.

    Pipeline
    --------
    Build (first run or ``rebuild=True``)
        1. Load corpus from ``corpus_path``.
        2. Chunk every document with ``chunk_corpus()``.
        3. Call ``ollama.embed()`` for every chunk → float32 matrix.
        4. Persist matrix as ``<stem>_embeddings.npy`` and chunk texts as
           ``<stem>_chunks.json`` inside ``index_dir``.

    Load (subsequent runs)
        Skip steps 1–4 and load from disk — startup is then near-instant.

    Retrieve
        Embed the query at runtime, compute vectorised cosine similarity
        against the stored matrix, return top-k (chunk, score) pairs.

    Parameters
    ----------
    corpus_path : str
        Path to a ``.jsonl`` file (or directory of ``.jsonl`` files).
    index_dir : str | None
        Where to store the cached index.  Defaults to
        ``<corpus_dir>/../index/``.
    embed_model : str
        Ollama embedding model tag.
    rebuild : bool
        Discard any cached index and re-embed from scratch.
    """

    def __init__(
        self,
        corpus_path: str,
        index_dir: str | None = None,
        embed_model: str = EMBED_MODEL,
        rebuild: bool = False,
    ) -> None:
        self.corpus_path = corpus_path
        self.embed_model = embed_model

        # Derive cache directory next to the corpus file
        if index_dir is None:
            corpus_dir  = os.path.dirname(os.path.abspath(corpus_path))
            index_dir   = os.path.normpath(os.path.join(corpus_dir, "..", "index"))
        self.index_dir = index_dir
        os.makedirs(self.index_dir, exist_ok=True)

        # Stable, corpus-specific file names inside the cache directory
        stem = os.path.splitext(os.path.basename(corpus_path))[0]
        self._emb_path   = os.path.join(self.index_dir, f"{stem}_embeddings.npy")
        self._chunk_path = os.path.join(self.index_dir, f"{stem}_chunks.json")

        if rebuild or not self._index_exists():
            self._build_index()
        else:
            self._load_index()

    # ── Index management ─────────────────────────────────────────────────────

    def _index_exists(self) -> bool:
        return (
            os.path.isfile(self._emb_path)
            and os.path.isfile(self._chunk_path)
        )

    def _build_index(self) -> None:
        """Chunk the corpus, embed every chunk, and persist to disk."""
        print(f"[RAG] Building embedding index from: {self.corpus_path}")
        corpus = load_jsonl_corpus(self.corpus_path)
        if not corpus:
            raise FileNotFoundError(
                f"No text documents found at corpus path: {self.corpus_path}"
            )

        self.chunks: list[str] = chunk_corpus(corpus)
        total = len(self.chunks)
        print(f"[RAG] Embedding {total} chunks with '{self.embed_model}' …")

        raw_embeddings: list[list[float]] = []
        for i, chunk in enumerate(self.chunks, start=1):
            resp = ollama.embed(model=self.embed_model, input=chunk)
            raw_embeddings.append(resp.embeddings[0])
            if i % 500 == 0 or i == total:
                print(f"[RAG]   {i}/{total} chunks embedded")

        self.embeddings: np.ndarray = np.array(raw_embeddings, dtype=np.float32)

        # Persist
        np.save(self._emb_path, self.embeddings)
        with open(self._chunk_path, "w", encoding="utf-8") as fh:
            json.dump(self.chunks, fh, ensure_ascii=False, indent=2)

        print(f"[RAG] Index saved → {self._emb_path}")
        print(f"[RAG]              → {self._chunk_path}")

    def _load_index(self) -> None:
        """Load the pre-built index from disk (fast path)."""
        print(f"[RAG] Loading cached embedding index …")
        self.embeddings = np.load(self._emb_path)
        with open(self._chunk_path, "r", encoding="utf-8") as fh:
            self.chunks = json.load(fh)
        print(f"[RAG] Loaded {len(self.chunks)} chunks from cache.")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[tuple[str, float]]:
        """
        Embed *query* and return the *top_k* most similar corpus chunks.

        Returns
        -------
        list[tuple[str, float]]
            ``(chunk_text, cosine_similarity)`` pairs, highest score first.
        """
        resp = ollama.embed(model=self.embed_model, input=query)
        query_vec = np.array(resp.embeddings[0], dtype=np.float32)

        # Vectorised cosine similarity against all stored chunk embeddings
        # scores = (E @ q) / (||E|| * ||q||)
        chunk_norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm  = np.linalg.norm(query_vec)

        with np.errstate(divide="ignore", invalid="ignore"):
            scores = np.where(
                (chunk_norms * query_norm) == 0.0,
                0.0,
                (self.embeddings @ query_vec) / (chunk_norms * query_norm),
            )

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices]


# ── Stages 3 & 4: Context injection + LLM correction ─────────────────────────

def correct_transcript(
    raw_transcript: str,
    corpus_path: str       = "data/raw_text/waray_text.jsonl",
    llm_model: str         = LLM_MODEL,
    embed_model: str       = EMBED_MODEL,
    top_k: int             = TOP_K,
    rebuild: bool          = False,
) -> tuple[str, str]:
    """
    Corrects an ASR transcript using the four-step RAG pipeline.

    Steps
    -----
    1. Chunk the corpus (word-window, with overlap).
    2. Embed every chunk via *embed_model* (result cached to disk).
    3. Embed *raw_transcript* and retrieve top-k chunks by cosine similarity.
    4. Inject the retrieved chunks as context into the LLM correction prompt.

    Parameters
    ----------
    raw_transcript : str
        The raw output from Whisper (or any ASR engine).
    corpus_path : str
        Path to the ``.jsonl`` corpus — may be relative to the project root.
    llm_model : str
        Ollama generation model tag used for transcript correction.
    embed_model : str
        Ollama embedding model tag used for indexing and query embedding.
    top_k : int
        Number of retrieved chunks to include in the LLM context.
    rebuild : bool
        Force a full re-index of the corpus.

    Returns
    -------
    (corrected_text, retrieved_context_string)
    """
    # Resolve relative path relative to the *project root* (one level up from src/)
    if not os.path.isabs(corpus_path):
        project_root = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        )
        corpus_path = os.path.normpath(os.path.join(project_root, corpus_path))

    # Stage 1–3: Retrieve relevant context chunks
    try:
        retriever = EmbeddingRetriever(
            corpus_path=corpus_path,
            embed_model=embed_model,
            rebuild=rebuild,
        )
        results = retriever.retrieve(raw_transcript, top_k=top_k)
        context_str = "\n".join(
            f"- [{score:.4f}] {chunk}" for chunk, score in results
        )
        if not context_str.strip():
            context_str = "No relevant passages found in corpus."
    except Exception as exc:
        print(f"[RAG] Warning: retrieval failed — {exc}")
        print("[RAG] Proceeding without retrieved context.")
        context_str = "No reference context available."

    # Stage 4: LLM correction — inject retrieved context before the instruction
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert ASR transcript correction tool specialised in "
                "Philippine languages (Waray, Cebuano, Tagalog). "
                "Fix spelling and phonetic errors in the user's transcript using "
                "the retrieved reference passages as domain-specific context. "
                "CRITICAL: Output ONLY the corrected transcript text. "
                "Do not include explanations, labels, or quotation marks."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Reference passages retrieved from the text corpus "
                f"(format: [similarity] text):\n{context_str}\n\n"
                f"ASR transcript to correct:\n{raw_transcript}"
            ),
        },
    ]

    try:
        response = ollama.chat(
            model=llm_model,
            messages=messages,
            think=True,    # chain-of-thought improves phonetic reasoning on low-resource languages
            options={
                "temperature": 0.0,
                "num_predict": 512,   # must cover thinking tokens + actual output
            },
        )
        content = response.message.content.strip(' "\'\u2019\n')
        return content, context_str

    except Exception as exc:
        error_msg = f"ERROR: Ollama chat call failed — {exc}"
        print(error_msg)
        return error_msg, context_str


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rebuild_flag = "--rebuild" in sys.argv

    # Test case: Waray transcript with typical ASR phonetic confusion
    test_raw = "Paado kun diri bukad-bukad it' igsul-ot, a-absonan na la kimo?"
    print(f"Raw transcript : {test_raw}\n")

    corrected, retrieved_context = correct_transcript(
        test_raw,
        corpus_path="data/raw_text/waray_text.jsonl",
        rebuild=rebuild_flag,
    )

    print("\n── Retrieved Context ────────────────────────────────────────────")
    print(retrieved_context)
    print("\n── Corrected Transcript ─────────────────────────────────────────")
    print(corrected)