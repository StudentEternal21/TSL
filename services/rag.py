import os
import sys
import json
from typing import Literal
import numpy as np
import ollama

# ── Tuneable constants ──────────────────────────────────────────────────────
CHUNK_SIZE      = 200                       # words per chunk
CHUNK_OVERLAP   = 40                        # words of overlap between adjacent chunks
TOP_K           = 5                         # number of retrieved chunks to inject as context
EMBED_MODEL     = "snowflake-arctic-embed2" # ollama embedding model tag
LLM_MODEL       = "gemma4:12b"             # ollama generation model tag
LLM_TEMPERATURE = 0.0                       # 0.0 = deterministic output
LLM_NUM_PREDICT = 4096                      # max tokens for LLM response (thinking + content)
LLM_THINK       = False                      # Gemma 4 thinking traces can easily exceed 1024
                                            # tokens; raise if content is consistently empty
# ────────────────────────────────────────────────────────────────────────────

# ── Language → corpus file mapping ──────────────────────────────────────────
LanguageID = Literal["ceb", "ilo", "hil", "war", "kap"]

CORPUS_MAP: dict[str, str] = {
    "ceb": "data/raw_text/cebuano_text.jsonl",
    "ilo": "data/raw_text/ilokano_text.jsonl",
    "hil": "data/raw_text/hiligaynon_text.jsonl",
    "war": "data/raw_text/waray_text.jsonl",
    "kap": "data/raw_text/kapamgpangan_text.jsonl",
}

_LANGUAGE_NAMES: dict[str, str] = {
    "ceb": "Cebuano",
    "ilo": "Ilocano",
    "hil": "Hiligaynon",
    "war": "Waray",
    "kap": "Kapampangan",
}
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
    language: LanguageID   = "war",
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
    language : {"ceb", "ilo", "hil", "war", "kap"}
        BCP-47-style language ID that selects the corpus to retrieve from.
        Must be one of the keys defined in ``CORPUS_MAP``.
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
    # Resolve corpus path from the language selection
    if language not in CORPUS_MAP:
        raise ValueError(
            f"[RAG] Unknown language id '{language}'. "
            f"Valid options: {list(CORPUS_MAP.keys())}"
        )
    corpus_path = CORPUS_MAP[language]
    language_name = _LANGUAGE_NAMES[language]

    project_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    )
    corpus_path = os.path.normpath(os.path.join(project_root, corpus_path))
    print(f"[RAG] Language: {language_name} | Corpus: {corpus_path}")

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
    #
    # Prompt structure follows Gemma 4 best-practice:
    #   • System turn  → contract-style sections (Role / Success Criteria /
    #                     Constraints / Output Contract)
    #   • User turn    → explicit anchoring phrase + context block + transcript
    #                     + one-shot example to lock the output format
    messages = [
        {
            "role": "system",
            "content": (
                # ── Role ──────────────────────────────────────────────────────
                f"Role: You are a specialised ASR post-correction engine for "
                f"{language_name}, a Philippine language.\n\n"
                # ── Success Criteria ──────────────────────────────────────────
                "Success Criteria:\n"
                "- Correct phonetic transcription errors and misspellings using "
                "the reference passages supplied by the user.\n"
                "- Preserve original word order; change only words that are "
                "clearly wrong given the reference evidence.\n"
                "- Maintain the exact sentence count of the input.\n\n"
                # ── Constraints ───────────────────────────────────────────────
                "Constraints:\n"
                "- Do not add, remove, or reorder sentences.\n"
                "- Do not translate, summarise, or paraphrase.\n"
                "- Do not hallucinate words absent from both the transcript and "
                "the reference passages.\n\n"
                # ── Output Contract ───────────────────────────────────────────
                "Output Contract:\n"
                "Return only the corrected transcript text — no labels, "
                "no explanations, no quotation marks, no markdown."
            ),
        },
        {
            "role": "user",
            "content": (
                # ── Context anchoring ─────────────────────────────────────────
                "Based on the reference passages below, correct the ASR transcript "
                "that follows.\n\n"
                f"Reference passages (format: [similarity] text):\n{context_str}\n\n"
                # ── One-shot format example ───────────────────────────────────
                # Note: use natural chat phrasing — NOT a completion-style
                # "Output:" suffix, which causes ollama.chat() to return an
                # empty string (the model treats the label as end-of-turn).
                "Example — wrong:   an bisita nga mga tawo nga naganhi\n"
                "Example — correct: an bisita nga mga tawo nga nagaabot\n\n"
                # ── Actual task ───────────────────────────────────────────────
                f"Now correct this transcript:\n{raw_transcript}"
            ),
        },
    ]

    try:
        response = ollama.chat(
            model=llm_model,
            messages=messages,
            think=LLM_THINK,    # chain-of-thought improves phonetic reasoning on low-resource languages
            options={
                "temperature": LLM_TEMPERATURE,
                "num_predict": LLM_NUM_PREDICT,
            },
        )

        # ── Debug: inspect what the model actually returned ───────────────
        thinking = getattr(response.message, "thinking", None) or ""
        raw_content = response.message.content or ""
        print(f"[RAG-DEBUG] thinking length : {len(thinking)} chars")
        print(f"[RAG-DEBUG] content  length : {len(raw_content)} chars")
        if raw_content.strip():
            print(f"[RAG-DEBUG] raw content     : {raw_content[:200]!r}")
        else:
            print(f"[RAG-DEBUG] content is EMPTY — checking thinking tail …")
            # Show last 300 chars of thinking so we can see if the answer
            # was placed inside the thinking block instead.
            print(f"[RAG-DEBUG] thinking tail   : …{thinking[-300:]!r}")

        content = raw_content.strip(' "\'\u2019\n')
        return content, context_str

    except Exception as exc:
        error_msg = f"ERROR: Ollama chat call failed — {exc}"
        print(error_msg)
        return error_msg, context_str


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rebuild_flag = "--rebuild" in sys.argv

    # Test case: Waray transcript with typical ASR phonetic confusion
    test_raw = "Pero wara igsumat kan Kim kun ano an imo a-aplayan kay hiring yana."
    print(f"Raw transcript : {test_raw}\n")

    corrected, retrieved_context = correct_transcript(
        test_raw,
        language="war",
        rebuild=rebuild_flag,
    )

    print("\n── Retrieved Context ────────────────────────────────────────────")
    print(retrieved_context)
    print("\n── Corrected Transcript ─────────────────────────────────────────")
    print(corrected)