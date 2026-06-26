import os
import sys
import json
import difflib
from typing import Literal
import numpy as np
import ollama

# ── Tuneable constants ──────────────────────────────────────────────────────
CHUNK_SIZE          = 200     # words per chunk
CHUNK_OVERLAP       = 40      # words of overlap between adjacent chunks
TOP_K               = 5       # number of retrieved chunks to inject as context
EMBED_MODEL         = "snowflake-arctic-embed2"
LLM_MODEL           = "gemma4:12b"
LLM_TEMPERATURE     = 0.0     # 0.0 = deterministic output
LLM_NUM_PREDICT     = 4096    # max tokens for LLM response
LLM_THINK           = False

# ── Guard thresholds ────────────────────────────────────────────────────────
# Minimum cosine similarity for the best retrieved chunk.
# If the top-1 score falls below this, the LLM step is skipped entirely and
# the raw transcript is returned unchanged (Fix 3: score-gated correction).
SIMILARITY_THRESHOLD = 0.30

# Maximum fraction of words that may change between the raw transcript and the
# LLM-corrected output.  Corrections that exceed this are treated as
# hallucinations and the raw transcript is restored (Fix 5: edit-distance guard).
MAX_EDIT_RATIO = 1
# ────────────────────────────────────────────────────────────────────────────

# ── Language → corpus file mapping ──────────────────────────────────────────
LanguageID = Literal["ceb", "ilo", "hil", "war", "kap"]

CORPUS_MAP: dict[str, str] = {
    "ceb": "data/raw_text/cebuano_text.jsonl",
    "ilo": "data/raw_text/ilokano_text.jsonl",
    "hil": "data/raw_text/hiligaynon_text.jsonl",
    "war": "data/raw_text/waray_text.jsonl",
    "kap": "data/raw_text/kapampangan_text.jsonl",
}

_LANGUAGE_NAMES: dict[str, str] = {
    "ceb": "Cebuano",
    "ilo": "Ilocano",
    "hil": "Hiligaynon",
    "war": "Waray",
    "kap": "Kapampangan",
}

# Languages that should be explicitly excluded from LLM output to prevent
# the model from drifting toward its higher-resource training data (Fix 2).
_BANNED_LANGUAGES: dict[str, list[str]] = {
    "ceb": ["Tagalog", "Filipino", "Hiligaynon", "Waray"],
    "ilo": ["Tagalog", "Filipino", "Cebuano"],
    "hil": ["Tagalog", "Filipino", "Cebuano", "Waray"],
    "war": ["Tagalog", "Filipino", "Cebuano", "Hiligaynon"],
    "kap": ["Tagalog", "Filipino", "Cebuano"],
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
            start += chunk_size - overlap
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

        if index_dir is None:
            corpus_dir = os.path.dirname(os.path.abspath(corpus_path))
            index_dir  = os.path.normpath(os.path.join(corpus_dir, "..", "index"))
        self.index_dir = index_dir
        os.makedirs(self.index_dir, exist_ok=True)

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

        np.save(self._emb_path, self.embeddings)
        with open(self._chunk_path, "w", encoding="utf-8") as fh:
            json.dump(self.chunks, fh, ensure_ascii=False, indent=2)

        print(f"[RAG] Index saved → {self._emb_path}")
        print(f"[RAG]              → {self._chunk_path}")

    def _load_index(self) -> None:
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


# ── Fix 3 helper: per-sentence retrieval aggregation ─────────────────────────

def _retrieve_for_transcript(
    retriever: EmbeddingRetriever,
    transcript: str,
    top_k: int,
) -> list[tuple[str, float]]:
    """
    Split the transcript into sentences, retrieve top-2 chunks per sentence,
    deduplicate, and return the top-k by score.

    Long transcripts embed as a single dense vector that may not match any
    corpus chunk well.  Querying sentence-by-sentence produces much more
    targeted retrieval (Fix 4: query expansion).
    """
    # Simple sentence split on periods; handles most Waray / Philippine text.
    sentences = [s.strip() for s in transcript.replace("?", ".").replace("!", ".").split(".") if s.strip()]

    # Fall back to the full transcript if it is already a single short sentence.
    if not sentences:
        sentences = [transcript]

    seen:    set[str]               = set()
    results: list[tuple[str, float]] = []

    for sent in sentences:
        for chunk, score in retriever.retrieve(sent, top_k=2):
            if chunk not in seen:
                seen.add(chunk)
                results.append((chunk, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


# ── Fix 5 helper: word-level edit-distance ratio ─────────────────────────────

def _edit_distance_ratio(a: str, b: str) -> float:
    """
    Return the SequenceMatcher word-level similarity ratio between *a* and *b*.
    A ratio of 1.0 means identical; 0.0 means nothing in common.
    """
    return difflib.SequenceMatcher(None, a.split(), b.split()).ratio()


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
    3. Embed *raw_transcript* (sentence-by-sentence) and retrieve top-k chunks.
    4. Gate on retrieval confidence — skip LLM if score < SIMILARITY_THRESHOLD.
    5. Inject the retrieved chunks as context into the LLM correction prompt.
    6. Apply edit-distance guard — revert to raw if LLM diverges too much.

    Parameters
    ----------
    raw_transcript : str
        The raw output from Whisper (or any ASR engine).
    language : {"ceb", "ilo", "hil", "war", "kap"}
        BCP-47-style language ID that selects the corpus to retrieve from.
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
    if language not in CORPUS_MAP:
        raise ValueError(
            f"[RAG] Unknown language id '{language}'. "
            f"Valid options: {list(CORPUS_MAP.keys())}"
        )

    corpus_path   = CORPUS_MAP[language]
    language_name = _LANGUAGE_NAMES[language]
    banned_langs  = ", ".join(_BANNED_LANGUAGES.get(language, []))

    project_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    )
    corpus_path = os.path.normpath(os.path.join(project_root, corpus_path))
    print(f"[RAG] Language: {language_name} | Corpus: {corpus_path}")

    # ── Stage 1–2: Build / load index ────────────────────────────────────────
    try:
        retriever = EmbeddingRetriever(
            corpus_path=corpus_path,
            embed_model=embed_model,
            rebuild=rebuild,
        )
    except Exception as exc:
        print(f"[RAG] Warning: could not build retriever — {exc}")
        print("[RAG] Returning raw transcript unchanged.")
        return raw_transcript, "Retriever unavailable."

    # ── Stage 3: Per-sentence retrieval (Fix 4: query expansion) ─────────────
    try:
        results = _retrieve_for_transcript(retriever, raw_transcript, top_k)
    except Exception as exc:
        print(f"[RAG] Warning: retrieval failed — {exc}")
        results = []
    if not results:
        raise ValueError("[RAG] No retrieval results found.")

    best_score = results[0][1]
    print(f"[RAG] Best retrieval score: {best_score:.4f} (threshold: {SIMILARITY_THRESHOLD})")

    # ── Fix 3: Score-gated correction ────────────────────────────────────────
    if best_score < SIMILARITY_THRESHOLD:
        raise ValueError(
            f"[RAG] Retrieval confidence too low ({best_score:.4f} < {SIMILARITY_THRESHOLD})."
        )

    context_str = "\n".join(
        f"- [{score:.4f}] {chunk}" for chunk, score in results
    )

    # ── Stage 4: LLM correction with tightened prompt ────────────────────────
    messages = [
        {
            "role": "system",
            "content": (
                f"Role: You are a specialised ASR post-correction engine for "
                f"{language_name}, a Philippine language.\n"
                f"You must output ONLY {language_name} text.\n\n"
                "Success Criteria:\n"
                "- Correct phonetic transcription errors and misspellings using "
                "the reference passages supplied by the user.\n"
                "- Preserve original word order; change only words that are "
                "clearly wrong given the reference evidence.\n"
                "- Maintain the exact sentence count of the input.\n\n"
                "Constraints:\n"
                f"- CRITICAL: Never output {banned_langs} words. "
                f"If the correct {language_name} form cannot be confirmed from "
                "the reference passages, preserve the original word exactly.\n"
                "- Only substitute a word if the replacement appears VERBATIM "
                "in the reference passages. Do not infer, guess, or invent "
                "corrections — if no passage-grounded fix exists, leave the "
                "word as it is.\n"
                "- Do not add, remove, or reorder sentences.\n"
                "- Do not translate, summarise, or paraphrase.\n"
                "- Do not hallucinate words absent from both the transcript "
                "and the reference passages.\n\n"
                "Output Contract:\n"
                "Return only the corrected transcript text — no labels, "
                "no explanations, no quotation marks, no markdown."
            ),
        },
        {
            "role": "user",
            "content": (
                "Based on the reference passages below, correct the ASR transcript "
                "that follows.\n\n"
                f"Reference passages (format: [similarity] text):\n{context_str}\n\n"
                f"Example — wrong:   an bisita nga mga tawo nga naganhi\n"
                f"Example — correct: an bisita nga mga tawo nga nagaabot\n\n"
                f"Now correct this {language_name} transcript:\n{raw_transcript}"
            ),
        },
    ]

    try:
        response = ollama.chat(
            model=llm_model,
            messages=messages,
            think=LLM_THINK,
            options={
                "temperature": LLM_TEMPERATURE,
                "num_predict": LLM_NUM_PREDICT,
            },
        )

        thinking    = getattr(response.message, "thinking", None) or ""
        raw_content = response.message.content or ""
        print(f"[RAG-DEBUG] thinking length : {len(thinking)} chars")
        print(f"[RAG-DEBUG] content  length : {len(raw_content)} chars")

        content = raw_content.strip(' "\'\u2019\n')

        # ── Fix 5: Edit-distance guard ────────────────────────────────────────
        if content:
            similarity = _edit_distance_ratio(raw_transcript, content)
            change_fraction = 1.0 - similarity
            print(
                f"[RAG-DEBUG] edit change fraction: {change_fraction:.2f} "
                f"(max allowed: {MAX_EDIT_RATIO})"
            )
            if change_fraction > MAX_EDIT_RATIO:
                raise ValueError(
                    f"[RAG] Correction diverges too much "
                    f"({change_fraction:.0%} changed > {MAX_EDIT_RATIO:.0%} limit)."
                )
        else:
            raise ValueError("[RAG] LLM returned empty content.")

        return content, context_str

    except Exception as exc:
        if isinstance(exc, ValueError):
            raise exc
        raise RuntimeError(f"Ollama chat call failed — {exc}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rebuild_flag = "--rebuild" in sys.argv

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