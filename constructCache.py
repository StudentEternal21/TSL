from services.rag import EmbeddingRetriever

for corpus in ["data/raw_text/waray_text.jsonl",
               "data/raw_text/cebuano_text.jsonl",
               "data/raw_text/hiligaynon_text.jsonl",
               "data/raw_text/ilocano_text.jsonl",
               "data/raw_text/kapangpangan_text.jsonl"]:
    EmbeddingRetriever(corpus_path=corpus, rebuild=False)

print("✅ Embedding complete!")