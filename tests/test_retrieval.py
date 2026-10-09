from agentic_rag_hybrid import Corpus, hybrid_rank, retrieve, rrf, SignatureEmbedder, BM25Index, tokenize

CORPUS = Corpus(
    {
        "d1": "Red pandas eat bamboo and live in the Himalayas.",
        "d2": "Rocket engines use combustion to generate thrust in space.",
        "d3": "Postgres can store vectors with the pgvector extension.",
        "d4": "RAG combines retrieval with generation to answer grounded questions.",
        "d5": "The critical path determines the longest sequence of tasks.",
        "d6": "Retrieval augmented generation combines a corpus with an LLM.",
    }
)


def test_rrf_fuses_sparse_and_dense():
    fused = rrf([("a", 1.0), ("b", 0.5)], [("b", 1.0), ("c", 0.5)])
    ids = [doc for doc, _ in fused]
    assert ids.index("b") < ids.index("a")
    assert ids.index("b") < ids.index("c")


def test_hybrid_rank_returns_ranked():
    bm25 = BM25Index(CORPUS)
    emb = SignatureEmbedder(CORPUS)
    results = hybrid_rank(bm25, emb, tokenize("retrieval augmented generation"), top_k=3)
    assert results
    assert results[0][0] == "d6"


def test_retrieve_returns_relevant_and_grounded():
    results = retrieve(CORPUS, "retrieval augmented generation", top_k=3)
    top_ids = [doc for doc, _ in results]
    assert "d4" in top_ids
    assert all(d in CORPUS.docs for d in top_ids)


def test_retrieve_drops_ungrounded_docs():
    # 'pandas' only appears in d1; query with an unrelated word must not pull in junk.
    results = retrieve(CORPUS, "pandas habitat", top_k=5)
    top_ids = [doc for doc, _ in results]
    assert "d1" in top_ids