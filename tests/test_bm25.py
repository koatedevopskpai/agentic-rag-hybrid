from agentic_rag_hybrid import BM25Index, Corpus, tokenize

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


def test_bm25_scores_query_terms_highest():
    idx = BM25Index(CORPUS)
    best = idx.rank(tokenize("retrieval augmented generation"), top_k=3)
    assert best[0][0] == "d6"
    assert set(id_ for id_, _ in best) >= {"d4", "d6"}


def test_bm25_no_match_scores_zero():
    idx = BM25Index(CORPUS)
    scores = [idx.score(tokenize("quantum teleportation rocks"), doc) for doc in CORPUS.docs]
    assert max(scores) == 0.0