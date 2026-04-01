import math

QUERY_INTENT_VOCAB = {
    "nav": 0,
    "info": 1,
    "qa": 2,
    "transaction": 3,
    "UNK": 4,
}

DOC_TYPE_VOCAB = {
    "natural": 0,
    "sc": 1,
    "UNK": 2,
}

DENSE_FEATURE_NAMES = [
    "query_len",
    "query_freq_log",
    "doc_quality",
    "doc_authority",
    "doc_freshness",
    "title_overlap",
    "content_overlap",
    "bm25_score_norm",
    "semantic_score",
    "is_exact_match",
    "field_match_score",
]


def safe_float(x, default=0.0):
    if x is None:
        return default
    return float(x)


def encode_query_intent(x):
    return QUERY_INTENT_VOCAB.get(x, QUERY_INTENT_VOCAB["UNK"])


def encode_doc_type(x):
    return DOC_TYPE_VOCAB.get(x, DOC_TYPE_VOCAB["UNK"])


def build_dense_features(sample):
    query_len = safe_float(sample.get("query_len"), 0.0)
    query_freq = safe_float(sample.get("query_freq"), 1.0)
    query_freq_log = math.log(query_freq + 1.0)

    doc_quality = safe_float(sample.get("doc_quality"), 0.5)
    doc_authority = safe_float(sample.get("doc_authority"), 0.5)
    doc_freshness = safe_float(sample.get("doc_freshness"), 0.5)
    title_overlap = safe_float(sample.get("title_overlap"), 0.5)
    content_overlap = safe_float(sample.get("content_overlap"), 0.5)
    bm25_score = safe_float(sample.get("bm25_score"), 8.0)
    semantic_score = safe_float(sample.get("semantic_score"), 0.5)
    is_exact_match = safe_float(sample.get("is_exact_match"), 0.0)
    field_match_score = safe_float(sample.get("field_match_score"), 0.5)

    bm25_score_norm = bm25_score / 16.0

    feats = [
        query_len,
        query_freq_log,
        doc_quality,
        doc_authority,
        doc_freshness,
        title_overlap,
        content_overlap,
        bm25_score_norm,
        semantic_score,
        is_exact_match,
        field_match_score,
    ]
    return feats


def get_dense_feature_dim():
    return len(DENSE_FEATURE_NAMES)
