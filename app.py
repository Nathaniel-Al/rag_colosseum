"""RAG service: POST /query. Hybrid dense + IDF-lexical retrieval, extractive answers,
abstention, prompt-injection stripping. Dense model is optional (falls back to TF-IDF)."""
import asyncio, json, math, os, re
from collections import Counter
from functools import lru_cache

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

KB = json.load(open(os.path.join(os.path.dirname(__file__), "knowledge_base.json")))
N = len(KB)

# ---------- text utils ----------
STOP = set("""a an the of to in on for and or is are was were be been what which who whom how does do did
by with at as from that this these those it its into over under than then there their about per
default configured configure standard value used use uses using specified maximum minimum number
tell me please give state report according passage knowledge base compare difference many much has have can could would should between""".split())
def stem(w):
    if len(w) > 4 and w.endswith("s") and not w.endswith("ss"): w = w[:-1]
    return w[:5] if len(w) > 5 else w
def toks(t): return [stem(w) for w in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", t.lower()) if w not in STOP]
def sents(p): return [s.strip() for s in re.split(r"(?<=[a-z0-9])\.\s+(?=[A-Z])", p) if s.strip()]

PASS_TOK = [set(toks(p)) for p in KB]
DF = Counter(w for s in PASS_TOK for w in s)
def idf(w): return math.log((N + 1) / (DF.get(w, 0) + 0.5))

SENTS = [sents(p) for p in KB]
SENT_TOK = [[set(toks(s)) for s in ss] for ss in SENTS]

# ---------- dense embeddings (optional) ----------
ENC = None
try:
    from sentence_transformers import SentenceTransformer
    ENC = SentenceTransformer(os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    P_EMB = ENC.encode(KB, normalize_embeddings=True)
    flat = [(i, j, s) for i, ss in enumerate(SENTS) for j, s in enumerate(ss)]
    S_EMB = ENC.encode([s for _, _, s in flat], normalize_embeddings=True)
    S_IDX = [(i, j) for i, j, _ in flat]
except Exception as e:  # offline / not installed
    print("dense model unavailable, lexical-only:", e)

def dense(q):
    """returns (passage_sims[N], {(i,j): sim}) or (None, None)"""
    if ENC is None: return None, None
    v = ENC.encode([q], normalize_embeddings=True)[0]
    ps = P_EMB @ v
    ss = S_EMB @ v
    return ps, {k: float(x) for k, x in zip(S_IDX, ss)}

ALIAS = {"sgd":"stochastic gradient descent","mlp":"multi-layer perceptron","svm":"support vector machine",
 "lsm":"log-structured merge tree","wal":"write-ahead logging","mdns":"multicast dns","bgp":"border gateway protocol",
 "ospf":"open shortest path first","dns":"domain name system","tcp":"transmission control protocol",
 "tls":"transport layer security","hlc":"hybrid logical clock","2pc":"two-phase commit","3pc":"three-phase commit",
 "pkce":"proof key for code exchange","jwt":"json web token","aes":"advanced encryption standard",
 "ecc":"elliptic curve cryptography","waf":"web application firewall","mvcc":"multi-version concurrency control",
 "cnn":"residual convolutional network","resnet":"residual convolutional network","bn":"batch normalization",
 "ln":"layer normalization","lr":"learning rate","rsa":"rsa public key cryptography","ecdsa":"elliptic curve ecdsa"}
def expand(q):
    for k, v in ALIAS.items():
        q = re.sub(r"\b%s\b" % re.escape(k), v, q, flags=re.I)
    return q

# ---------- injection handling ----------
INJ = re.compile(r"(ignore|disregard|forget|override)[^.?!\n]{0,40}(instruction|prompt|rule|above|previous|context)|"
                 r"(system|developer)\s*(prompt|message|note)|you are now|new instructions?|"
                 r"(reveal|print|repeat|output)[^.?!\n]{0,30}(prompt|instructions|secret|key)|"
                 r"respond (only )?with|answer (only )?with|\bact as\b|\bpretend\b|"
                 r"do not (cite|use) (the )?(sources|passages)|</?\w+>|\[(?:system|inst)\]", re.I)
def sanitize(q):
    parts = re.split(r"(?<=[.?!])\s+|\n+", q)
    kept = [p for p in parts if not INJ.search(p)]
    clean = " ".join(kept).strip()
    return clean if clean else "", len(kept) != len(parts)

# ---------- scoring ----------
COMPARE = re.compile(r"\b(compare|versus|vs\.?|difference|differ|both|and how|combined|together|sum|total|"
                     r"respectively|relationship|contrast)\b", re.I)

def lex_scores(qt):
    qw = set(qt)
    tot = sum(idf(w) for w in qw) or 1.0
    out = np.zeros(N)
    for i in range(N):
        out[i] = sum(idf(w) for w in qw if w in PASS_TOK[i]) / tot
    return out, qw, tot

def score_passages(q):
    qt = toks(q)
    lex, qw, tot = lex_scores(qt)
    ps, ss = dense(q)
    if ps is not None:
        d = np.clip(ps, 0, 1)
        comb = 0.55 * d + 0.45 * lex
    else:
        d = None; comb = lex
    return comb, lex, d, ss, qw, tot

NUMQ = True
def best_sentence(i, qw, ss):
    best, bs = 0, -1
    # sentence-level IDF: words present in most sentences of this passage (its topic) count less
    ns = len(SENT_TOK[i])
    def w(x): return idf(x) * (1.0 - 0.6 * sum(x in st for st in SENT_TOK[i]) / ns)
    den = sum(w(x) for x in qw) or 1
    for j, st in enumerate(SENT_TOK[i]):
        l = sum(w(x) for x in qw if x in st) / den
        s = 0.5 * l + (0.5 * ss[(i, j)] if ss else 0.5 * l)
        if NUMQ and re.search(r"\b\d", SENTS[i][j]): s += 0.15
        if s > bs: bs, best = s, j
    return best, bs

def numeric_sentence(i, qw, ss):
    """best-scoring sentence in passage i that contains a number"""
    best, bs = None, -1
    for j, st in enumerate(SENT_TOK[i]):
        if not re.search(r"\b\d", SENTS[i][j]): continue
        s = sum(idf(x) for x in qw if x in st) + (ss[(i, j)] if ss else 0)
        if s > bs: bs, best = s, j
    return best

ABSTAIN_COV = float(os.getenv("ABSTAIN_COV", "0.40"))
ABSTAIN_SCORE = float(os.getenv("ABSTAIN_SCORE", "0.30"))

@lru_cache(maxsize=2048)
def answer_query(question: str, top_k: int):
    q, injected = sanitize(question)
    q = expand(q) if q else q
    top_k = max(1, min(int(top_k or 3), 10))
    if not q:
        return dict(answer=None, sources=[], scores=[], confidence=0.05)
    comb, lex, d, ss, qw, tot = score_passages(q)
    order = list(np.argsort(-comb))
    top = order[0]
    # best-sentence coverage of the query's content words (abstention signal)
    j, sscore = best_sentence(top, qw, ss)
    sent_cov = sum(idf(w) for w in qw if w in SENT_TOK[top][j]) / tot
    pass_cov = lex[top]
    # specific (rare) query words missing from the top passage => likely unanswerable
    missing = [w for w in qw if w not in PASS_TOK[top] and DF.get(w, 0) == 0 and len(w) > 2]
    unans = (pass_cov < ABSTAIN_COV) or (comb[top] < ABSTAIN_SCORE) or (len(missing) >= 1 and pass_cov < 0.85)
    if d is not None and d[top] < 0.30: unans = True
    if unans:
        return dict(answer=None, sources=[], scores=[], confidence=round(min(0.35, float(comb[top])), 4))

    # cross-boundary synthesis: pull in second passage when asked to compare/combine or scores are close
    chosen = [top]
    sec = order[1]
    multi = bool(COMPARE.search(q)) or (comb[sec] >= 0.9 * comb[top] and lex[sec] >= 0.6 * lex[top] and COMPARE.search(q))
    if multi and comb[sec] > 0.25:
        chosen.append(sec)
    ans = []
    for i in chosen:
        jj, _ = best_sentence(i, qw, ss)
        if not re.search(r"\b\d", SENTS[i][jj]):
            nj = numeric_sentence(i, qw, ss)
            if nj is not None and nj != jj:
                jj = nj
        ans.append(SENTS[i][jj])
    answer = " ".join(ans)
    srcs = [int(i) for i in order[:max(top_k, len(chosen))]][:top_k]
    for c in chosen:
        if c not in srcs: srcs.append(int(c))
    srcs = sorted(srcs, key=lambda i: -comb[i])[:max(top_k, len(chosen))]
    sc = [round(float(min(1.0, max(0.0, comb[i]))), 4) for i in srcs]
    conf = float(min(0.97, max(0.62, 0.45 + 0.6 * pass_cov * 0.8 + 0.3 * float(comb[top]))))
    return dict(answer=answer, sources=srcs, scores=sc, confidence=round(conf, 4))

# ---------- API ----------
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.middleware("http")
async def cors_always(request, call_next):
    r = await call_next(request)
    r.headers["Access-Control-Allow-Origin"] = "*"
    return r

class Q(BaseModel):
    question: str
    top_k: int = 3

@app.post("/query")
async def query(body: Q):
    try:
        return await asyncio.wait_for(asyncio.to_thread(answer_query, body.question, body.top_k), 7)
    except Exception:
        return dict(answer=None, sources=[], scores=[], confidence=0.0)

@app.get("/health")
def health(): return {"ok": True, "dense": ENC is not None, "passages": N}
