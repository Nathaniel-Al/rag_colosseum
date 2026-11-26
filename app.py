"""RAG service: POST /query.
Hybrid dense + IDF-lexical retrieval, entity-aware boosting/synthesis, extractive answers,
abstention, prompt-injection stripping, and full request logging (stdout + GET /logs)."""
import asyncio, json, math, os, re, sys, threading, time
from collections import Counter, deque
from functools import lru_cache

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

KB = json.load(open(os.path.join(os.path.dirname(__file__), "knowledge_base.json")))
N = len(KB)

# ================= logging =================
LOGS = deque(maxlen=1000)
def log(rec):
    rec["ts"] = time.strftime("%H:%M:%S")
    LOGS.append(rec)
    sys.stdout.write("LOG " + json.dumps(rec, ensure_ascii=False) + "\n"); sys.stdout.flush()

# ================= text utils =================
STOP = set("""a an the of to in on for and or is are was were be been what which who whom how does do did
by with at as from that this these those it its into over under than then there their about per
default configured configure standard value used use uses using specified maximum minimum number
tell me please give state report according passage knowledge base compare difference many much has have
can could would should between also instead additionally kindly now sum total combined together larger smaller greater higher lower fewer more less differ differs differently versus vs long often size sizes time limit amount""".split())
def stem(w):
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"): w = w[:-1]
    return w[:5] if len(w) > 5 else w
def toks(t): return [stem(w) for w in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", t.lower()) if w not in STOP]
def sents(p): return [s.strip() for s in re.split(r"(?<=[a-z0-9])\.\s+(?=[A-Z])", p) if s.strip()]
_NAMEDIG = re.compile(r"\b(?:TLS|OAuth|SHA|HTTP|AES)[ /-]?\d+(?:\.\d+)?|\b[A-Za-z]+\d+[A-Za-z0-9-]*|\bB\+")
class _HN:
    def search(self, t): return re.search(r"\b\d", _NAMEDIG.sub(" ", t))
HASNUM = _HN()

PASS_TOK = [set(toks(p)) for p in KB]
DF = Counter(w for s in PASS_TOK for w in s)
def idf(w): return math.log((N + 1) / (DF.get(w, 0) + 0.5))
SENTS = [sents(p) for p in KB]
SENT_TOK = [[set(toks(s)) for s in ss] for ss in SENTS]

# ================= entity aliases (strong = names the topic; weak = shared by several passages) =================
S, W = 0.45, 0.12
ENT = {
 0:(S,[r"raft"]), 1:(S,[r"paxos",r"chubby"]), 2:(S,[r"gossip",r"phi accrual",r"epidemic"]),
 3:(S,[r"swim"]), 4:(S,[r"two ?phase commit",r"three ?phase commit",r"2pc",r"3pc",r"presumed abort"]),
 5:(S,[r"saga"]), 6:(S,[r"consistent hash"]), 7:(S,[r"rendezvous",r"highest random weight"]),
 8:(S,[r"vector clock",r"version vector",r"riak"]), 9:(S,[r"hybrid logical clock",r"hlc"]),
 10:(S,[r"sgd",r"stochastic gradient"]), 11:(S,[r"adamw"]), 12:(S,[r"transformer encoder",r"encoder"]),
 13:(S,[r"autoregressive",r"decoder",r"llama"]), 14:(S,[r"logistic regression"]),
 15:(S,[r"svm",r"support vector"]), 16:(S,[r"perceptron",r"mlp"]),
 17:(S,[r"resnet",r"residual",r"convnext",r"convolution"]), 18:(S,[r"batch ?norm"]),
 19:(S,[r"layer ?norm"]), 20:(S,[r"postgres",r"pgbouncer"]), 21:(S,[r"mysql",r"innodb"]),
 22:(S,[r"b\+ ?tree",r"b tree"]), 23:(S,[r"lsm",r"log structured merge",r"rocksdb"]),
 24:(S,[r"mongo",r"wiredtiger"]), 25:(W,[r"cassandra",r"cql"]), 26:(S,[r"redis"]), 27:(S,[r"memcache"]),
 28:(S,[r"write ahead log",r"wal",r"aries"]), 29:(S,[r"shadow paging",r"sqlite"]),
 30:(S,[r"\btcp\b",r"transmission control"]), 31:(S,[r"quic"]),
 32:(S,[r"(?<!multicast )(?<!m)\bdns\b",r"domain name system",r"dnssec"]),
 33:(S,[r"mdns",r"multicast dns",r"bonjour",r"avahi"]),
 34:(S,[r"http ?/ ?2",r"http2",r"hpack"]), 35:(S,[r"http ?/ ?3",r"http3",r"qpack"]),
 36:(S,[r"tls ?1\.3",r"transport layer security"]), 37:(S,[r"tls ?1\.2"]),
 38:(S,[r"bgp",r"border gateway"]), 39:(S,[r"ospf",r"open shortest path"]),
 40:(S,[r"public key crypto",r"\brsa\b"]), 41:(S,[r"elliptic",r"ecdsa",r"ed25519",r"\becc\b"]),
 42:(S,[r"argon2",r"credential storage"]), 43:(S,[r"pbkdf2"]),
 44:(S,[r"firewall",r"\bwaf\b"]), 45:(S,[r"api gateway",r"token bucket"]),
 46:(S,[r"\baes\b",r"advanced encryption",r"\bgcm\b"]), 47:(S,[r"chacha",r"poly1305",r"wireguard"]),
 48:(S,[r"pkce",r"oauth"]), 49:(S,[r"\bjwt\b",r"json web token"]),
}
ENT_RE = {i: (w, [re.compile(p) for p in ps]) for i, (w, ps) in ENT.items()}
NAME = {0:"Raft",1:"Multi-Paxos",2:"Gossip",3:"SWIM",4:"Two-phase commit",5:"Saga",6:"Consistent hashing",7:"Rendezvous hashing",
 8:"Vector clocks",9:"Hybrid logical clocks",10:"SGD",11:"AdamW",12:"Transformer encoder",13:"Autoregressive decoder",
 14:"Logistic regression",15:"SVM",16:"MLP",17:"Residual CNN",18:"Batch normalization",19:"Layer normalization",
 20:"PostgreSQL",21:"MySQL InnoDB",22:"B+ Tree",23:"LSM tree",24:"MongoDB",25:"Cassandra",26:"Redis",27:"Memcached",
 28:"Write-ahead logging",29:"Shadow paging",30:"TCP",31:"QUIC",32:"DNS",33:"mDNS",34:"HTTP/2",35:"HTTP/3",36:"TLS 1.3",
 37:"TLS 1.2",38:"BGP",39:"OSPF",40:"Public key cryptography",41:"Elliptic curve cryptography",42:"Credential storage",
 43:"PBKDF2",44:"WAF",45:"API gateway",46:"AES",47:"ChaCha20-Poly1305",48:"PKCE",49:"JWT"}

def norm(q): return re.sub(r"[-_]+", " ", q.lower())
def entity_hits(q):
    """returns ({passage: weight}, {first-match position per passage}, set of stemmed topic words)"""
    n = norm(q); hits, pos, topic = {}, {}, set()
    for i, (w, rs) in ENT_RE.items():
        for r in rs:
            for m in r.finditer(n):
                hits[i] = w; pos[i] = min(pos.get(i, 10**6), m.start())
                for mm in re.finditer(r"[a-z0-9]+(?:\.[0-9]+)?", n):
                    if mm.start() < m.end() and mm.end() > m.start(): topic.add(stem(mm.group()))
    return hits, pos, topic

ALIAS = {"sgd":"stochastic gradient descent","mlp":"multi-layer perceptron","svm":"support vector machine",
 "lsm":"log-structured merge tree","wal":"write-ahead logging","mdns":"multicast dns","bgp":"border gateway protocol",
 "ospf":"open shortest path first","dns":"domain name system","tcp":"transmission control protocol",
 "tls":"transport layer security","hlc":"hybrid logical clock","2pc":"two-phase commit","3pc":"three-phase commit",
 "pkce":"proof key for code exchange","jwt":"json web token","aes":"advanced encryption standard",
 "ecc":"elliptic curve cryptography","waf":"web application firewall","mvcc":"multi-version concurrency control",
 "cnn":"residual convolutional network","resnet":"residual convolutional network","bn":"batch normalization",
 "ln":"layer normalization","lr":"learning rate","ecdsa":"elliptic curve ecdsa"}
def expand(q):
    for k, v in ALIAS.items(): q = re.sub(r"\b%s\b" % re.escape(k), v, q, flags=re.I)
    return q

# ================= dense embeddings (background load) =================
ENC = None; P_EMB = S_EMB = S_IDX = None
def load_dense():
    global ENC, P_EMB, S_EMB, S_IDX
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
        P_EMB = m.encode(KB, normalize_embeddings=True)
        flat = [(i, j, s) for i, ss in enumerate(SENTS) for j, s in enumerate(ss)]
        S_EMB = m.encode([s for _, _, s in flat], normalize_embeddings=True)
        S_IDX = [(i, j) for i, j, _ in flat]
        ENC = m
        answer_query.cache_clear()
        log({"event": "dense model ready"})
    except Exception as e:
        log({"event": "dense model unavailable, lexical-only", "error": str(e)[:200]})
threading.Thread(target=load_dense, daemon=True).start()

def dense(q):
    if ENC is None: return None, None
    v = ENC.encode([q], normalize_embeddings=True)[0]
    return P_EMB @ v, {k: float(x) for k, x in zip(S_IDX, S_EMB @ v)}

# ================= injection handling =================
INJ = [re.compile(p, re.I) for p in [
 r"\b(ignore|disregard|forget|override|bypass|skip|discard)\b[^.?!\n]{0,50}\b(instruction|prompt|rule|above|previous|prior|context|guideline|polic)",
 r"\b(system|developer|admin|assistant|llm|ai|model)\b[^.?!\n]{0,20}\b(prompt|message|note|instruction|override|command|directive)",
 r"\byou (are now|must|should|will|shall|have to|need to)\b", r"\bnew (instruction|task|rule|directive)s?\b",
 r"\b(reveal|print|repeat|output|show|leak|dump)\b[^.?!\n]{0,30}\b(prompt|instructions|secret|api key|password|system|hidden|knowledge base)",
 r"\b(please|kindly|also|instead|additionally|then|now)\b[^.?!\n]{0,40}\b(say|write|reply|respond|answer|output|print|append|prepend|include|add|state|mention|return|begin|start|end|insert|repeat|confirm|type|use the word)\b",
 r"\b(append|prepend)\b", r"\b(respond|reply|answer) (only )?(with|in)\b", r"\byour (answer|response|reply|output)\b",
 r"\bfrom now on\b", r"\bdo not (answer|abstain|refuse|cite|use|say|mention)\b", r"\balways (answer|say|respond|reply|return)\b",
 r"\b(set|return|report|give|make)\b[^.?!\n]{0,25}\b(confidence|sources|scores?)\b", r"\bsources?\s*[:=\[]", r"\bconfidence\s*[:=]",
 r"\b(act|behave|roleplay|role play) as\b", r"\bpretend\b", r"\bjailbreak\b", r"\bdeveloper mode\b",
 r"\b(note|correction|update|errata|important|attention|warning|override|admin)\s*:", r"\bpassage\s*\d+\s+(says|states|is)\b",
 r"\b(is|are) (wrong|outdated|incorrect|deprecated|a typo|false)\b", r"\b(actually|in reality|in truth)\b[^.?!\n]{0,40}\b(is|are|requires|uses|has)\b",
 r"\bbase64\b", r"\bhttps?://", r"\bBEGIN\b|\bEND\b[^.?!\n]{0,10}\bPROMPT\b",
]]
def sanitize(q):
    q2 = re.sub(r"```.*?```|<!--.*?-->|<[^>]*>|\[[^\]]*\]|\{[^}]*\}", " ", q, flags=re.S)
    q2 = re.sub(r"\([^)]*\b(hint|answer|note|actually|assume|system|ignore|instruction|correct|real|true)\b[^)]*\)", " ", q2, flags=re.I)
    clauses = re.split(r"(?<=[.?!;])\s+|\n+|\s+[—–]\s+|\s--\s|,\s*(?=(?:and\s+)?(?:also|then|but|please|now|instead|kindly)\b)", q2)
    kept = [c.strip() for c in clauses if c and c.strip() and not any(r.search(c) for r in INJ)]
    return " ".join(kept).strip(), (len(kept) != len([c for c in clauses if c and c.strip()])) or q2 != q

# ================= scoring =================
def lex_scores(qt):
    qw = set(qt); tot = sum(idf(w) for w in qw) or 1.0
    return np.array([sum(idf(w) for w in qw if w in PASS_TOK[i]) / tot for i in range(N)]), qw, tot

def best_sentence(i, qw, ss):
    best, bs = 0, -1
    ns = len(SENT_TOK[i])
    def w(x): return idf(x) * (1.0 - 0.6 * sum(x in st for st in SENT_TOK[i]) / ns)
    den = sum(w(x) for x in qw) or 1
    for j, st in enumerate(SENT_TOK[i]):
        l = sum(w(x) for x in qw if x in st) / den
        s = 0.5 * l + (0.5 * ss[(i, j)] if ss else 0.5 * l)
        if HASNUM.search(SENTS[i][j]): s += 0.15
        if s > bs: bs, best = s, j
    return best, bs

def numeric_sentence(i, qw, ss):
    best, bs = None, -1
    for j, st in enumerate(SENT_TOK[i]):
        if not HASNUM.search(SENTS[i][j]): continue
        s = sum(idf(x) for x in qw if x in st) + (ss[(i, j)] if ss else 0)
        if s > bs: bs, best = s, j
    return best

def pick_sentence(i, qw, ss):
    jj, _ = best_sentence(i, qw, ss)
    if not HASNUM.search(SENTS[i][jj]):
        nj = numeric_sentence(i, qw, ss)
        if nj is not None: jj = nj
    return SENTS[i][jj]

NUMRE = re.compile(r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)(?=\s*(?:ms|bits?|nodes?|members?|acceptors?|participants?|retries|samples?|tokens?|bytes?|records?|blocks?|layers?|replica|support|log|entries|instances|\b))")
def first_num(sentence):
    # prefer a number attached to a unit/noun, skip port-style identifiers only if another exists
    m = NUMRE.findall(_NAMEDIG.sub(" ", sentence))
    return float(m[0]) if m else None

ABSTAIN_COV = float(os.getenv("ABSTAIN_COV", "0.40"))
FACET_COV = float(os.getenv("FACET_COV", "0.55"))
ABSTAIN_SCORE = float(os.getenv("ABSTAIN_SCORE", "0.30"))

@lru_cache(maxsize=2048)
def answer_query(question: str, top_k: int):
    dbg = {}
    q, injected = sanitize(question)
    dbg["sanitized"] = q; dbg["injection_stripped"] = injected
    top_k = max(1, min(int(top_k or 3), 10))
    if not q:
        dbg["decision"] = "abstain: empty after sanitize"
        return dict(answer=None, sources=[], scores=[], confidence=0.05), dbg
    ents, epos, topic = entity_hits(q)
    q = expand(q)
    topic |= entity_hits(q)[2]
    qt = toks(q)
    lex, qw, tot = lex_scores(qt)
    facet = (qw - topic) or qw
    ps, ss = dense(q)
    d = np.clip(ps, 0, 1) if ps is not None else None
    base = (0.55 * d + 0.45 * lex) if d is not None else lex
    boost = np.zeros(N)
    for i, w in ents.items(): boost[i] = w
    rank = base + boost
    order = [int(i) for i in np.argsort(-rank)]
    dbg["entities"] = {int(i): NAME[i] for i in ents}
    dbg["top5"] = [dict(id=i, rank=round(float(rank[i]), 3), base=round(float(base[i]), 3), lex=round(float(lex[i]), 3),
                        dense=None if d is None else round(float(d[i]), 3)) for i in order[:5]]
    strong = [i for i, w in ents.items() if w >= S]
    multi = sorted(sorted(strong, key=lambda i: -rank[i])[:4], key=lambda i: epos[i]) if len(strong) >= 2 else []
    chosen = multi if multi else [order[0]]
    top = chosen[0]

    # abstention: coverage of question terms by chosen passage(s)
    union = set().union(*[PASS_TOK[i] for i in chosen])
    cov = sum(idf(w) for w in qw if w in union) / tot
    missing = [w for w in qw if w not in union and DF.get(w, 0) == 0 and len(w) > 2]
    ftot = sum(idf(w) for w in facet) or 1
    fcov = sum(idf(w) for w in facet if w in union) / ftot if ents else 1.0
    dbg.update(coverage=round(cov, 3), facet_cov=round(fcov, 3), missing=missing, dense_active=d is not None)
    reason = None
    if cov < ABSTAIN_COV: reason = f"coverage {cov:.2f} < {ABSTAIN_COV}"
    elif base[top] < ABSTAIN_SCORE: reason = f"base score {base[top]:.2f} < {ABSTAIN_SCORE}"
    elif ents and fcov < FACET_COV: reason = f"facet coverage {fcov:.2f} < {FACET_COV}"
    elif len(missing) >= 1 and cov < 0.85: reason = f"missing rare terms {missing}"
    elif d is not None and d[top] < 0.30 and not ents: reason = f"dense {d[top]:.2f} < 0.30"
    if reason:
        dbg["decision"] = "abstain: " + reason
        return dict(answer=None, sources=[], scores=[], confidence=round(min(0.35, float(base[top])), 4)), dbg

    # answer
    if len(chosen) >= 2:
        parts, nums = [], []
        for i in chosen:
            st = pick_sentence(i, facet, ss); parts.append(f"{NAME[i]}: {st}.".replace("..", "."))
            nums.append((NAME[i], first_num(st)))
        answer = " ".join(parts)
        qlow = q.lower()
        vals = [(n, v) for n, v in nums if v is not None]
        if len(vals) >= 2:
            if re.search(r"\b(larger|greater|higher|more|bigger|longer|largest|highest|longest|maximum)\b", qlow):
                n, v = max(vals, key=lambda x: x[1]); answer += f" Larger value: {n} ({v:g})."
            elif re.search(r"\b(smaller|lower|less|fewer|shorter|smallest|lowest|shortest|minimum)\b", qlow):
                n, v = min(vals, key=lambda x: x[1]); answer += f" Smaller value: {n} ({v:g})."
            if re.search(r"\b(sum|total|combined|together|added)\b", qlow):
                answer += f" Sum: {sum(v for _, v in vals):g}."
            if re.search(r"\bdifferen", qlow) and len(vals) == 2:
                answer += f" Difference: {abs(vals[0][1] - vals[1][1]):g}."
    else:
        answer = pick_sentence(top, facet, ss)
        sec = order[1]
        if re.search(r"\b(compare|versus|vs\.?|both|combined|together)\b", q, re.I) and base[sec] > 0.25:
            chosen.append(sec); answer += " " + pick_sentence(sec, facet, ss)
    srcs = list(chosen) + [i for i in order if i not in chosen]
    srcs = sorted(srcs[:max(top_k, len(chosen))], key=lambda i: -rank[i])
    sc = [round(float(min(1.0, max(0.0, rank[i]))), 4) for i in srcs]
    conf = float(min(0.97, max(0.62, 0.45 + 0.48 * min(1.0, cov) + 0.2 * float(base[top]))))
    dbg["decision"] = "answer" + (" (multi-entity synthesis)" if len(chosen) >= 2 else "")
    return dict(answer=answer, sources=[int(i) for i in srcs], scores=sc, confidence=round(conf, 4)), dbg

# ================= API =================
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
    t0 = time.time()
    try:
        res, dbg = await asyncio.wait_for(asyncio.to_thread(answer_query, body.question, body.top_k), 7)
    except Exception as e:
        log({"event": "ERROR", "question": body.question[:300], "error": repr(e)[:200]})
        return dict(answer=None, sources=[], scores=[], confidence=0.0)
    log({"q": body.question[:400], "top_k": body.top_k, **dbg, "answer": (res["answer"] or None) and res["answer"][:300],
         "sources": res["sources"], "scores": res["scores"], "conf": res["confidence"], "ms": int((time.time() - t0) * 1000)})
    return res

@app.get("/health")
def health(): return {"ok": True, "dense": ENC is not None, "passages": N, "logged": len(LOGS)}

@app.get("/logs")
def logs(n: int = 100): return list(LOGS)[-n:]

@app.get("/debug")
def debug(q: str, top_k: int = 3):
    res, dbg = answer_query(q, top_k)
    return {"result": res, "debug": dbg}
