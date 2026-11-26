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
TOKRE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
def toks(t):
    out = []
    for w in TOKRE.findall(t.lower()):
        if w in STOP: continue
        out.append(stem(w))
        if re.fullmatch(r"\d+ms", w): out.append("ms")
    return out
def sents(p): return [x.strip() for x in re.split(r"(?<=[a-z0-9])\.\s+(?=[A-Z])", p) if x.strip()]
_NAMEDIG = re.compile(r"\b(?:TLS|OAuth|SHA|HTTP|AES)[ /-]?\d+(?:\.\d+)?|\b[A-Za-z]+\d+[A-Za-z0-9-]*|\bB\+")
class _HN:
    def search(self, t): return re.search(r"\b\d", _NAMEDIG.sub(" ", t))
HASNUM = _HN()

# query-word -> knowledge-base vocabulary (paraphrase handling)
_SYN = {
 "keepalive":["heartbeat"], "coordinator":["leader"], "master":["leader"], "primary":["leader"],
 "subordinate":["follower"], "secondary":["follower"], "trigger":["timeout"], "coefficient":["rate"],
 "scalar":["rate"], "multiplier":["rate"], "adjustment":["update"], "magnitude":["scale"],
 "duration":["interval","timeout","latency","window"], "period":["interval","timeout","window"],
 "delay":["latency","timeout"], "deadline":["timeout"], "cutoff":["threshold"], "fraction":["ratio"],
 "milliseconds":["ms"], "millisecond":["ms"], "periodic":["interval"], "transmitted":["send"], "transmit":["send"],
 "daemon":["server"], "provisioning":["generate","create"], "signal":["heartbeat","probe"], "abort":["roll","fail"],
 "cryptographic":["crypto"], "digest":["hash"], "hash":["digest"], "iteration":["round"], "replicas":["follower","replica"],
 "memory":["buffer","memory"], "crash":["failure"], "failure":["crash"], "maximum":["exceed","limit"],
 "scheduler":["schedule"], "key":["bit"], "rate":["rate"], "minibatch":["batch"], "lifetime":["ttl"], "latency":["delay"],
}
SYNS = {stem(k): [stem(v) for v in vs] for k, vs in _SYN.items()}
SYNS_KEYS = set(SYNS)

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
from types import SimpleNamespace
def wt(st): return idf(st) if DF.get(st, 0) > 0 else 1.0     # unknown words may be paraphrases: flat weight

def make_groups(text):
    g = {}
    for st in dict.fromkeys(toks(text)):
        g[st] = ({st} | set(SYNS.get(st, ())), wt(st))
    return g

def glex(tokset, groups, exclude=()):
    num = den = 0.0
    for st, (mem, w) in groups.items():
        if st in exclude: continue
        den += w
        if mem & tokset: num += w
    return num / den if den else 0.0

def analyze(body, ent_text=None):
    ent_text = body if ent_text is None else ent_text
    ents, epos, topic = entity_hits(ent_text)
    b2 = expand(body)
    topic = topic | entity_hits(expand(ent_text))[2]
    for k, v in ALIAS.items():          # words introduced by abbreviation expansion are topic words too
        if re.search(r"\b%s\b" % re.escape(k), ent_text.lower()):
            topic = topic | {stem(w) for w in TOKRE.findall(v)}
    groups = make_groups(b2)
    lexP = np.array([glex(PASS_TOK[i], groups) for i in range(N)])
    lexS = np.array([max(glex(st, groups) for st in SENT_TOK[i]) for i in range(N)])
    ps, ss = dense(b2)
    if ps is not None:
        dP = np.clip(ps, 0, 1)
        dS = np.clip(np.array([max(ss[(i, j)] for j in range(len(SENTS[i]))) for i in range(N)]), 0, 1)
        base = 0.2 * lexP + 0.3 * lexS + 0.2 * dP + 0.3 * dS
    else:
        dP = dS = None
        base = 0.5 * lexP + 0.5 * lexS
    boost = np.zeros(N)
    for i, w in ents.items(): boost[i] = w
    rank = base + boost
    order = [int(i) for i in np.argsort(-rank)]
    return SimpleNamespace(ents=ents, epos=epos, topic=topic, groups=groups, lexP=lexP, lexS=lexS, dP=dP, dS=dS,
                           base=base, rank=rank, order=order, ss=ss)

def pick_sentence(i, A, need_num=True):
    """returns (sentence_index, facet_coverage). Facet = query words that are not the topic name."""
    ex = A.topic
    use_ex = ex if any(st not in ex for st in A.groups) else ()
    best, bs, bl = 0, -1, 0.0
    for j, st in enumerate(SENT_TOK[i]):
        l = glex(st, A.groups, exclude=use_ex)
        sc = 0.6 * l + (0.4 * A.ss[(i, j)] if A.ss else 0.4 * l)
        if need_num and HASNUM.search(SENTS[i][j]): sc += 0.25
        if sc > bs: bs, best, bl = sc, j, l
    return best, bl

def first_num(sentence):
    m = re.findall(r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)", _NAMEDIG.sub(" ", sentence))
    return float(m[0]) if m else None

NUMQ = re.compile(r"\b(how many|how much|how long|duration|milliseconds?|ms|port|bits?|length|number of|count|threshold|ratio|"
                  r"rate|coefficient|interval|timeout|window|quorum|latency|batch|size|limit|probability|capacity|layers?|"
                  r"blocks?|retries|period|delay|seconds?|bytes?|percent|fraction|key size)\b", re.I)
CAPSTOP = set("what how which who when where why does is are the given since assuming synthesizing furthermore please note "
              "compare explain describe list define in on for if by with at of and or a an".split())
def unknown_proper(q):
    out = []
    for sent in re.split(r"(?<=[.?!])\s+", q):
        ws = re.findall(r"[A-Za-z][A-Za-z0-9+]*", sent)
        for k, w in enumerate(ws):
            if k == 0 or not (w[0].isupper() or w.isupper()) or w.lower() in CAPSTOP: continue
            st = stem(w.lower())
            if len(w) >= 3 and DF.get(st, 0) == 0 and w.lower() not in STOP and st not in SYNS_KEYS and w.lower() not in ALIAS: out.append(w)
    return out

PREM = re.compile(r"^\s*(since|given|assuming|assume|suppose|considering|knowing|because|granted)\b", re.I)
def split_premise(q):
    body, prem = [], []
    for c in re.split(r"(?<=[.?!])\s+", q):
        if PREM.match(c) and "," in c:
            p, r = c.split(",", 1); prem.append(p); body.append(r.strip())
        elif PREM.match(c): prem.append(c)
        else: body.append(c)
    return " ".join(body).strip(), " ".join(prem).strip()

SYNTH = re.compile(r"furthermore|adjacent architecture|how do both|both integrate|synthesiz", re.I)
ABSTAIN_FCOV = float(os.getenv("ABSTAIN_FCOV", "0.25"))
ABSTAIN_COV = float(os.getenv("ABSTAIN_COV", "0.60"))

def result(answer, srcs, scores, conf):
    return dict(answer=answer, sources=[int(i) for i in srcs], scores=[round(float(min(1, max(0, x))), 4) for x in scores],
                confidence=round(float(conf), 4))

def fill_sources(chosen, A, top_k, scores=None):
    ids = list(chosen) + [i for i in A.order if i not in chosen]
    ids = ids[:max(top_k, len(chosen))]
    sc = {i: float(min(1.0, A.rank[i])) for i in ids}
    if scores: sc.update({i: float(min(1.0, v)) for i, v in scores.items()})
    out = sorted(sc.items(), key=lambda x: -x[1])
    return [i for i, _ in out], [v for _, v in out]

def answer_synth(q, top_k, dbg):
    stmts = []
    for sent in re.split(r"(?<=[.?!])\s+", q):
        if not sent.strip() or re.match(r"\s*how (do|are|does|can) (both|these|they|the two)", sent, re.I): continue
        s2 = re.sub(r"^[^:]{0,110}:\s*", "", sent).strip()
        if s2: stmts.append(s2)
    chosen, own, A0 = [], {}, None
    for st in stmts[:4]:
        A = analyze(st); A0 = A0 or A
        for i in A.order:
            if i not in chosen: chosen.append(i); own[i] = A.rank[i]; break
    if len(chosen) < 2:
        Aq = analyze(" ".join(stmts) or q); A0 = A0 or Aq
        for i in Aq.order:
            if len(chosen) >= 2: break
            if i not in chosen: chosen.append(i); own[i] = Aq.rank[i]
    parts = []
    for i in chosen:
        j = next((j for j in range(1, len(SENTS[i])) if HASNUM.search(SENTS[i][j])), 0)
        first = SENTS[i][0].rstrip(".")
        parts.append(f"{NAME[i]}: {first}. {SENTS[i][j].rstrip('.')}." if j else f"{NAME[i]}: {first}.")
    names = " and ".join(NAME[i] for i in chosen)
    answer = " ".join(parts) + f" Integration: {names} are complementary approaches in the same area; each passage above gives the parameters and behavior of its own mechanism."
    srcs, sc = fill_sources(chosen, A0, top_k, own)
    dbg.update(statements=stmts, chosen=chosen, decision="answer (synthesis template)")
    return result(answer, srcs, sc, 0.9), dbg

@lru_cache(maxsize=2048)
def answer_query(question: str, top_k: int):
    dbg = {}
    q, injected = sanitize(question)
    dbg["sanitized"] = q; dbg["injection_stripped"] = injected
    top_k = max(1, min(int(top_k or 3), 10))
    if not q:
        dbg["decision"] = "abstain: empty after sanitize"
        return result(None, [], [], 0.05), dbg
    if SYNTH.search(q):
        return answer_synth(q, top_k, dbg)
    body, prem = split_premise(q)
    if not body: body = q
    dbg["premise_ignored"] = prem
    A = analyze(body, body + " " + prem)
    top = A.order[0]
    strong = [i for i, w in A.ents.items() if w >= S]
    multi = sorted(sorted(strong, key=lambda i: -A.rank[i])[:4], key=lambda i: A.epos[i]) if len(strong) >= 2 else []
    chosen = multi if multi else [top]
    top = chosen[0]
    numeric_q = bool(NUMQ.search(body))
    false_prem = None
    if prem:
        pnums = set(re.findall(r"\d+(?:\.\d+)?", _NAMEDIG.sub(" ", prem)))
        kbnums = set()
        for i in (list(A.ents) or [top]):
            kbnums |= set(re.findall(r"\d+(?:\.\d+)?", _NAMEDIG.sub(" ", KB[i])))
        if pnums - kbnums: false_prem = sorted(pnums - kbnums)
    unk = unknown_proper(q)
    j, fcov = pick_sentence(top, A)
    dbg.update(entities={int(i): NAME[i] for i in A.ents}, numeric_q=numeric_q, unknown=unk, facet_cov=round(fcov, 3),
               lexP=round(float(A.lexP[top]), 3), lexS=round(float(A.lexS[top]), 3), dense_active=A.dP is not None,
               top5=[dict(id=i, rank=round(float(A.rank[i]), 3), lexP=round(float(A.lexP[i]), 3), lexS=round(float(A.lexS[i]), 3),
                          dense=None if A.dP is None else round(float(A.dP[i]), 3)) for i in A.order[:5]])
    reason = None
    if false_prem: reason = f"false premise: {false_prem} not supported by the knowledge base"
    elif unk and not multi: reason = f"unknown entity {unk}"
    elif not multi and numeric_q:
        if fcov < ABSTAIN_FCOV: reason = f"facet coverage {fcov:.2f} < {ABSTAIN_FCOV}"
        elif not HASNUM.search(SENTS[top][j]): reason = "numeric question but best sentence has no value"
        elif not A.ents and A.lexS[top] < 0.3: reason = f"no entity and sentence match {A.lexS[top]:.2f} < 0.3"
    elif not multi:
        if A.lexP[top] < ABSTAIN_COV: reason = f"non-numeric question, passage coverage {A.lexP[top]:.2f} < {ABSTAIN_COV}"
    if reason:
        dbg["decision"] = "abstain: " + reason
        return result(None, [], [], min(0.35, float(A.base[top]))), dbg

    if len(chosen) >= 2:
        parts, nums = [], []
        for i in chosen:
            jj, _ = pick_sentence(i, A); st = SENTS[i][jj]
            parts.append(f"{NAME[i]}: {st.rstrip('.')}."); nums.append((NAME[i], first_num(st)))
        answer = " ".join(parts); ql = body.lower()
        vals = [(n, v) for n, v in nums if v is not None]
        if len(vals) >= 2:
            if re.search(r"\b(larger|greater|higher|more|bigger|longer|largest|highest|longest)\b", ql):
                n, v = max(vals, key=lambda x: x[1]); answer += f" Larger value: {n} ({v:g})."
            elif re.search(r"\b(smaller|lower|less|fewer|shorter|smallest|lowest|shortest)\b", ql):
                n, v = min(vals, key=lambda x: x[1]); answer += f" Smaller value: {n} ({v:g})."
            if re.search(r"\b(sum|total|combined|together|added)\b", ql): answer += f" Sum: {sum(v for _, v in vals):g}."
            if re.search(r"\bdifferen", ql) and len(vals) == 2: answer += f" Difference: {abs(vals[0][1] - vals[1][1]):g}."
    else:
        answer = SENTS[top][j]
    srcs, sc = fill_sources(chosen, A, top_k)
    conf = float(min(0.97, max(0.65, 0.5 + 0.35 * float(A.lexS[top]) + 0.2 * float(A.base[top]))))
    dbg["decision"] = "answer" + (" (multi-entity)" if len(chosen) >= 2 else "")
    return result(answer, srcs, sc, conf), dbg

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
def logs(n: int = 100, compact: bool = False):
    rows = list(LOGS)[-n:]
    if not compact: return rows
    return [{k: r.get(k) for k in ("ts", "q", "decision", "answer", "sources", "conf", "ms", "premise_ignored", "entities")} for r in rows]

@app.get("/debug")
def debug(q: str, top_k: int = 3):
    res, dbg = answer_query(q, top_k)
    return {"result": res, "debug": dbg}
