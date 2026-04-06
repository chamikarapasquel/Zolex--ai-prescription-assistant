"""
🏥 Zolex AI Pipeline — Streamlit App
Loads the exact models from the notebook end-to-end.

Place this file in the SAME folder as:
  - medical_spellcheck_final/          (T5-small spell corrector)
  - intent_classifier_medical_safe.pkl
  - intent_classifier_metadata.json
  - zolex_disease_model_v2/            (DistilBERT disease predictor)
  - db_drug_interactions.csv

Run:
    pip install streamlit requests pandas transformers torch joblib
    streamlit run medical_ai_app.py
"""

import os, sys, re, json, string, unicodedata, time, warnings
import streamlit as st
import requests
import numpy as np
import pandas as pd
import joblib
import difflib
from itertools import combinations
from typing import Optional, List, Dict

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Zolex AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — dark clinical theme with emerald accents ──────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Fraunces:ital,wght@0,300;0,700;1,300&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Mono', monospace;
    background-color: #0a0f0d;
    color: #c8f5d5;
}
.main { background-color: #0a0f0d; }

.app-header {
    text-align: center;
    padding: 2rem 0 1rem;
    border-bottom: 1px solid #1e3a2a;
    margin-bottom: 2rem;
}
.app-header h1 {
    font-family: 'Fraunces', serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: #5fde95;
    letter-spacing: -1px;
    margin: 0;
}
.app-header p {
    color: #5a9070;
    font-size: 0.8rem;
    margin-top: 0.4rem;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.stage-card {
    background: #0e1a13;
    border: 1px solid #1e3a2a;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.3s;
}
.stage-card.active  { border-color: #5fde95; }
.stage-card.done    { border-color: #2e6644; }
.stage-card.skipped { border-color: #4a2a2a; background: #110e0e; }

.stage-label {
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #3a7a55;
    margin-bottom: 0.3rem;
}
.stage-title {
    font-family: 'Fraunces', serif;
    font-size: 1.15rem;
    color: #9ef5c0;
    font-weight: 300;
}
.stage-value {
    margin-top: 0.5rem;
    font-size: 0.82rem;
    color: #78c899;
    line-height: 1.6;
}

.badge-HIGH   { background:#5c1c1c; color:#ff8080; border:1px solid #a83030; padding:2px 8px; border-radius:4px; font-size:0.7rem; }
.badge-MEDIUM { background:#3d3010; color:#ffd080; border:1px solid #8a6a20; padding:2px 8px; border-radius:4px; font-size:0.7rem; }
.badge-LOW    { background:#0f2e1a; color:#80e5a0; border:1px solid #256a40; padding:2px 8px; border-radius:4px; font-size:0.7rem; }

.conf-bar-bg {
    background: #1a2e20;
    border-radius: 4px;
    height: 6px;
    margin-top: 4px;
    width: 100%;
}
.conf-bar-fill {
    height: 6px;
    border-radius: 4px;
    background: linear-gradient(90deg, #2e6644, #5fde95);
    transition: width 0.8s ease;
}

.stTextArea > div > div > textarea {
    background: #0e1a13 !important;
    border: 1px solid #1e3a2a !important;
    color: #c8f5d5 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    border-radius: 6px !important;
}
.stTextArea > div > div > textarea:focus {
    border-color: #5fde95 !important;
    box-shadow: 0 0 0 2px rgba(95,222,149,0.15) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #1a5c35, #2e8c55) !important;
    color: #c8f5d5 !important;
    border: 1px solid #3ea070 !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: 1px !important;
    font-size: 0.8rem !important;
    padding: 0.5rem 1.5rem !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #256644, #3aa065) !important;
    border-color: #5fde95 !important;
    box-shadow: 0 0 16px rgba(95,222,149,0.2) !important;
}

[data-testid="stSidebar"] {
    background: #080e0a;
    border-right: 1px solid #1a3025;
}
[data-testid="stSidebar"] * { color: #78c899 !important; }

.stSelectbox > div, .stSlider > div { color: #78c899; }

.disclaimer {
    background: #120e0a;
    border: 1px solid #4a3010;
    border-left: 4px solid #c07030;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    font-size: 0.72rem;
    color: #a07850;
    margin-top: 1.5rem;
}

.deflection-box {
    background: #0e100e;
    border: 1px solid #1a3a28;
    border-left: 4px solid #3ea070;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    color: #5fde95;
    font-size: 0.85rem;
    margin-top: 1rem;
}

::-webkit-scrollbar { width: 4px; background: #0a0f0d; }
::-webkit-scrollbar-thumb { background: #1e3a2a; border-radius: 2px; }

.metric-row { display: flex; gap: 1rem; margin-bottom: 1.2rem; }
.metric-card {
    flex: 1;
    background: #0e1a13;
    border: 1px solid #1e3a2a;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    text-align: center;
}
.metric-num { font-size: 1.8rem; color: #5fde95; font-family: 'Fraunces', serif; font-weight: 700; }
.metric-lbl { font-size: 0.62rem; color: #3a7a55; letter-spacing: 2px; text-transform: uppercase; }

.drug-chip {
    display: inline-block;
    background: #112618;
    border: 1px solid #1e4a2e;
    border-radius: 20px;
    padding: 2px 10px;
    margin: 2px;
    font-size: 0.7rem;
    color: #78c899;
}

.history-item {
    border-left: 2px solid #1e3a2a;
    padding-left: 0.7rem;
    margin-bottom: 0.6rem;
    font-size: 0.75rem;
    color: #5a9070;
}

hr { border-color: #1e3a2a; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# REAL MODEL LOADING  (exact notebook logic, cached so loads only once)
# ══════════════════════════════════════════════════════════════════════════════

SEED = 42
DEVICE = "cpu"
try:
    import torch
    torch.manual_seed(SEED)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    pass

# ─── Stage 1 : Spell Corrector (T5-small) ────────────────────────────────────
SPELL_PREFIX  = "fix: "
SPELL_MAX_LEN = 64

# Resolve spell model dir — works regardless of where streamlit run is invoked
def _find_spell_dir() -> Optional[str]:
    _name = "medical_spellcheck_final"
    candidates = []
    # Absolute path anchored to THIS file's directory (most reliable)
    try:
        _here = os.path.dirname(os.path.abspath(__file__))
        candidates.append(os.path.join(_here, _name))
    except Exception:
        pass
    # CWD — where the user typed `streamlit run`
    candidates.append(os.path.join(os.getcwd(), _name))
    # Plain relative (same as CWD usually, but keep as fallback)
    candidates.append(_name)

    for p in candidates:
        try:
            p = os.path.abspath(p)
            files = os.listdir(p)
            if any(f.endswith((".json", ".bin", ".safetensors", ".model", ".pt")) for f in files):
                return p
        except Exception:
            continue
    return None

SPELL_MODEL_DIR = _find_spell_dir()

@st.cache_resource(show_spinner="Loading Spell Corrector (T5-small)…")
def load_spell_model():
    if SPELL_MODEL_DIR is None:
        return None, None, False, "Folder not found. Place medical_spellcheck_final/ next to medical_ai_app.py"
    try:
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        # Load model weights from the saved folder
        model = T5ForConditionalGeneration.from_pretrained(SPELL_MODEL_DIR).to(DEVICE)
        model.eval()
        # Try saved tokenizer first; if corrupted fall back to base t5-small
        try:
            tok = AutoTokenizer.from_pretrained(SPELL_MODEL_DIR)
        except Exception:
            tok = AutoTokenizer.from_pretrained("t5-small")
        return tok, model, True, None
    except Exception as e:
        return None, None, False, str(e)


_spell_tok, _spell_model, SPELL_AVAILABLE, _spell_load_error = load_spell_model()

def spell_correct(text: str) -> str:
    if not SPELL_AVAILABLE or not isinstance(text, str) or not text.strip():
        return text
    try:
        import torch
        inputs = _spell_tok(
            SPELL_PREFIX + text,
            return_tensors="pt",
            max_length=SPELL_MAX_LEN,
            truncation=True,
        ).to(DEVICE)
        with torch.no_grad():
            out = _spell_model.generate(
                **inputs,
                max_length=SPELL_MAX_LEN,
                num_beams=4,
                no_repeat_ngram_size=2,
                early_stopping=True,
            )
        return _spell_tok.decode(out[0], skip_special_tokens=True)
    except Exception:
        return text


# ─── Stage 2 : Intent Classifier (TF-IDF + LR) ──────────────────────────────
INTENT_MODEL_PATH    = "intent_classifier_medical_safe.pkl"
INTENT_METADATA_PATH = "intent_classifier_metadata.json"

_CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "n't": " not",
    "'re": " are", "'s": " is", "'d": " would",
    "'ll": " will", "'t": " not", "'ve": " have", "'m": " am",
}

def _clean_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    for k, v in _CONTRACTIONS.items():
        text = text.replace(k, v)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()

@st.cache_resource(show_spinner="Loading Intent Classifier (TF-IDF + LR)…")
def load_intent_model():
    try:
        pipeline  = joblib.load(INTENT_MODEL_PATH)
        with open(INTENT_METADATA_PATH) as f:
            meta = json.load(f)
        threshold = meta.get("optimal_threshold", 0.5)
        return pipeline, threshold, True
    except Exception:
        return None, 0.5, False

_intent_pipeline, INTENT_THRESHOLD, INTENT_AVAILABLE = load_intent_model()

def classify_intent(text: str) -> dict:
    if not INTENT_AVAILABLE:
        return {"label": "Medical", "medical_prob": 1.0, "threshold_used": INTENT_THRESHOLD}
    cleaned      = _clean_text(text)
    proba        = _intent_pipeline.predict_proba([cleaned])[0]
    medical_prob = proba[list(_intent_pipeline.classes_).index(1)]
    label        = "Medical" if medical_prob >= INTENT_THRESHOLD else "Non-Medical"
    return {"label": label, "medical_prob": round(medical_prob, 4), "threshold_used": INTENT_THRESHOLD}


# ─── Stage 3 : Disease Predictor (DistilBERT) ────────────────────────────────
DISEASE_MODEL_DIR = "./zolex_disease_model_v2"
DISEASE_MAX_LEN   = 128
DISEASE_TOP_K     = 3

@st.cache_resource(show_spinner="Loading Disease Predictor (DistilBERT)…")
def load_disease_model():
    try:
        from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
        tok   = DistilBertTokenizerFast.from_pretrained(DISEASE_MODEL_DIR)
        model = DistilBertForSequenceClassification.from_pretrained(DISEASE_MODEL_DIR)
        model.to(DEVICE)
        model.eval()
        return tok, model, True
    except Exception:
        return None, None, False

_disease_tok, _disease_model, DISEASE_AVAILABLE = load_disease_model()

def predict_disease(text: str, top_k: int = DISEASE_TOP_K) -> list:
    if not DISEASE_AVAILABLE:
        return [{"disease": "Model not loaded — place zolex_disease_model_v2/ here", "confidence": 0.0}]
    try:
        import torch
        inputs = _disease_tok(
            text, return_tensors="pt", truncation=True,
            padding=True, max_length=DISEASE_MAX_LEN,
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = _disease_model(**inputs)
            probs   = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        top_idx = np.argsort(probs)[-top_k:][::-1]
        return [
            {"disease":    _disease_model.config.id2label[int(i)],
             "confidence": round(float(probs[i]), 4)}
            for i in top_idx
        ]
    except Exception as e:
        return [{"disease": f"Prediction error: {e}", "confidence": 0.0}]


# ─── Stage 4 : Drug Lookup (MED-RT / RxNav API) ──────────────────────────────
MEDRT_BASE_URL   = "https://rxnav.nlm.nih.gov/REST"
MEDRT_TIMEOUT    = 10
MEDRT_MAX_RETRY  = 3
MEDRT_RETRY_WAIT = 1.5
DRUG_LIMIT       = 10

def _medrt_get(url: str, params: dict) -> Optional[dict]:
    for attempt in range(1, MEDRT_MAX_RETRY + 1):
        try:
            r = requests.get(url, params=params, timeout=MEDRT_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            pass
        except (requests.exceptions.HTTPError, ValueError):
            return None
        if attempt < MEDRT_MAX_RETRY:
            time.sleep(MEDRT_RETRY_WAIT * attempt)
    return None

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_drugs_api(disease: str) -> list:
    data = _medrt_get(
        f"{MEDRT_BASE_URL}/rxclass/class/byName.json",
        {"className": disease, "relaSource": "MEDRT"},
    )
    if not data:
        return []
    concepts = data.get("rxclassMinConceptList", {}).get("rxclassMinConcept", [])
    if not concepts:
        return []
    class_id = concepts[0]["classId"]
    data2 = _medrt_get(
        f"{MEDRT_BASE_URL}/rxclass/classMembers.json",
        {"classId": class_id, "relaSource": "MEDRT", "rela": "may_treat", "trans": "0"},
    )
    if not data2:
        return []
    members = data2.get("drugMemberGroup", {}).get("drugMember", [])
    return sorted({m["minConcept"]["name"] for m in members})

def lookup_drugs(disease: str, limit: int = DRUG_LIMIT) -> dict:
    drugs  = _fetch_drugs_api(disease)
    source = "MED-RT API" if drugs else "No results from MED-RT"
    return {"drugs": drugs[:limit], "source": source}


# ─── Stage 5 : Interaction Checker (CSV rule-based) ──────────────────────────
INTERACTION_CSV = "db_drug_interactions.csv"
FUZZY_THRESHOLD = 0.82
_DRUG_SYNONYMS  = {
    "acetaminophen": "paracetamol", "tylenol": "paracetamol", "panadol": "paracetamol",
    "advil": "ibuprofen", "motrin": "ibuprofen", "nurofen": "ibuprofen",
    "aspirin": "acetylsalicylic acid", "asa": "acetylsalicylic acid",
    "lasix": "furosemide", "zoloft": "sertraline", "prozac": "fluoxetine",
    "plavix": "clopidogrel", "coumadin": "warfarin", "glucophage": "metformin",
    "synthroid": "levothyroxine", "lipitor": "atorvastatin", "zocor": "simvastatin",
    "nexium": "esomeprazole", "prilosec": "omeprazole", "xanax": "alprazolam",
    "valium": "diazepam", "amoxil": "amoxicillin", "zithromax": "azithromycin",
    "norvasc": "amlodipine", "ventolin": "albuterol", "aleve": "naproxen",
}
_HIGH_KW   = frozenset(["severe", "avoid", "major", "contraindicated", "life-threatening",
                         "fatal", "death", "serious", "do not", "potentially fatal"])
_MEDIUM_KW = frozenset(["moderate", "caution", "monitor", "increased risk",
                         "may increase", "reduced", "decreased", "adjust"])

def _severity(sev_raw: str, desc: str) -> str:
    s = sev_raw.strip().lower()
    if any(x in s for x in ["high", "major", "severe", "contraindicated"]):
        return "HIGH"
    if any(x in s for x in ["moderate", "medium"]):
        return "MEDIUM"
    if any(x in s for x in ["low", "minor"]):
        return "LOW"
    dl = desc.lower()
    if any(kw in dl for kw in _HIGH_KW):   return "HIGH"
    if any(kw in dl for kw in _MEDIUM_KW): return "MEDIUM"
    return "LOW"

@st.cache_resource(show_spinner="Loading Drug Interaction Database (CSV)…")
def load_interaction_checker():
    try:
        df = pd.read_csv(INTERACTION_CSV, low_memory=False)
        df.columns = df.columns.str.strip()
        col_map  = {c.lower(): c for c in df.columns}
        required = {"drug 1", "drug 2", "interaction description"}
        if required - set(col_map.keys()):
            return {}, [], False

        d1c  = col_map["drug 1"]
        d2c  = col_map["drug 2"]
        desc = col_map["interaction description"]
        sev  = col_map.get("severity")

        df = df.dropna(subset=[d1c, d2c]).copy()
        df[d1c]  = df[d1c].astype(str).str.strip().str.lower()
        df[d2c]  = df[d2c].astype(str).str.strip().str.lower()
        df[desc] = df[desc].astype(str).str.strip()

        db: Dict[str, tuple] = {}
        cols = [d1c, d2c, desc] + ([sev] if sev else [])
        for row in df[cols].itertuples(index=False):
            r1, r2, rdesc = row[0], row[1], row[2]
            sev_raw = row[3] if sev else ""
            key = "_".join(sorted([r1, r2]))
            if key not in db:
                db[key] = (_severity(sev_raw, rdesc), rdesc)

        all_drugs: set = set()
        for k in db:
            all_drugs.update(k.split("_", 1))
        return db, sorted(all_drugs), True
    except FileNotFoundError:
        return {}, [], False
    except Exception:
        return {}, [], False

_interaction_db, _all_drug_names, INTERACTION_AVAILABLE = load_interaction_checker()

def _normalise(name: str) -> str:
    syn = {k.lower(): v.lower() for k, v in _DRUG_SYNONYMS.items()}
    c = name.strip().lower()
    if c in syn:
        return syn[c]
    if c in set(_all_drug_names):
        return c
    matches = difflib.get_close_matches(c, _all_drug_names, n=1, cutoff=FUZZY_THRESHOLD)
    return matches[0] if matches else c

def check_interactions(drug_list: list) -> list:
    if not INTERACTION_AVAILABLE or len(drug_list) < 2:
        return []
    norm = [_normalise(d) for d in drug_list]
    seen, warnings = set(), []
    for d1, d2 in combinations(norm, 2):
        key = "_".join(sorted([d1, d2]))
        if key in seen:
            continue
        seen.add(key)
        if key in _interaction_db:
            sev, desc = _interaction_db[key]
            warnings.append({
                "Pair":    f"{d1.title()} ⟷ {d2.title()}",
                "Drug1":   d1, "Drug2": d2,
                "Risk":    sev, "Details": desc,
            })
    warnings.sort(key=lambda x: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(x["Risk"], 9))
    return warnings


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP UI  (unchanged from original design)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="app-header">
  <h1>🏥 Zolex AI</h1>
  <p>5-Stage Intelligent Clinical Analysis System</p>
</div>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []
if "result" not in st.session_state:
    st.session_state.result = None

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Settings")
    top_k      = st.slider("Top-K diseases", 1, 5, 3)
    drug_limit = st.slider("Drugs per disease", 3, 10, 6)

    st.markdown("---")
    st.markdown("### 🔬 Model Status")
    statuses = [
        ("1", "Spell Corrector",    "T5-small",       SPELL_AVAILABLE),
        ("2", "Intent Classifier",  "TF-IDF + LR",    INTENT_AVAILABLE),
        ("3", "Disease Predictor",  "DistilBERT",      DISEASE_AVAILABLE),
        ("4", "Drug Lookup",        "MED-RT API",      True),
        ("5", "Interaction Checker","CSV rule-based",  INTERACTION_AVAILABLE),
    ]
    for num, name, model_type, ok in statuses:
        icon       = "✅" if ok else "⚠️"
        note_color = "#5fde95" if ok else "#ff8080"
        note       = "READY" if ok else "NOT LOADED"
        st.markdown(
            f'<div style="font-size:0.7rem;color:#3a7a55;margin:6px 0 8px 0;line-height:1.6">'
            f'<span style="color:#5fde95">[{num}]</span> <span style="color:#9ef5c0">{name}</span><br>'
            f'<span style="color:#2a5040">{model_type}</span>&nbsp;'
            f'<span style="color:{note_color}">{icon} {note}</span></div>',
            unsafe_allow_html=True,
        )

    if not SPELL_AVAILABLE:
        _err_msg = _spell_load_error or "Unknown error"
        _short   = _err_msg[:120] + "…" if len(_err_msg) > 120 else _err_msg
        st.markdown(
            f'<div style="background:#0e1a13;border:1px solid #3a2020;border-left:3px solid #ff6060;'
            f'border-radius:6px;padding:0.5rem 0.7rem;font-size:0.62rem;color:#a07070;margin-top:4px;line-height:1.5">'
            f'📁 <b style="color:#c08080">Fix:</b> Place <code style="color:#78c899;background:#112618;'
            f'padding:1px 4px;border-radius:3px">medical_spellcheck_final/</code> '
            f'in the same folder as <code style="color:#78c899;background:#112618;padding:1px 4px;border-radius:3px">'
            f'medical_ai_app.py</code> and restart.<br>'
            f'<span style="color:#5a4040">ℹ️ {_short}</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown("---")
    st.markdown("### 📋 Query History")
    if st.session_state.history:
        for h in reversed(st.session_state.history[-8:]):
            label       = h["input"][:36] + "…" if len(h["input"]) > 36 else h["input"]
            intent_icon = "🟢" if h["intent"] == "Medical" else "🔴"
            st.markdown(
                f'<div class="history-item">{intent_icon} {label}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown('<span style="color:#2a5a3a;font-size:0.72rem;">No queries yet.</span>',
                    unsafe_allow_html=True)

# ── Example queries ────────────────────────────────────────────────────────
EXAMPLES = [
    "pt has hgh fevr and servere cough with difficulty breathing",
    "frequent urination, excessive thirst and blurred vision",
    "chest pain, shortness of breath and high blood pressure",
    "What is the weather like today?",
]

col_input, col_btn = st.columns([4, 1])
with col_input:
    user_input = st.text_area(
        "Describe symptoms or medical concern:",
        value="",
        height=90,
        placeholder="e.g. pt has hgh fevr and servere cough…",
        label_visibility="collapsed",
    )
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("▶ ANALYSE", use_container_width=True)

ex_cols = st.columns(4)
for i, ex in enumerate(EXAMPLES):
    with ex_cols[i]:
        if st.button(f"📌 Example {i+1}", key=f"ex_{i}", use_container_width=True):
            user_input = ex
            run_btn    = True

# ── Run Pipeline ───────────────────────────────────────────────────────────
if run_btn and user_input and user_input.strip():
    result      = {}
    progress_ph = st.empty()

    with progress_ph.container():
        prog = st.progress(0, text="Initialising pipeline…")

        # Stage 1
        corrected          = spell_correct(user_input.strip())
        result["original"] = user_input.strip()
        result["corrected"]= corrected
        prog.progress(20, text="[1/5] Spell correction complete")

        # Stage 2
        intent         = classify_intent(corrected)
        result["intent"] = intent
        prog.progress(40, text="[2/5] Intent classified")

        if intent["label"] == "Non-Medical":
            result["routed_to"]    = "Non-Medical"
            result["diseases"]     = []
            result["drugs"]        = {}
            result["interactions"] = []
            prog.progress(100, text="Non-medical query — deflected")
            time.sleep(0.4)
        else:
            result["routed_to"] = "Medical"

            # Stage 3
            diseases           = predict_disease(corrected, top_k=top_k)
            result["diseases"] = diseases
            prog.progress(60, text="[3/5] Diseases predicted")

            # Stage 4
            drug_results = {}
            for d in diseases:
                drug_results[d["disease"]] = lookup_drugs(d["disease"], limit=drug_limit)
            result["drugs"] = drug_results
            prog.progress(80, text="[4/5] Drugs retrieved from MED-RT")

            # Stage 5
            all_drugs_flat = []
            for v in drug_results.values():
                all_drugs_flat.extend(v["drugs"])
            seen_set, unique_drugs = set(), []
            for dr in all_drugs_flat:
                if dr.lower() not in seen_set:
                    seen_set.add(dr.lower())
                    unique_drugs.append(dr)
            result["all_drugs"]    = unique_drugs
            result["interactions"] = check_interactions(unique_drugs)
            prog.progress(100, text="[5/5] Interaction check complete ✓")
            time.sleep(0.4)

    progress_ph.empty()
    st.session_state.result = result
    st.session_state.history.append({"input": user_input.strip(), "intent": intent["label"]})

# ── Display Results ────────────────────────────────────────────────────────
if st.session_state.result:
    res = st.session_state.result
    st.markdown("---")

    if res["routed_to"] == "Medical":
        n_diseases        = len(res["diseases"])
        n_drugs           = len(res.get("all_drugs", []))
        n_interactions    = len(res["interactions"])
        high_interactions = sum(1 for i in res["interactions"] if i["Risk"] == "HIGH")

        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-card">
            <div class="metric-num">{n_diseases}</div>
            <div class="metric-lbl">Conditions Found</div>
          </div>
          <div class="metric-card">
            <div class="metric-num">{n_drugs}</div>
            <div class="metric-lbl">Medications Identified</div>
          </div>
          <div class="metric-card">
            <div class="metric-num">{n_interactions}</div>
            <div class="metric-lbl">Interactions Detected</div>
          </div>
          <div class="metric-card">
            <div class="metric-num" style="color:{'#ff6060' if high_interactions > 0 else '#5fde95'}">{high_interactions}</div>
            <div class="metric-lbl">High-Risk Interactions</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")

    with left:
        changed = res["corrected"] != res["original"]
        st.markdown(f"""
        <div class="stage-card done">
          <div class="stage-label">Stage 01 — T5-Small</div>
          <div class="stage-title">🔤 Spell Corrector</div>
          <div class="stage-value">
            <b>Original:</b> {res['original']}<br>
            <b>Corrected:</b> <span style="color:#5fde95">{res['corrected']}</span>
            {"&nbsp;<span style='color:#3ea070;font-size:0.68rem'>✓ Changes applied</span>" if changed else "&nbsp;<span style='color:#2a5040;font-size:0.68rem'>No changes needed</span>"}
          </div>
        </div>
        """, unsafe_allow_html=True)

        intent       = res["intent"]
        intent_color = "#5fde95" if intent["label"] == "Medical" else "#ff8080"
        bar_grad     = "linear-gradient(90deg,#2e6644,#5fde95)" if intent["label"] == "Medical" else "linear-gradient(90deg,#5c1c1c,#ff8080)"
        st.markdown(f"""
        <div class="stage-card done">
          <div class="stage-label">Stage 02 — TF-IDF + Logistic Regression</div>
          <div class="stage-title">🎯 Intent Classifier</div>
          <div class="stage-value">
            <b>Label:</b> <span style="color:{intent_color}">{intent['label']}</span><br>
            <b>Medical probability:</b> {intent['medical_prob']*100:.1f}%
          </div>
          <div class="conf-bar-bg">
            <div class="conf-bar-fill" style="width:{intent['medical_prob']*100:.0f}%;background:{bar_grad}"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if res["routed_to"] == "Medical":
            disease_html = ""
            for d in res["diseases"]:
                pct = d["confidence"] * 100
                disease_html += f"""
                <div style="margin-bottom:0.6rem">
                  <div style="display:flex;justify-content:space-between">
                    <span style="color:#9ef5c0">{d['disease']}</span>
                    <span style="color:#5fde95">{pct:.1f}%</span>
                  </div>
                  <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{pct:.0f}%"></div></div>
                </div>"""
            st.markdown(f"""
            <div class="stage-card done">
              <div class="stage-label">Stage 03 — DistilBERT</div>
              <div class="stage-title">🔬 Disease Predictor</div>
              <div class="stage-value">{disease_html}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="stage-card skipped">
              <div class="stage-label">Stage 03 — DistilBERT</div>
              <div class="stage-title" style="color:#5a3a3a">🔬 Disease Predictor</div>
              <div class="stage-value" style="color:#5a3a3a">Skipped — Non-medical query</div>
            </div>
            """, unsafe_allow_html=True)

    with right:
        if res["routed_to"] == "Medical":
            drug_html = ""
            for disease, dinfo in res["drugs"].items():
                chips     = "".join(f'<span class="drug-chip">{d}</span>' for d in dinfo["drugs"])
                src_badge = f'<span style="color:#2a5040;font-size:0.62rem">[{dinfo["source"]}]</span>'
                no_drugs  = '<span style="color:#2a5040">No drugs found in MED-RT</span>'
                drug_html += f"""
                <div style="margin-bottom:0.8rem">
                  <div style="color:#5fde95;font-size:0.72rem;margin-bottom:3px">{disease} {src_badge}</div>
                  <div>{chips if chips else no_drugs}</div>
                </div>"""
            st.markdown(f"""
            <div class="stage-card done">
              <div class="stage-label">Stage 04 — MED-RT / RxNav API</div>
              <div class="stage-title">💊 Drug Lookup</div>
              <div class="stage-value">{drug_html}</div>
            </div>
            """, unsafe_allow_html=True)

            interactions = res["interactions"]
            icons = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
            if interactions:
                ix_html = ""
                for ix in interactions:
                    ix_html += f"""
                    <div style="margin-bottom:0.7rem;border-bottom:1px solid #1a2e1a;padding-bottom:0.5rem">
                      <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
                        <span class="badge-{ix['Risk']}">{ix['Risk']}</span>
                        <span style="color:#9ef5c0;font-size:0.8rem">{ix['Pair']}</span>
                      </div>
                      <div style="color:#5a9070;font-size:0.72rem">{icons[ix['Risk']]} {ix['Details']}</div>
                    </div>"""
                st.markdown(f"""
                <div class="stage-card done">
                  <div class="stage-label">Stage 05 — Rule-Based CSV Engine</div>
                  <div class="stage-title">⚠️ Drug Interactions</div>
                  <div class="stage-value">{ix_html}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="stage-card done">
                  <div class="stage-label">Stage 05 — Rule-Based CSV Engine</div>
                  <div class="stage-title">✅ Drug Interactions</div>
                  <div class="stage-value" style="color:#2e8c55">No significant interactions detected among retrieved medications.</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="stage-card skipped">
              <div class="stage-label">Stage 04 — MED-RT / RxNav API</div>
              <div class="stage-title" style="color:#5a3a3a">💊 Drug Lookup</div>
              <div class="stage-value" style="color:#5a3a3a">Skipped — Non-medical query</div>
            </div>
            <div class="stage-card skipped">
              <div class="stage-label">Stage 05 — Rule-Based CSV Engine</div>
              <div class="stage-title" style="color:#5a3a3a">⚠️ Interaction Checker</div>
              <div class="stage-value" style="color:#5a3a3a">Skipped — Non-medical query</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="deflection-box">
              🏥 I'm a medical assistant and can only help with health-related questions.
              Please describe your symptoms or medical concern.
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
      ⚕️ <b>Clinical Disclaimer:</b> This is an AI-assisted analysis for educational and research purposes only.
      Disease predictions and drug recommendations are generated by machine learning models and should not
      replace professional medical advice. Always consult a licensed physician before making any
      healthcare decisions.
    </div>
    """, unsafe_allow_html=True)

elif not user_input and not st.session_state.result:
    st.markdown("""
    <div style="text-align:center;padding:3rem 0;color:#2a5040">
      <div style="font-size:2.5rem;margin-bottom:1rem">🏥</div>
      <div style="font-family:'Fraunces',serif;font-size:1.3rem;color:#3a7a55">
        Enter your symptoms above to begin analysis
      </div>
      <div style="font-size:0.72rem;margin-top:0.5rem;letter-spacing:2px;text-transform:uppercase">
        or choose an example query
      </div>
    </div>
    """, unsafe_allow_html=True)
