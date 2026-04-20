from __future__ import annotations

import os
import random
import re
import subprocess
from pathlib import Path

import requests
import streamlit as st

from knowledge_base import (
    build_knowledge_base,
    build_search_index,
    concept_matches,
    format_page_suffix,
    load_catalog,
    search_catalog,
    snippet,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
GENERATED_DIR = BASE_DIR / "generated"
OLLAMA_EXE = Path(os.getenv("OLLAMA_EXE", Path.home() / "AppData/Local/Programs/Ollama/ollama.exe"))
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

MODES = [
    "/learn-topic",
    "/recall-lecture",
    "/quiz-topic",
    "/homework-helper",
    "/plan",
]

MODE_DESCRIPTIONS = {
    "/learn-topic": "Leiame teema jaoks seotud konseptsioonid, allikad ja otsetsitaadid.",
    "/recall-lecture": "Läheme materjali etapiliselt läbi ja tõstame välja active recall küsimused.",
    "/quiz-topic": "Küsime teemast kontrollküsimusi ja võrdleme sinu vastust materjaliga.",
    "/homework-helper": "Leiame taustamaterjalid, aga ei anna valmis vastust ega koodi ette.",
    "/plan": "Koostame prioriteetse õppimisplaani vastavalt ajale ja eesmärgile.",
}


st.set_page_config(page_title="TalTech Active Recall Agent", page_icon="🎓", layout="wide")
st.title("🎓 TalTech Active Recall Agent")
st.caption("Viitepõhine õppimisagent: konseptsioonid, allikad, tsitaadid ja õppimisrežiimid.")


def list_local_models() -> list[str]:
    if not OLLAMA_EXE.exists():
        return []

    result = subprocess.run(
        [str(OLLAMA_EXE), "list"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []

    models: list[str] = []
    for line in result.stdout.splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        models.append(line.split()[0])
    return models


@st.cache_resource(show_spinner=False)
def load_runtime_bundle() -> dict:
    catalog = load_catalog(GENERATED_DIR) if (GENERATED_DIR / "catalog.json").exists() else build_knowledge_base(DATA_DIR, GENERATED_DIR)
    search_index = build_search_index(catalog, BASE_DIR)
    return {"catalog": catalog, "search_index": search_index}


def rebuild_catalog() -> dict:
    catalog = build_knowledge_base(DATA_DIR, GENERATED_DIR)
    load_runtime_bundle.clear()
    st.cache_data.clear()
    return {"catalog": catalog, "search_index": build_search_index(catalog, BASE_DIR)}


def generate_with_ollama(mode: str, user_query: str, matches: list[dict], model: str) -> str:
    context = "\n\n".join(
        f"Chunk {match['chunk_id']} | Allikas: {match['source']}{format_page_suffix(match.get('page'))} | Konseptsioonid: {', '.join(match['concepts'])}\n{match['text']}"
        for match in matches
    )
    prompt = (
        f"Sa oled õppimisagent režiimis {mode}. "
        "Vasta eesti keeles ainult antud viidete põhjal. "
        "Ära lisa fakti, mida viidetes ei ole. "
        "Vastus peab olema lühike ja praktiline.\n\n"
        f"Kasutaja sisend: {user_query}\n\n"
        f"Materjal:\n{context}\n\n"
        "Kasuta punktloendit."
    )
    response = requests.post(
        "http://127.0.0.1:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def extract_relevant_quotes(query: str, matches: list[dict], max_quotes: int = 3) -> list[dict]:
    query_terms = {
        term
        for term in re.findall(r"[A-Za-zÕÄÖÜŠŽõäöüšž0-9-]{4,}", query.lower())
    }
    quotes: list[dict] = []

    for match in matches:
        sentences = re.split(r"(?<=[.!?])\s+", match["text"])
        best_sentence = ""
        best_overlap = -1
        for sentence in sentences:
            lowered = sentence.lower()
            overlap = sum(1 for term in query_terms if term in lowered)
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = sentence.strip()

        quote_text = best_sentence or snippet(match["text"], 220)
        quotes.append(
            {
                "text": quote_text,
                "source": match["source"],
                "page": match.get("page"),
                "chunk_id": match["chunk_id"],
                "score": match["score"],
            }
        )

    quotes.sort(key=lambda item: item["score"], reverse=True)
    return quotes[:max_quotes]


def build_grounded_learn_answer(topic: str, matches: list[dict], related_concepts: list[dict]) -> str:
    quotes = extract_relevant_quotes(topic, matches, max_quotes=4)
    lines = [f"**Teema:** {topic}", "", "**Kõige asjakohasemad viited materjalist:**"]
    for quote in quotes:
        lines.append(
            f"- \"{quote['text']}\"  \n  Allikas: `{quote['source']}`{format_page_suffix(quote['page'])}, chunk `{quote['chunk_id']}`"
        )

    if related_concepts:
        lines.append("")
        lines.append("**Seotud konseptsioonid:**")
        lines.extend(f"- {item['concept']}" for item in related_concepts[:5])

    return "\n".join(lines)


def build_grounded_homework_answer(matches: list[dict]) -> str:
    quotes = extract_relevant_quotes(" ".join(match["concepts"][0] for match in matches if match["concepts"]), matches, max_quotes=4)
    lines = [
        "Ma ei anna valmis lahendust. Vaata kõigepealt neid taustaviiteid ja sõnasta seejärel ise probleem.",
        "",
        "**Soovitatud materjalid:**",
    ]
    for quote in quotes:
        lines.append(
            f"- `{quote['source']}`{format_page_suffix(quote['page'])}, chunk `{quote['chunk_id']}`  \n  \"{quote['text']}\""
        )
    lines.append("")
    lines.append("Järgmine samm: loe esimene viide läbi ja kirjuta ühe lausega, mis on ülesande sisuline eesmärk.")
    return "\n".join(lines)


def render_matches(matches: list[dict]) -> None:
    st.markdown("**Leitud viited**")
    for idx, match in enumerate(matches, start=1):
        concepts = ", ".join(match["concepts"][:5]) if match["concepts"] else "puudub"
        st.markdown(
            f"{idx}. `{match['source']}`{format_page_suffix(match.get('page'))} | chunk `{match['chunk_id']}` | sobivus `{match['score']}` | konseptsioonid: {concepts}"
        )
        st.caption(snippet(match["text"], 320))


def build_quiz(matches: list[dict]) -> list[dict]:
    quiz_items = []
    for match in matches[:3]:
        concept = match["concepts"][0] if match["concepts"] else "teema"
        quiz_items.append(
            {
                "question": f"Selgita oma sõnadega, mida tähendab '{concept}' materjali põhjal.",
                "reference": f"{snippet(match['text'], 260)}\n\nAllikas: {match['source']}{format_page_suffix(match.get('page'))}, chunk {match['chunk_id']}",
            }
        )
    return quiz_items


def build_recall_steps(matches: list[dict]) -> list[str]:
    steps = []
    for idx, match in enumerate(matches[:5], start=1):
        concept_part = ", ".join(match["concepts"][:3]) or "põhiteema"
        steps.append(
            f"{idx}. Loe `{match['source']}`{format_page_suffix(match.get('page'))} chunk `{match['chunk_id']}`. "
            f"Peata lugemine ja proovi ilma vaatamata seletada: {concept_part}."
        )
    return steps


def build_plan(matches: list[dict], grade_goal: int, hours: int) -> str:
    concepts = []
    for match in matches:
        concepts.extend(match["concepts"][:3])

    unique = []
    for concept in concepts:
        if concept not in unique:
            unique.append(concept)

    priority_count = 3 if grade_goal <= 2 else 5 if grade_goal <= 4 else 8
    chosen = unique[:priority_count]

    lines = [
        f"Eesmärk: hinne {grade_goal}/5, aeg {hours}h",
        "",
        "Prioriteetsed konseptsioonid:",
    ]
    lines.extend(f"- {concept}" for concept in chosen)
    lines.append("")
    lines.append("Soovituslik tööjaotus:")
    lines.append("- 40% ajast: esmased mõisted ja definitsioonid")
    lines.append("- 40% ajast: näited, ülesanded ja võrdlused")
    lines.append("- 20% ajast: enesekontroll ja recall")
    return "\n".join(lines)


def fallback_related_concepts(matches: list[dict]) -> list[dict]:
    counts: dict[str, int] = {}
    sources: set[str] = set()
    for match in matches:
        sources.add(match["source"])
        for concept in match["concepts"]:
            counts[concept] = counts.get(concept, 0) + 1

    ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return [
        {"concept": concept, "count": count, "sources": sorted(sources)}
        for concept, count in ordered[:10]
    ]


def generated_markdown_files() -> list[Path]:
    if not GENERATED_DIR.exists():
        return []
    return sorted(GENERATED_DIR.rglob("*.md"))


runtime = load_runtime_bundle()
catalog = runtime["catalog"]
search_index = runtime["search_index"]
models = list_local_models()

with st.sidebar:
    st.subheader("Teadmistebaas")
    st.write(f"Allikaid: {len(catalog['sources'])}")
    st.write(f"Konseptsioone: {len(catalog['concepts'])}")
    st.write(f"Tükke: {len(catalog['chunks'])}")
    if st.button("Rebuild knowledge base"):
        runtime = rebuild_catalog()
        catalog = runtime["catalog"]
        search_index = runtime["search_index"]
        st.success("Teadmistebaas uuendatud.")

    st.subheader("Automaatika")
    st.code(f'python "{BASE_DIR / "index_materials.py"}"')
    st.caption("Pane see Windows Task Schedulerisse, kui tahad cronilaadset uuendust.")

    st.subheader("Ollama")
    if models:
        selected_model = st.selectbox("Mudel", models, index=0)
    else:
        selected_model = None
        st.warning("Ollama mudelit ei leitud.")
        st.code(f'"{OLLAMA_EXE}" pull {DEFAULT_MODEL}')

    st.subheader("Markdown väljundid")
    generated_files = generated_markdown_files()
    if generated_files:
        preview_file = st.selectbox(
            "Vaata faili",
            options=generated_files,
            format_func=lambda path: str(path.relative_to(GENERATED_DIR)),
        )
        st.caption(str(preview_file))
        st.text_area("Sisu eelvaade", preview_file.read_text(encoding="utf-8"), height=220)
    else:
        st.caption("Generated markdown faile veel ei ole.")


mode = st.selectbox("Mode", MODES, help="Vali töörežiim vastavalt õppimise eesmärgile.")
st.write(MODE_DESCRIPTIONS[mode])

if mode == "/plan":
    col1, col2 = st.columns(2)
    with col1:
        user_query = st.text_input("Mis aine või teema jaoks plaani tahad?", placeholder="nt SQL eksam")
    with col2:
        hours = st.slider("Tunde õppimiseks", min_value=1, max_value=40, value=6)
    grade_goal = st.slider("Hinde eesmärk", min_value=1, max_value=5, value=3)
else:
    user_query = st.text_input("Sisend", placeholder="nt normaliseerimine, relatsioonid, indeksid")
    hours = 0
    grade_goal = 0


if user_query:
    matches = search_catalog(search_index, user_query, top_k=6)
    related_concepts = concept_matches(catalog, user_query, BASE_DIR)
    if not related_concepts:
        related_concepts = fallback_related_concepts(matches)

    left, right = st.columns([1.6, 1])
    with left:
        if not matches:
            st.error("Sellele päringule ei leidunud piisavalt tugevaid vasteid.")
        else:
            if mode == "/learn-topic":
                st.markdown(build_grounded_learn_answer(user_query, matches, related_concepts))

            elif mode == "/homework-helper":
                st.markdown(build_grounded_homework_answer(matches))

            elif mode == "/recall-lecture":
                st.markdown("**Recall flow**")
                for step in build_recall_steps(matches):
                    st.markdown(step)
                if selected_model and matches[0]["score"] >= 0.5:
                    try:
                        st.markdown("**Lühike juhendatud kokkuvõte**")
                        st.markdown(generate_with_ollama(mode, user_query, matches[:3], selected_model))
                    except Exception as exc:
                        st.caption(f"Ollama kokkuvõte ebaõnnestus: {exc}")

            elif mode == "/quiz-topic":
                st.markdown("**Mini-quiz**")
                for item in build_quiz(matches):
                    st.markdown(f"- {item['question']}")
                    with st.expander("Vaata materjaliviidet"):
                        st.write(item["reference"])

            elif mode == "/plan":
                st.markdown(build_plan(matches, grade_goal, hours))

            render_matches(matches)

    with right:
        st.markdown("**Seotud konseptsioonid**")
        if related_concepts:
            for item in related_concepts[:12]:
                st.markdown(f"- `{item['concept']}` | viiteid {item['count']} | allikaid {len(item['sources'])}")
        else:
            all_concepts = list(catalog["concepts"].items())
            sample = random.sample(all_concepts, k=min(10, len(all_concepts))) if all_concepts else []
            for name, info in sample:
                st.markdown(f"- `{name}` | viiteid {info['count']}")
