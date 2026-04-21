from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
_EMBEDDING_BACKEND_CACHE: dict[str, EmbeddingBackend | None] = {}
DEFAULT_SYNONYMS = {
    "dubleerimine": ["redundantsus", "normaliseerimine", "anomaliad", "korduvus", "liiasus"],
    "dubleerimist": ["redundantsus", "normaliseerimine", "anomaliad", "korduvus", "liiasus"],
    "kordamine": ["redundantsus", "normaliseerimine", "liiasus"],
    "voorvoti": ["valisvoti"],
    "valisvoti": ["voorvoti"],
}
STOPWORDS = {
    "aga",
    "all",
    "alla",
    "alates",
    "ei",
    "ehk",
    "et",
    "iga",
    "ikka",
    "ja",
    "kas",
    "kuidas",
    "kui",
    "kus",
    "mis",
    "moni",
    "mida",
    "ning",
    "nii",
    "nuud",
    "oli",
    "oma",
    "on",
    "ole",
    "oled",
    "olla",
    "peab",
    "seda",
    "see",
    "seega",
    "siin",
    "siis",
    "ta",
    "teha",
    "tema",
    "kohta",
    "toimub",
    "voi",
    "voib",
    "vaga",
    "uks",
}
NOISE_TERMS = {
    "endobj",
    "endstream",
    "stream",
    "filter",
    "decode",
    "flatedecode",
    "colorspace",
    "devicergb",
    "patterntype",
    "functiontype",
    "shading",
    "shadingtype",
    "bitspersample",
    "coords",
    "procset",
    "subtype",
    "range",
    "domain",
    "true",
    "false",
    "layout",
    "contents",
    "parent",
    "type",
    "size",
    "length",
    "group",
    "font",
    "https",
}
DEFAULT_TOPIC_BLACKLIST = {
    "naide",
    "definitsioon",
    "tahendab",
    "course",
    "learning",
    "introduction",
    "introductory",
    "slides",
    "slide",
    "exam",
    "lecture",
    "loeng",
    "peatukk",
    "teema",
    "ulesanne",
    "ulesanded",
    "praktikum",
    "materjal",
    "materjalid",
    "peatuub",
    "oppimine",
    "oppematerjal",
    "general",
    "ica0019",
    "fall",
    "kevad",
    "sygis",
    "kevadsemester",
    "sugissemester",
}


@dataclass
class ChunkRecord:
    chunk_id: str
    source: str
    course: str
    text: str
    concepts: list[str]
    page: int | None


@dataclass
class EmbeddingBackend:
    model_name: str
    model: Any


def normalize_token(value: str) -> str:
    normalized = value.lower()
    replacements = {
        "õ": "o",
        "ä": "a",
        "ö": "o",
        "ü": "u",
        "š": "s",
        "ž": "z",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return normalized


def slugify(value: str) -> str:
    normalized = normalize_token(value)
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized)
    return normalized.strip("-") or "item"


def tokenize_terms(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-zÕÄÖÜŠŽõäöüšž0-9-]{4,}", text.lower())
    normalized = [normalize_token(token) for token in tokens]
    return [token for token in normalized if is_valid_term(token)]


def extract_query_terms(query: str, synonyms: dict[str, list[str]] | None = None) -> list[str]:
    raw_tokens = re.findall(r"[A-Za-zÕÄÖÜŠŽõäöüšž0-9-]{2,}", query.lower())
    normalized = [normalize_token(token) for token in raw_tokens]

    terms: list[str] = []
    seen: set[str] = set()
    for token in normalized:
        if token in seen:
            continue
        seen.add(token)
        if is_valid_term(token):
            terms.append(token)

    if synonyms:
        for token in list(terms):
            for synonym in synonyms.get(token, []):
                if synonym not in seen and is_valid_term(synonym):
                    seen.add(synonym)
                    terms.append(synonym)

    if terms:
        return terms

    fallback_terms: list[str] = []
    for token in normalized:
        if token in {"ja", "ning", "ehk", "voi", "mis", "kas"}:
            continue
        if token not in fallback_terms:
            fallback_terms.append(token)
    return fallback_terms[:6]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    clean_text = re.sub(r"\s+", " ", text).strip()
    if not clean_text:
        return []

    chunks: list[str] = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(clean_text):
        chunk = clean_text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def extract_candidate_terms(text: str, limit: int = 8) -> list[str]:
    counts = Counter(tokenize_terms(text))
    return [token for token, _ in counts.most_common(limit)]


def is_valid_term(token: str) -> bool:
    if token in STOPWORDS or token in NOISE_TERMS:
        return False
    if token.isdigit():
        return False
    if len(token) > 24:
        return False
    if re.search(r"(.)\1\1", token):
        return False
    if not re.search(r"[aeiou]", token):
        return False
    return True


def snippet(text: str, length: int = 220) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact[:length].rstrip() + ("..." if len(compact) > length else "")


def source_title(path: str, data_dir: Path) -> str:
    try:
        relative_path = Path(path).resolve().relative_to(data_dir.resolve())
    except ValueError:
        relative_path = Path(path).name
    return relative_path.as_posix() if isinstance(relative_path, Path) else str(relative_path)


def detect_course(path: str, data_dir: Path) -> str:
    try:
        relative_path = Path(path).resolve().relative_to(data_dir.resolve())
        parts = relative_path.parts
        if len(parts) >= 2:
            return parts[0]
    except ValueError:
        pass
    return "general"


def load_source_documents(data_dir: Path) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            reader = PdfReader(str(path))
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                cleaned = re.sub(r"\s+", " ", text).strip()
                if cleaned:
                    documents.append({"path": str(path), "text": cleaned, "page": page_number})
        elif suffix in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            cleaned = re.sub(r"\s+", " ", text).strip()
            if cleaned:
                documents.append({"path": str(path), "text": cleaned, "page": None})
    return documents


def load_query_synonyms(base_dir: Path) -> dict[str, list[str]]:
    config_path = base_dir / "query_synonyms.json"
    synonyms = {key: list(value) for key, value in DEFAULT_SYNONYMS.items()}
    if not config_path.exists():
        return synonyms

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    for key, values in raw.items():
        normalized_key = normalize_token(key)
        synonyms[normalized_key] = [normalize_token(value) for value in values]
    return synonyms


def load_topic_blacklist(base_dir: Path) -> set[str]:
    blacklist = set(DEFAULT_TOPIC_BLACKLIST)
    config_path = base_dir / "topic_blacklist.json"
    if not config_path.exists():
        return blacklist

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        blacklist.update(normalize_token(str(item)) for item in raw)
    return blacklist


def is_good_topic_term(token: str, blacklist: set[str]) -> bool:
    normalized = normalize_token(token)
    if normalized in blacklist:
        return False
    if len(normalized) < 5:
        return False
    if normalized.endswith(("2025", "2026")):
        return False
    return is_valid_term(normalized)


def build_knowledge_base(data_dir: Path, generated_dir: Path) -> dict[str, Any]:
    docs = load_source_documents(data_dir)
    if not docs:
        raise ValueError(f"`data` kaust on tühi: {data_dir}")
    topic_blacklist = load_topic_blacklist(data_dir.parent)

    chunks: list[ChunkRecord] = []
    source_index: dict[str, dict[str, Any]] = {}
    raw_chunk_concepts: dict[str, list[str]] = {}
    global_concepts: Counter[str] = Counter()
    course_chunks: dict[str, list[str]] = defaultdict(list)

    for doc in docs:
        raw_path = doc["path"]
        source = source_title(raw_path, data_dir)
        course = detect_course(raw_path, data_dir)
        doc_chunks = chunk_text(doc["text"])
        if source not in source_index:
            source_index[source] = {
                "path": raw_path,
                "course": course,
                "chunk_ids": [],
                "concepts": [],
                "chunk_count": 0,
                "pages": [],
            }

        source_index[source]["chunk_count"] += len(doc_chunks)
        page = doc.get("page")
        if page is not None and page not in source_index[source]["pages"]:
            source_index[source]["pages"].append(page)

        for position, text in enumerate(doc_chunks, start=1):
            concepts = extract_candidate_terms(text)
            page_label = f"p{page:03d}-" if page is not None else ""
            chunk_id = f"{slugify(source)}-{page_label}{position:03d}"
            record = ChunkRecord(
                chunk_id=chunk_id,
                source=source,
                course=course,
                text=text,
                concepts=concepts,
                page=page,
            )
            chunks.append(record)
            source_index[source]["chunk_ids"].append(chunk_id)
            raw_chunk_concepts[chunk_id] = concepts
            global_concepts.update(set(concepts))
            course_chunks[course].append(chunk_id)

    allowed_concepts = {
        concept
        for concept, count in global_concepts.items()
        if count >= 3 and count <= max(10, len(chunks) // 2) and is_good_topic_term(concept, topic_blacklist)
    }

    concept_index: dict[str, list[str]] = defaultdict(list)
    source_concepts: dict[str, Counter[str]] = defaultdict(Counter)
    for chunk in chunks:
        filtered = [concept for concept in raw_chunk_concepts[chunk.chunk_id] if concept in allowed_concepts][:5]
        chunk.concepts = filtered
        source_concepts[chunk.source].update(filtered)
        for concept in filtered:
            concept_index[concept].append(chunk.chunk_id)

    for source in source_index:
        source_index[source]["pages"].sort()
        source_index[source]["concepts"] = [name for name, _ in source_concepts[source].most_common(12)]

    concept_items: dict[str, dict[str, Any]] = {}
    chunk_lookup = {chunk.chunk_id: chunk for chunk in chunks}
    for concept, chunk_ids in concept_index.items():
        sources = sorted({chunk_lookup[chunk_id].source for chunk_id in chunk_ids})
        courses = sorted({chunk_lookup[chunk_id].course for chunk_id in chunk_ids})
        concept_items[concept] = {
            "sources": sources,
            "courses": courses,
            "chunk_ids": chunk_ids[:20],
            "count": len(chunk_ids),
        }

    course_items = build_course_catalog(course_chunks, chunk_lookup, topic_blacklist=topic_blacklist)

    prepare_generated_dir(generated_dir)

    write_source_markdown(generated_dir, source_index, chunk_lookup)
    write_concept_markdown(generated_dir, concept_items, chunk_lookup)
    write_course_markdown(generated_dir, course_items)
    write_overview_markdown_v2(generated_dir, source_index, concept_items, course_items)

    catalog = {
        "courses": course_items,
        "sources": source_index,
        "concepts": concept_items,
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "course": chunk.course,
                "text": chunk.text,
                "concepts": chunk.concepts,
                "page": chunk.page,
            }
            for chunk in chunks
        ],
    }
    (generated_dir / "catalog.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return catalog


def prepare_generated_dir(generated_dir: Path) -> None:
    generated_dir.mkdir(parents=True, exist_ok=True)

    for filename in ("catalog.json", "overview.md"):
        target = generated_dir / filename
        if target.exists():
            target.unlink()

    for dirname in ("sources", "concepts", "courses"):
        target_dir = generated_dir / dirname
        target_dir.mkdir(exist_ok=True)
        for file_path in target_dir.glob("*.md"):
            file_path.unlink()


def build_search_index(catalog: dict[str, Any], base_dir: Path | None = None) -> dict[str, Any]:
    texts = [chunk["text"] for chunk in catalog["chunks"]]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    char_matrix = char_vectorizer.fit_transform(texts)
    chunk_terms = [set(tokenize_terms(text)) for text in texts]
    chunk_concepts = [set(chunk.get("concepts", [])) for chunk in catalog["chunks"]]
    synonyms = load_query_synonyms(base_dir or Path.cwd())
    embedding_backend = load_embedding_backend()
    semantic_matrix = build_semantic_matrix(texts, embedding_backend)
    return {
        "catalog": catalog,
        "vectorizer": vectorizer,
        "matrix": matrix,
        "char_vectorizer": char_vectorizer,
        "char_matrix": char_matrix,
        "chunk_terms": chunk_terms,
        "chunk_concepts": chunk_concepts,
        "synonyms": synonyms,
        "embedding_backend": embedding_backend,
        "semantic_matrix": semantic_matrix,
    }


def load_embedding_backend() -> EmbeddingBackend | None:
    if DEFAULT_EMBEDDING_MODEL in _EMBEDDING_BACKEND_CACHE:
        return _EMBEDDING_BACKEND_CACHE[DEFAULT_EMBEDDING_MODEL]

    if SentenceTransformer is None:
        _EMBEDDING_BACKEND_CACHE[DEFAULT_EMBEDDING_MODEL] = None
        return None

    try:
        model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
    except Exception:
        _EMBEDDING_BACKEND_CACHE[DEFAULT_EMBEDDING_MODEL] = None
        return None

    backend = EmbeddingBackend(model_name=DEFAULT_EMBEDDING_MODEL, model=model)
    _EMBEDDING_BACKEND_CACHE[DEFAULT_EMBEDDING_MODEL] = backend
    return backend


def build_semantic_matrix(texts: list[str], backend: EmbeddingBackend | None) -> np.ndarray | None:
    if backend is None or not texts:
        return None

    try:
        embeddings = backend.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    except Exception:
        return None

    return np.asarray(embeddings, dtype=np.float32)


def semantic_query_scores(
    query: str,
    backend: EmbeddingBackend | None,
    semantic_matrix: np.ndarray | None,
) -> np.ndarray | None:
    if backend is None or semantic_matrix is None or semantic_matrix.size == 0:
        return None

    try:
        query_embedding = backend.model.encode([query], normalize_embeddings=True, show_progress_bar=False)
    except Exception:
        return None

    query_vector = np.asarray(query_embedding, dtype=np.float32)[0]
    return semantic_matrix @ query_vector


def build_course_catalog(
    course_chunks: dict[str, list[str]],
    chunk_lookup: dict[str, ChunkRecord],
    main_topic_limit: int = 7,
    subtopic_limit: int = 8,
    topic_blacklist: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    course_items: dict[str, dict[str, Any]] = {}
    topic_blacklist = topic_blacklist or set()

    for course, chunk_ids in sorted(course_chunks.items()):
        concept_counts: Counter[str] = Counter()
        sources = sorted({chunk_lookup[chunk_id].source for chunk_id in chunk_ids})
        for chunk_id in chunk_ids:
            concept_counts.update(chunk_lookup[chunk_id].concepts)

        main_topics = [
            concept
            for concept, count in concept_counts.most_common()
            if count >= 3 and is_good_topic_term(concept, topic_blacklist)
        ][:main_topic_limit]
        topic_items: dict[str, dict[str, Any]] = {}
        for topic in main_topics:
            related_chunk_ids = [chunk_id for chunk_id in chunk_ids if topic in chunk_lookup[chunk_id].concepts]
            subtopic_counts: Counter[str] = Counter()
            topic_sources = sorted({chunk_lookup[chunk_id].source for chunk_id in related_chunk_ids})
            for chunk_id in related_chunk_ids:
                for concept in chunk_lookup[chunk_id].concepts:
                    if concept != topic and concept not in main_topics and is_good_topic_term(concept, topic_blacklist):
                        subtopic_counts[concept] += 1

            topic_items[topic] = {
                "count": concept_counts[topic],
                "subtopics": [concept for concept, _ in subtopic_counts.most_common(subtopic_limit)],
                "sources": topic_sources,
                "chunk_ids": related_chunk_ids[:20],
            }

        course_items[course] = {
            "sources": sources,
            "concepts": [
                concept
                for concept, _ in concept_counts.most_common()
                if is_good_topic_term(concept, topic_blacklist)
            ][:20],
            "topics": topic_items,
            "chunk_count": len(chunk_ids),
        }

    return course_items


def write_source_markdown(
    generated_dir: Path,
    source_index: dict[str, dict[str, Any]],
    chunk_lookup: dict[str, ChunkRecord],
) -> None:
    for source, info in source_index.items():
        concept_lines = "\n".join(f"- [[{concept}]]" for concept in info["concepts"]) or "- Puudub"
        quote_lines = "\n".join(
            f"- `{chunk_id}`{format_page_suffix(chunk_lookup[chunk_id].page)}: {snippet(chunk_lookup[chunk_id].text)}"
            for chunk_id in info["chunk_ids"][:8]
        )
        content = (
            f"# {source}\n\n"
            f"## Asukoht\n"
            f"- Fail: `{info['path']}`\n"
            f"- Tükke: {info['chunk_count']}\n"
            f"- Leheküljed: {format_pages(info['pages'])}\n\n"
            f"## Peamised konseptsioonid\n"
            f"{concept_lines}\n\n"
            f"## Viite-lõigud\n"
            f"{quote_lines}\n"
        )
        target = generated_dir / "sources" / f"{slugify(source)}.md"
        target.write_text(content, encoding="utf-8")


def write_concept_markdown(
    generated_dir: Path,
    concept_items: dict[str, dict[str, Any]],
    chunk_lookup: dict[str, ChunkRecord],
) -> None:
    for concept, info in concept_items.items():
        source_lines = "\n".join(f"- [[{source}]]" for source in info["sources"]) or "- Puudub"
        quote_lines = "\n".join(
            f"- `{chunk_id}` ({chunk_lookup[chunk_id].source}{format_page_suffix(chunk_lookup[chunk_id].page)}): {snippet(chunk_lookup[chunk_id].text)}"
            for chunk_id in info["chunk_ids"][:6]
        )
        content = (
            f"# {concept}\n\n"
            f"## Seotud allikad\n"
            f"{source_lines}\n\n"
            f"## Viite-lõigud\n"
            f"{quote_lines}\n"
        )
        target = generated_dir / "concepts" / f"{slugify(concept)}.md"
        target.write_text(content, encoding="utf-8")


def write_course_markdown(generated_dir: Path, course_items: dict[str, dict[str, Any]]) -> None:
    for course, info in course_items.items():
        topic_lines: list[str] = []
        for topic, topic_info in info["topics"].items():
            subtopics = ", ".join(topic_info["subtopics"][:5]) or "puudub"
            topic_lines.append(f"- **{topic}** ({topic_info['count']} viidet) | alamteemad: {subtopics}")

        source_lines = "\n".join(f"- [[{source}]]" for source in info["sources"]) or "- Puudub"
        content = (
            f"# {course}\n\n"
            f"## Allikad\n"
            f"{source_lines}\n\n"
            f"## Peateemad\n"
            f"{chr(10).join(topic_lines) or '- Puudub'}\n"
        )
        target = generated_dir / "courses" / f"{slugify(course)}.md"
        target.write_text(content, encoding="utf-8")


def write_overview_markdown(
    generated_dir: Path,
    source_index: dict[str, dict[str, Any]],
    concept_items: dict[str, dict[str, Any]],
) -> None:
    source_lines = "\n".join(
        f"- [[{source}]]: {', '.join(info['concepts'][:5])}"
        for source, info in sorted(source_index.items())
    )
    concept_lines = "\n".join(
        f"- [[{concept}]] ({info['count']} viidet)"
        for concept, info in sorted(concept_items.items(), key=lambda item: item[1]["count"], reverse=True)[:40]
    )
    content = (
        "# Õppematerjalide ülevaade\n\n"
        "## Allikad\n"
        f"{source_lines}\n\n"
        "## Top konseptsioonid\n"
        f"{concept_lines}\n"
    )
    (generated_dir / "overview.md").write_text(content, encoding="utf-8")


def write_overview_markdown_v2(
    generated_dir: Path,
    source_index: dict[str, dict[str, Any]],
    concept_items: dict[str, dict[str, Any]],
    course_items: dict[str, dict[str, Any]],
) -> None:
    course_lines = "\n".join(
        f"- [[{course}]]: {', '.join(info['concepts'][:5])}"
        for course, info in sorted(course_items.items())
    )
    source_lines = "\n".join(
        f"- [[{source}]]: {', '.join(info['concepts'][:5])}"
        for source, info in sorted(source_index.items())
    )
    concept_lines = "\n".join(
        f"- [[{concept}]] ({info['count']} viidet)"
        for concept, info in sorted(concept_items.items(), key=lambda item: item[1]["count"], reverse=True)[:40]
    )
    content = (
        "# Oppematerjalide ulevaade\n\n"
        "## Ained\n"
        f"{course_lines}\n\n"
        "## Allikad\n"
        f"{source_lines}\n\n"
        "## Top konseptsioonid\n"
        f"{concept_lines}\n"
    )
    (generated_dir / "overview.md").write_text(content, encoding="utf-8")


def load_catalog(generated_dir: Path) -> dict[str, Any]:
    catalog_path = generated_dir / "catalog.json"
    if not catalog_path.exists():
        raise FileNotFoundError(f"Kataloogi ei leitud: {catalog_path}")
    return json.loads(catalog_path.read_text(encoding="utf-8"))


def search_catalog(
    search_index: dict[str, Any],
    query: str,
    top_k: int = 5,
    course: str | None = None,
    main_topic: str | None = None,
    subtopic: str | None = None,
    sources: list[str] | None = None,
) -> list[dict[str, Any]]:
    catalog = search_index["catalog"]
    min_score = 0.12
    normalized_query = re.sub(r"\s+", " ", query).strip()
    if not normalized_query:
        return []

    expanded_query = expand_query(normalized_query, search_index["synonyms"])
    query_terms = set(extract_query_terms(normalized_query, search_index["synonyms"]))
    lexical_query = " ".join(query_terms) if query_terms else expanded_query

    query_vector = search_index["vectorizer"].transform([lexical_query])
    similarities = cosine_similarity(query_vector, search_index["matrix"]).flatten()
    char_query_vector = search_index["char_vectorizer"].transform([expanded_query])
    char_similarities = cosine_similarity(char_query_vector, search_index["char_matrix"]).flatten()
    semantic_similarities = semantic_query_scores(
        expanded_query,
        search_index.get("embedding_backend"),
        search_index.get("semantic_matrix"),
    )

    scored_indexes: list[tuple[float, int]] = []
    for idx, chunk in enumerate(catalog["chunks"]):
        if course and chunk.get("course") != course:
            continue
        if sources and chunk.get("source") not in sources:
            continue
        chunk_concepts = search_index["chunk_concepts"][idx]
        if main_topic and main_topic not in chunk_concepts:
            continue
        if subtopic and subtopic not in chunk_concepts:
            continue
        overlap = len(query_terms & search_index["chunk_terms"][idx])
        concept_overlap = len(query_terms & chunk_concepts)
        partial_overlap = sum(
            1
            for term in query_terms
            if any(term in chunk_term or chunk_term in term for chunk_term in search_index["chunk_terms"][idx])
        )
        score = (
            float(similarities[idx]) * 0.35
            + float(char_similarities[idx]) * 0.15
            + (float(semantic_similarities[idx]) * 0.35 if semantic_similarities is not None else 0.0)
            + overlap * 0.12
            + concept_overlap * 0.16
            + partial_overlap * 0.05
        )
        scored_indexes.append((score, idx))

    ranked_indexes = [idx for _, idx in sorted(scored_indexes, reverse=True)]
    score_map = {idx: score for score, idx in scored_indexes}
    matches: list[dict[str, Any]] = []
    for idx in ranked_indexes:
        score = score_map[idx]
        if score < min_score:
            continue
        chunk = catalog["chunks"][idx]
        matches.append(
            {
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "course": chunk.get("course"),
                "text": chunk["text"],
                "concepts": chunk["concepts"],
                "page": chunk.get("page"),
                "score": round(float(score), 3),
            }
        )
        if len(matches) >= top_k:
            break
    return matches


def concept_matches(
    catalog: dict[str, Any],
    query: str,
    base_dir: Path | None = None,
    limit: int = 10,
    course: str | None = None,
) -> list[dict[str, Any]]:
    synonyms = load_query_synonyms(base_dir or Path.cwd())
    expanded_query = expand_query(query, synonyms)
    query_terms = set(extract_query_terms(expanded_query, synonyms))
    scored_matches: list[dict[str, Any]] = []

    for name, info in catalog["concepts"].items():
        if course and course not in info.get("courses", []):
            continue
        concept_terms = set(tokenize_terms(name))
        overlap = len(query_terms & concept_terms)
        partial_overlap = sum(1 for term in query_terms if term in name or name in term)
        score = overlap * 2 + partial_overlap
        if score > 0:
            scored_matches.append({"concept": name, "score": score, **info})

    scored_matches.sort(key=lambda item: (item["score"], item["count"]), reverse=True)
    return scored_matches[:limit]


def format_page_suffix(page: int | None) -> str:
    return f", lk {page}" if page is not None else ""


def format_pages(pages: list[int]) -> str:
    if not pages:
        return "pole määratud"
    if len(pages) == 1:
        return str(pages[0])
    return f"{pages[0]}-{pages[-1]}"


def expand_query(query: str, synonyms: dict[str, list[str]]) -> str:
    tokens = [normalize_token(token) for token in re.findall(r"[A-Za-zÕÄÖÜŠŽõäöüšž0-9-]+", query.lower())]
    expanded = [query]
    for token in tokens:
        expanded.extend(synonyms.get(token, []))
    return " ".join(expanded)
