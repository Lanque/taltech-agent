from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
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


@dataclass
class ChunkRecord:
    chunk_id: str
    source: str
    text: str
    concepts: list[str]
    page: int | None


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


def build_knowledge_base(data_dir: Path, generated_dir: Path) -> dict[str, Any]:
    docs = load_source_documents(data_dir)
    if not docs:
        raise ValueError(f"`data` kaust on tühi: {data_dir}")

    chunks: list[ChunkRecord] = []
    source_index: dict[str, dict[str, Any]] = {}
    raw_chunk_concepts: dict[str, list[str]] = {}
    global_concepts: Counter[str] = Counter()

    for doc in docs:
        raw_path = doc["path"]
        source = source_title(raw_path, data_dir)
        doc_chunks = chunk_text(doc["text"])
        if source not in source_index:
            source_index[source] = {
                "path": raw_path,
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
                text=text,
                concepts=concepts,
                page=page,
            )
            chunks.append(record)
            source_index[source]["chunk_ids"].append(chunk_id)
            raw_chunk_concepts[chunk_id] = concepts
            global_concepts.update(set(concepts))

    allowed_concepts = {
        concept
        for concept, count in global_concepts.items()
        if count >= 3 and count <= max(10, len(chunks) // 2)
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
        concept_items[concept] = {
            "sources": sources,
            "chunk_ids": chunk_ids[:20],
            "count": len(chunk_ids),
        }

    prepare_generated_dir(generated_dir)

    write_source_markdown(generated_dir, source_index, chunk_lookup)
    write_concept_markdown(generated_dir, concept_items, chunk_lookup)
    write_overview_markdown(generated_dir, source_index, concept_items)

    catalog = {
        "sources": source_index,
        "concepts": concept_items,
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
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

    for dirname in ("sources", "concepts"):
        target_dir = generated_dir / dirname
        target_dir.mkdir(exist_ok=True)
        for file_path in target_dir.glob("*.md"):
            file_path.unlink()


def build_search_index(catalog: dict[str, Any], base_dir: Path | None = None) -> dict[str, Any]:
    texts = [chunk["text"] for chunk in catalog["chunks"]]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)
    chunk_terms = [set(tokenize_terms(text)) for text in texts]
    chunk_concepts = [set(chunk.get("concepts", [])) for chunk in catalog["chunks"]]
    synonyms = load_query_synonyms(base_dir or Path.cwd())
    return {
        "catalog": catalog,
        "vectorizer": vectorizer,
        "matrix": matrix,
        "chunk_terms": chunk_terms,
        "chunk_concepts": chunk_concepts,
        "synonyms": synonyms,
    }


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


def load_catalog(generated_dir: Path) -> dict[str, Any]:
    catalog_path = generated_dir / "catalog.json"
    if not catalog_path.exists():
        raise FileNotFoundError(f"Kataloogi ei leitud: {catalog_path}")
    return json.loads(catalog_path.read_text(encoding="utf-8"))


def search_catalog(search_index: dict[str, Any], query: str, top_k: int = 5) -> list[dict[str, Any]]:
    catalog = search_index["catalog"]
    expanded_query = expand_query(query, search_index["synonyms"])
    query_vector = search_index["vectorizer"].transform([expanded_query])
    similarities = cosine_similarity(query_vector, search_index["matrix"]).flatten()
    query_terms = set(tokenize_terms(expanded_query))

    scored_indexes: list[tuple[float, int]] = []
    for idx, chunk in enumerate(catalog["chunks"]):
        overlap = len(query_terms & search_index["chunk_terms"][idx])
        concept_overlap = len(query_terms & search_index["chunk_concepts"][idx])
        score = float(similarities[idx]) + overlap * 0.08 + concept_overlap * 0.12
        scored_indexes.append((score, idx))

    ranked_indexes = [idx for _, idx in sorted(scored_indexes, reverse=True)]
    score_map = {idx: score for score, idx in scored_indexes}
    matches: list[dict[str, Any]] = []
    for idx in ranked_indexes:
        score = score_map[idx]
        if score <= 0:
            continue
        chunk = catalog["chunks"][idx]
        matches.append(
            {
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "text": chunk["text"],
                "concepts": chunk["concepts"],
                "page": chunk.get("page"),
                "score": round(float(score), 3),
            }
        )
        if len(matches) >= top_k:
            break
    return matches


def concept_matches(catalog: dict[str, Any], query: str, base_dir: Path | None = None, limit: int = 10) -> list[dict[str, Any]]:
    expanded_query = expand_query(query, load_query_synonyms(base_dir or Path.cwd()))
    query_terms = set(tokenize_terms(expanded_query))
    scored_matches: list[dict[str, Any]] = []

    for name, info in catalog["concepts"].items():
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
