from pathlib import Path

from knowledge_base import build_knowledge_base


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
GENERATED_DIR = BASE_DIR / "generated"


def main() -> None:
    catalog = build_knowledge_base(DATA_DIR, GENERATED_DIR)
    print(f"Allikaid: {len(catalog['sources'])}")
    print(f"Konseptsioone: {len(catalog['concepts'])}")
    print(f"Tükke: {len(catalog['chunks'])}")
    print(f"Väljund: {GENERATED_DIR}")


if __name__ == "__main__":
    main()
