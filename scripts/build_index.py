"""Build (or rebuild) all per-store indexes from data/seed/.

Usage:  python -m scripts.build_index
"""
from src.ingestion.indexer import build_all

if __name__ == "__main__":
    print("Building indexes from seed corpus...")
    build_all()
    print("Done.")
