#!/usr/bin/env python3
"""Debug script to check what ChromaDB retrieves for test queries."""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from granite_switch.tutorials.chroma_loader import load_or_build_govt_chroma


def check_retrieval(chroma_path: Path, queries: list[str], top_k: int = 5):
    """Check what documents are retrieved for given queries."""
    # Load collection with correct embedding function (same as rag_101.ipynb)
    print("Loading ChromaDB with correct embedding function...")
    collection = load_or_build_govt_chroma(
        chroma_path=str(chroma_path),
        load_only_tutorial_docs=False,
        query_device="cpu",
    )
    print(f"Collection 'govt' has {collection.count():,} documents\n")
    print("=" * 80)

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 80)

        results = collection.query(query_texts=[query], n_results=top_k)
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]

        print(f"Retrieved {len(documents)} documents:\n")
        for i, (doc, dist) in enumerate(zip(documents, distances), 1):
            print(f"{i}. Distance: {dist:.4f}")
            print(f"   Text: {doc[:200]}...")
            print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chroma-path", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    # Test queries from the realistic set
    test_queries = [
        "How do I submit a Freedom of Information Act request to the VA?",
        "What medical services are available at VA hospitals?",
        "What causes the seasons on Earth?",
        "What are the CDC guidelines for COVID-19 biosafety?",
    ]

    check_retrieval(args.chroma_path, test_queries, args.top_k)
