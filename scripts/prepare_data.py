"""Télécharge le jeu MultiHop-RAG (corpus + questions) dans data/.

Source : dépôt officiel https://github.com/yixuantt/MultiHop-RAG/ (dossier dataset/).
Best-effort : si le réseau est indisponible, affiche les instructions manuelles.

Usage :
    python scripts/prepare_data.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

RAW = "https://raw.githubusercontent.com/yixuantt/MultiHop-RAG/main/dataset"
FILES = {
    "corpus.json": f"{RAW}/corpus.json",
    "MultiHopRAG.json": f"{RAW}/MultiHopRAG.json",
}
DATA_DIR = Path("data")


def main() -> int:
    DATA_DIR.mkdir(exist_ok=True)
    ok = True
    for name, url in FILES.items():
        dest = DATA_DIR / name
        if dest.exists():
            print(f"= déjà présent : {dest}")
            continue
        try:
            print(f"↓ {url}")
            urllib.request.urlretrieve(url, dest)
            print(f"✔ écrit : {dest} ({dest.stat().st_size // 1024} Ko)")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"✗ échec {name} : {e}", file=sys.stderr)

    if not ok:
        print(
            "\nTéléchargement manuel :\n"
            "  1. Va sur https://github.com/yixuantt/MultiHop-RAG/tree/main/dataset\n"
            "  2. Récupère corpus.json et MultiHopRAG.json\n"
            "  3. Place-les dans ./data/\n",
            file=sys.stderr,
        )
        return 1
    print("\nPrêt. Lance :  python -m evaluation.run_eval configs/baseline.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
