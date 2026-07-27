import json
import re

JUNK_MARKERS = ("nutri-score", "http", "www.")
MIN_LEN = 8


def is_valid(rec: dict) -> tuple[bool, str]:
    """Return (keep?, reason-if-dropped) for one record."""
    text = (rec.get("text") or "").strip()

    if len(text) < MIN_LEN:
        return False, "too short"

    low = text.lower()
    if any(marker in low for marker in JUNK_MARKERS):
        return False, "junk marker"

    # reject text that's mostly SHOUTING/garbled: >70% uppercase letters
    letters = [c for c in text if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.7:
        return False, "mostly uppercase noise"

    return True, ""


def clean_file(path: str) -> None:
    with open(path) as f:
        records = json.load(f)

    kept, dropped = [], []
    seen_texts = set()
    for rec in records:
        ok, reason = is_valid(rec)
        if not ok:
            dropped.append((rec.get("text", "")[:50], reason))
            continue
        # de-dupe on a normalised version of the text
        key = re.sub(r"\s+", " ", (rec["text"] or "").lower()).strip()
        if key in seen_texts:
            dropped.append((rec.get("text", "")[:50], "duplicate"))
            continue
        seen_texts.add(key)
        kept.append(rec)

    out_path = path.replace(".json", ".clean.json")
    with open(out_path, "w") as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)

    print(f"\n{path}:  kept {len(kept)}, dropped {len(dropped)}  ->  {out_path}")
    for text, reason in dropped:
        print(f"    dropped [{reason}]: {text}")


if __name__ == "__main__":
    for p in ("data/products_off.json", "data/recipes_mealdb.json"):
        try:
            clean_file(p)
        except FileNotFoundError:
            print(f"(skipping {p} — not found)")