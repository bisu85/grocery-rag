import json
import time

import httpx

# (search term, cuisine tag) — things an Indian foodie in NL actually buys
TERMS = [
    ("paneer", "Indian"),
    ("basmati rice", "Indian"),
    ("garam masala", "Indian"),
    ("coconut milk", "Indian"),
    ("red lentils", "Indian"),
    ("chickpeas", "Indian"),
    ("greek yogurt", "general"),
]
BASE = "https://world.openfoodfacts.org/cgi/search.pl"
HEADERS = {"User-Agent": "grocery-rag-learning/0.1 (student learning project)"}
PER_TERM = 5


def map_dietary(product: dict) -> list[str]:
    """Translate OFF's label/analysis tags into our dietary_tags vocabulary."""
    tags = set()
    sources = (product.get("labels_tags") or []) + (product.get("ingredients_analysis_tags") or [])
    for t in sources:
        if t == "en:vegan":
            tags.add("vegan")
        elif t == "en:vegetarian":
            tags.add("vegetarian")
        elif t == "en:gluten-free":
            tags.add("gluten-free")
        elif t == "en:halal":
            tags.add("halal")
    return sorted(tags)


def main() -> None:
    records: dict[str, dict] = {}  # keyed by barcode -> dedupes automatically
    with httpx.Client(headers=HEADERS, timeout=30) as client:
        for term, cuisine in TERMS:
            params = {
                "search_terms": term,
                "search_simple": 1,
                "action": "process",
                "json": 1,
                "page_size": PER_TERM,
            }
            r = client.get(BASE, params=params)
            r.raise_for_status()
            products = r.json().get("products", [])
            for p in products:
                code = p.get("code")
                name = (p.get("product_name") or p.get("product_name_en") or "").strip()
                if not code or not name:  # crowdsourced data: skip incomplete rows
                    continue
                brand = (p.get("brands") or "").strip()
                qty = (p.get("quantity") or "").strip()
                text = ", ".join(x for x in [name, brand, qty] if x)
                records[code] = {
                    "doc_type": "product",
                    "text": text,
                    "store": None,  # OFF has no reliable store
                    "price": None,  # OFF has no price -> comes via a tool later
                    "dietary_tags": map_dietary(p),
                    "cuisine": cuisine,
                }
            print(f"  '{term}': {len(products)} fetched")
            time.sleep(0.5)  # be polite to a free community API

    out = list(records.values())
    with open("data/products_off.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(out)} unique products to data/products_off.json")


if __name__ == "__main__":
    main()