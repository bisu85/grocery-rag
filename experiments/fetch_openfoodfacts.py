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
BASE = "https://nl.openfoodfacts.org/cgi/search.pl"   # country instance = less hammered
HEADERS = {"User-Agent": "grocery-rag-learning/0.1 (student learning project)"}
PER_TERM = 5
MAX_RETRIES = 4


def get_with_retry(client: httpx.Client, url: str, params: dict) -> httpx.Response | None:
    """GET with exponential backoff on transient errors (503/429/timeouts).

    Returns the response, or None if it never succeeded."""
    for attempt in range(MAX_RETRIES):
        try:
            r = client.get(url, params=params)
            if r.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError("transient", request=r.request, response=r)
            r.raise_for_status()
            return r
        except (httpx.HTTPStatusError, httpx.TransportError) as e:
            wait = 2 ** attempt  # 1, 2, 4, 8 seconds
            print(f"    attempt {attempt + 1} failed ({e.__class__.__name__}); retrying in {wait}s")
            time.sleep(wait)
    return None


def map_dietary(product: dict) -> list[str]:
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
    records: dict[str, dict] = {}
    with httpx.Client(headers=HEADERS, timeout=30) as client:
        for term, cuisine in TERMS:
            params = {
                "search_terms": term,
                "search_simple": 1,
                "action": "process",
                "json": 1,
                "page_size": PER_TERM,
            }
            r = get_with_retry(client, BASE, params)
            if r is None:
                print(f"  '{term}': SKIPPED after {MAX_RETRIES} attempts")
                continue

            products = r.json().get("products", [])
            for p in products:
                code = p.get("code")
                name = (p.get("product_name") or p.get("product_name_en") or "").strip()
                if not code or not name:
                    continue
                brand = (p.get("brands") or "").strip()
                qty = (p.get("quantity") or "").strip()
                text = ", ".join(x for x in [name, brand, qty] if x)
                records[code] = {
                    "doc_type": "product",
                    "text": text,
                    "store": None,
                    "price": None,
                    "dietary_tags": map_dietary(p),
                    "cuisine": cuisine,
                }
            print(f"  '{term}': {len(products)} fetched")
            time.sleep(1.0)  # a bit slower between terms, to stay under the rate limit

    out = list(records.values())
    with open("data/products_off.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(out)} unique products to data/products_off.json")


if __name__ == "__main__":
    main()