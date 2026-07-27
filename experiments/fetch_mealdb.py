import json
import time

import httpx

BASE = "https://www.themealdb.com/api/json/v1/1"
HEADERS = {"User-Agent": "grocery-rag-learning/0.1 (student learning project)"}
MAX_RECIPES = 15
MAX_RETRIES = 4


def get_with_retry(client: httpx.Client, url: str, params: dict) -> httpx.Response | None:
    for attempt in range(MAX_RETRIES):
        try:
            r = client.get(url, params=params)
            if r.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError("transient", request=r.request, response=r)
            r.raise_for_status()
            return r
        except (httpx.HTTPStatusError, httpx.TransportError) as e:
            wait = 2 ** attempt
            print(f"    attempt {attempt + 1} failed ({e.__class__.__name__}); retry in {wait}s")
            time.sleep(wait)
    return None


def ingredients_of(meal: dict) -> list[str]:
    items = []
    for i in range(1, 21):
        ing = (meal.get(f"strIngredient{i}") or "").strip()
        meas = (meal.get(f"strMeasure{i}") or "").strip()
        if ing:
            items.append(f"{meas} {ing}".strip())
    return items


def main() -> None:
    with httpx.Client(headers=HEADERS, timeout=30) as client:
        r = get_with_retry(client, f"{BASE}/filter.php", {"a": "Indian"})
        if r is None:
            print("Could not reach TheMealDB after retries. Try again shortly.")
            return

        meals = r.json().get("meals") or []
        if not meals:
            print("TheMealDB returned an EMPTY meal list (meals: null). "
                  "This is usually transient — wait a minute and re-run.")
            return

        meals = meals[:MAX_RECIPES]
        print(f"Found {len(r.json()['meals'])} Indian meals; using {len(meals)}.")

        records = []
        for m in meals:
            d = get_with_retry(client, f"{BASE}/lookup.php", {"i": m["idMeal"]})
            if d is None:
                print(f"  ! skipped id {m['idMeal']} (lookup failed)")
                continue
            full = (d.json().get("meals") or [None])[0]
            if not full:
                continue
            name = (full.get("strMeal") or "").strip()
            ings = ingredients_of(full)
            instr = (full.get("strInstructions") or "").strip().replace("\r\n", " ")
            text = (
                f"{name}: Indian {full.get('strCategory', 'dish')}. "
                f"Ingredients: {', '.join(ings)}. {instr[:250]}"
            )
            records.append({
                "doc_type": "recipe",
                "text": text,
                "store": None,
                "price": None,
                "dietary_tags": [],
                "cuisine": "Indian",
            })
            print(f"  + {name}")
            time.sleep(0.3)

    # only overwrite the file if we actually got recipes — don't clobber good data with 0
    if records:
        with open("data/recipes_mealdb.json", "w") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(records)} recipes to data/recipes_mealdb.json")
    else:
        print("No recipes collected; left existing file untouched.")


if __name__ == "__main__":
    main()