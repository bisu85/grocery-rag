import json
import time

import httpx

BASE = "https://www.themealdb.com/api/json/v1/1"
HEADERS = {"User-Agent": "grocery-rag-learning/0.1 (student learning project)"}
MAX_RECIPES = 15


def ingredients_of(meal: dict) -> list[str]:
    """MealDB stores ingredients across strIngredient1..20 / strMeasure1..20."""
    items = []
    for i in range(1, 21):
        ing = (meal.get(f"strIngredient{i}") or "").strip()
        meas = (meal.get(f"strMeasure{i}") or "").strip()
        if ing:
            items.append(f"{meas} {ing}".strip())
    return items


def main() -> None:
    with httpx.Client(headers=HEADERS, timeout=30) as client:
        # 1) list Indian meals (basic info only: id, name, thumbnail)
        r = client.get(f"{BASE}/filter.php", params={"a": "Indian"})
        r.raise_for_status()
        meals = (r.json().get("meals") or [])[:MAX_RECIPES]
        print(f"Found Indian meals; using {len(meals)}.")

        records = []
        for m in meals:
            # 2) fetch full details by id
            d = client.get(f"{BASE}/lookup.php", params={"i": m["idMeal"]})
            d.raise_for_status()
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
                "dietary_tags": [],  # could enrich later from ingredients
                "cuisine": "Indian",
            })
            print(f"  + {name}")
            time.sleep(0.3)

    with open("data/recipes_mealdb.json", "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(records)} recipes to data/recipes_mealdb.json")


if __name__ == "__main__":
    main()