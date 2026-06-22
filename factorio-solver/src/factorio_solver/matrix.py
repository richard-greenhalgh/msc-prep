# matrix.py
# use vector/matrix representation of items/recipes

import numpy as np
import csv
from pathlib import Path
from factorio_solver.recipes import Recipe, RecipeDB
from factorio_solver.solver import SolveResult

def build_recipe_matrix():
    unique_items = set()

    for recipe in RecipeDB.values():
        unique_items.update(recipe.inputs)
        unique_items.update(recipe.outputs)

    items = sorted(unique_items)
    recipes = sorted(RecipeDB)

    item_id = {item: i for i, item in enumerate(items)}
    recipe_id = {recipe: j for j, recipe in enumerate(recipes)}

    matrix = np.zeros((len(items), len(recipes)), dtype=np.float64)

    for recipe_name, recipe in RecipeDB.items():
        col = recipe_id[recipe_name]

        for item, quantity in recipe.inputs.items():
            matrix[item_id[item], col] -= quantity

        for item, quantity in recipe.outputs.items():
            matrix[item_id[item], col] += quantity

    return {
        "items": items,
        "recipes": recipes,
        "item_id": item_id,
        "recipe_id": recipe_id,
        "matrix": matrix,
    }

def verify_balance(result:SolveResult, target_item:str, target_rate:float):
    data = build_recipe_matrix()
    item_id, recipe_id = data["item_id"], data["recipe_id"]

    # create vector of recipe rates
    recipe_rates = np.zeros(  len(data["recipes"])  )
    for recipe, rate in result.recipe_rates.items():
        recipe_rates[recipe_id[recipe]] = rate
    
    # Actual net item flow: A @ x
    net_rates = data["matrix"] @ recipe_rates

    expected = np.zeros_like(net_rates)
    expected[item_id[target_item]] = target_rate

    for raw_item, raw_rate in result.raw_rates.items():
        expected[item_id[raw_item]] = -raw_rate

    # Residual: actual - expected
    residual = net_rates - expected

    # output non-zero net rates
    tolerance = 1e-9
    max_residual = np.max(np.abs(residual))

    print("\nNet flow test vs. expected:")
    if max_residual < tolerance:
        print(f"  PASS (max residual: {max_residual:.2e})")
    else:
        print(f"  FAIL (max residual: {max_residual:.2e})")

        print("  Residuals by item:")
        for i, value in enumerate(residual):
            if abs(value) >= tolerance:
                print(f"    {data['items'][i]:<30}: {value:8.2f}")
    
    residual_dict = {item:r for (item,r) in zip(data['items'], residual)}
    return residual_dict
    



def save_recipe_matrix(path: str) -> None:
    data = build_recipe_matrix()

    items = data["items"]
    recipes = data["recipes"]
    matrix = data["matrix"]

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as file:
        writer = csv.writer(file)

        # First cell labels the row names; remaining cells label columns.
        writer.writerow(["item", *recipes])

        for item, row in zip(items, matrix):
            writer.writerow([item, *row])

if __name__ == "__main__":
    save_recipe_matrix("outputs/recipe_matrix.csv")

    


