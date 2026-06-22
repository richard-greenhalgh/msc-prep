# solver.py
from dataclasses import dataclass
from math import ceil
from collections import defaultdict
from factorio_solver.recipes import RecipeDB, RAW_ITEMS
from factorio_solver.machines import default_machine_for

@dataclass
class MachineRequirement:
    recipe_name: str
    machine_name: str
    exact_count: float
    rounded_count: int
@dataclass
class SolveResult:
    item_rates: defaultdict[str, float]
    recipe_rates: defaultdict[str, float]
    raw_rates: defaultdict[str, float]
    machines_by_recipe: dict[str, MachineRequirement]
    machines_by_type: dict[str, int]

def solve(item: str, rate_per_min: float, machine_policy:str = "EARLY_GAME") -> SolveResult:
    result = SolveResult(item_rates=defaultdict(float), recipe_rates=defaultdict(float),
                         raw_rates=defaultdict(float), machines_by_recipe={}, machines_by_type={})
    get_item_requirements(item, rate_per_min, result)
    get_machine_requirements(result, machine_policy)
    return result

def get_item_requirements(item: str, rate: float, results: SolveResult):
    if item in RAW_ITEMS:
        results.raw_rates[item] += rate
        return results
    if item not in RecipeDB:
        raise ValueError(f"Unknown item: {item}")

    R = RecipeDB[item]

    results.item_rates[item] += rate
    results.recipe_rates[item] += rate / R.outputs[item]

    for ingredient, qty in R.inputs.items():
        # inputs required for a single item (some crafts produce more)
        adj_qty = qty / R.outputs[item]
        get_item_requirements(ingredient, rate * adj_qty, results)
    return results

def get_machine_requirements(result: SolveResult, machine_policy:str = "EARLY_GAME"):
    # machines by recipe
    for item, recipe_crafts_per_min in result.recipe_rates.items():
        recipe = RecipeDB[item]
        machine = default_machine_for(recipe, policy=machine_policy)
        machines_needed = recipe_crafts_per_min * recipe.time / 60 / machine.speed
        roundup = ceil(machines_needed)
        result.machines_by_recipe[item] = MachineRequirement(item, machine.name, machines_needed, roundup)
        result.machines_by_type[machine.name] = 0
    
    # machines by type
    for r, req in result.machines_by_recipe.items():
        result.machines_by_type[req.machine_name] += req.rounded_count


def print_solver_result(solver_result: SolveResult):
    _len = 25
    # Required inputs (excl raw)
    print("Items per minute:")
    for item, qty in solver_result.item_rates.items():
        if item in RecipeDB: print(f" {item:<25} : {qty:8.2f}    |    Yellow Belts : {qty/900.0:8.2f}")

    print("\nCrafts per minute:")
    for item, qty in solver_result.recipe_rates.items():
        if item in RecipeDB: print(f" {item:<25} : {qty:8.2f}")

    # Required inputs (raw only)
    print("\nRaw materials per minute:")
    for item, qty in solver_result.raw_rates.items():
        if item not in RecipeDB: print(f" {item:<25} : {qty:8.2f}")
    
    # Machines required
    print("\nMachines required (by recipe):")
    for r, req in solver_result.machines_by_recipe.items():
        print(f" {r:<25} : {req.machine_name:<25} : {req.rounded_count:4d}  ({req.exact_count:6.2f})")
    
    # Machines required
    print("\nMachines required (by machine type):")
    for r, req in solver_result.machines_by_type.items():
        print(f" {r:<25} : {req:4d}")


if __name__ == "__main__":
    #result = solve("electronic_circuit", 1000)
    result = solve("automation_science_pack", 60)
    print_solver_result(result)
    #print(RecipeDB)