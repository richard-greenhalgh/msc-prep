# machines.py
from dataclasses import dataclass
from factorio_solver.recipes import Recipe
@dataclass
class Machine:
    name: str               # name, e.g. "assembling_machine_1"
    categories: set[str]    # set of supported categories
    speed: float            # crafting speed

def default_machine_for(recipe: Recipe, policy="EARLY_GAME"):
    machine_name = DEFAULT_MACHINE_BY_CATEGORY[policy][recipe.category]
    return MachineDB[machine_name]

def get_machine(name):
    return MachineDB[name]

DEFAULT_MACHINE_BY_CATEGORY = {
    "EARLY_GAME": {
        "crafting": "assembling_machine_1",
        "smelting": "stone_furnace",
        "oil": "oil_refinery",
        "chemical": "chemical_plant",
    },
    "MID_GAME": {
        "crafting": "assembling_machine_2",
        "smelting": "steel_furnace",
        "oil": "oil_refinery",
        "chemical": "chemical_plant",
    },
    "LATE_GAME": {
        "crafting": "assembling_machine_3",
        "smelting": "electric_furnace",
        "oil": "oil_refinery",
        "chemical": "chemical_plant",
    },
}

MachineDB = {
    "assembling_machine_1": Machine(
        name="assembling_machine_1",
        categories={"crafting"},
        speed=0.5,
    ),
    "assembling_machine_2": Machine(
        name="assembling_machine_2",
        categories={"crafting"},
        speed=0.75,
    ),
    "assembling_machine_3": Machine(
        name="assembling_machine_3",
        categories={"crafting"},
        speed=1.25,
    ),
    "stone_furnace": Machine(
        name="stone_furnace",
        categories={"smelting"},
        speed=1.0,
    ),
    "steel_furnace": Machine(
        name="steel_furnace",
        categories={"smelting"},
        speed=2.0,
    ),
    "electric_furnace": Machine(
        name="electric_furnace",
        categories={"smelting"},
        speed=2.0,
    ),
    "oil_refinery": Machine(
        name="oil_refinery",
        categories={"oil"},
        speed=1.0,
    ),
    "chemical_plant": Machine(
        name="chemical_plant",
        categories={"chemical"},
        speed=1.0,
    ),
}
