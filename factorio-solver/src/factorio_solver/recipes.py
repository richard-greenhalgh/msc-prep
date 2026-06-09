# recipes.py
from dataclasses import dataclass

@dataclass
class Recipe:
    name: str
    category: str
    inputs: dict[str, float]
    outputs: dict[str, float]
    time: float

RAW_ITEMS = {
    "iron_ore",
    "copper_ore",
}

# Hard-coded recipes for MVP
RecipeDB = {}

# short-hand for common
IRON = "iron_plate"
GEAR = "iron_gear_wheel"
COPPER = "copper_plate"
CABLE = "copper_cable"
STEEL = "steel_plate"
GREEN = "electronic_circuit"
RED = "advanced_circuit"
BLUE = "processing_unit"

def register_recipe(name:str, cat:str, inputs:dict, nout:int, time:float):
    assert cat in {"crafting", "smelting"}
    RecipeDB[name] = Recipe(
        name=name,
        category=cat,
        inputs=inputs,
        outputs={name:nout},
        time=time
    )

register_recipe("iron_plate", "smelting", {"iron_ore": 1}, 1, 3.2)
register_recipe("copper_plate", "smelting", {"copper_ore": 1}, 1, 3.2)
register_recipe("copper_cable", "crafting", {COPPER: 1}, 2, 0.5)
register_recipe("electronic_circuit", "crafting", {IRON: 1, CABLE: 3}, 1, 0.5)
register_recipe("iron_gear_wheel", "crafting", {IRON: 2}, 1, 0.5)
register_recipe("automation_science_pack", "crafting", {COPPER: 1, GEAR: 1}, 1, 5.0)
register_recipe("transport_belt", "crafting", {GEAR: 1, IRON: 1}, 2, 0.5)
register_recipe("inserter", "crafting", {GREEN: 1, GEAR: 1, IRON: 1}, 1, 0.5)
register_recipe("logistic_science_pack", "crafting", {"inserter": 1, "transport_belt": 1}, 1, 6.0)
