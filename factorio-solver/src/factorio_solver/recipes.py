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
    "coal",
    "crude_oil",
    "water"
}

# Hard-coded recipes for MVP
RecipeDB = {}

# short-hand for common
COAL = "coal"
IRON = "iron_plate"
GEAR = "iron_gear_wheel"
COPPER = "copper_plate"
CABLE = "copper_cable"
STEEL = "steel_plate"
PLASTIC = "plastic_bar"
SULFUR = "sulfur"

WATER = "water"
CRUDE = "crude_oil"
PETROL = "petroleum_gas"
ACID = "sulfuric_acid"

GREEN = "electronic_circuit"
RED = "advanced_circuit"
BLUE = "processing_unit"

def register_recipe(name:str, cat:str, inputs:dict, nout:int, time:float):
    assert cat in {"crafting", "smelting", "oil", "chemical"}
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

register_recipe("petroleum_gas", "oil", {CRUDE:100, WATER:50}, 55, 5)
register_recipe("plastic_bar", "chemical", {COAL:1, PETROL:20}, 2, 1)
register_recipe("advanced_circuit", "crafting", {CABLE:4, GREEN:2, PLASTIC:2}, 1, 6)

register_recipe("sulfur", "chemical", {PETROL:30, WATER:30}, 2, 1)
register_recipe("sulfuric_acid", "chemical", {IRON:1, SULFUR:5, WATER:100}, 50, 1)
register_recipe("processing_unit", "crafting", {RED:2, GREEN:20, ACID:5}, 1, 10)
