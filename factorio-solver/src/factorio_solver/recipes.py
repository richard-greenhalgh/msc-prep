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
    "stone",
    "crude_oil",
    "water"
}

# Hard-coded recipes for MVP
RecipeDB = {}

# short-hand for common
COAL = "coal"
STONE = "stone"
IRON = "iron_plate"
GEAR = "iron_gear_wheel"
COPPER = "copper_plate"
CABLE = "copper_cable"
STEEL = "steel_plate"
PLASTIC = "plastic_bar"
SULFUR = "sulfur"
PIPE = "pipe"
BRICK = "stone_brick"
LDS = "low_density_structure"

WATER = "water"
CRUDE = "crude_oil"
PETROL = "petroleum_gas"
ACID = "sulfuric_acid"

GREEN = "electronic_circuit"
RED = "advanced_circuit"
BLUE = "processing_unit"

# recipes with single output, where name of the recipe is the output
def register_recipe(name:str, cat:str, inputs:dict, nout:int, time:float):
    assert cat in {"crafting", "smelting", "oil", "chemical"}
    RecipeDB[name] = Recipe(
        name=name,
        category=cat,
        inputs=inputs,
        outputs={name:nout},
        time=time
    )

# handles recipes with multiple outputs, e.g. advanced oil processing
def register_complex_recipe(name:str, cat:str, inputs:dict, outputs:dict, time:float):
    assert cat in {"crafting", "smelting", "oil", "chemical"}
    RecipeDB[name] = Recipe(
        name=name,
        category=cat,
        inputs=inputs,
        outputs=outputs,
        time=time
    )

register_recipe("iron_plate", "smelting", {"iron_ore": 1}, 1, 3.2)
register_recipe("copper_plate", "smelting", {"copper_ore": 1}, 1, 3.2)
register_recipe("steel_plate", "smelting", {IRON:5}, 1, 16.0)
register_recipe("copper_cable", "crafting", {COPPER: 1}, 2, 0.5)
register_recipe("electronic_circuit", "crafting", {IRON: 1, CABLE: 3}, 1, 0.5)
register_recipe("iron_gear_wheel", "crafting", {IRON: 2}, 1, 0.5)
register_recipe("automation_science_pack", "crafting", {COPPER: 1, GEAR: 1}, 1, 5.0)
register_recipe("transport_belt", "crafting", {GEAR: 1, IRON: 1}, 2, 0.5)
register_recipe("inserter", "crafting", {GREEN: 1, GEAR: 1, IRON: 1}, 1, 0.5)
register_recipe("logistic_science_pack", "crafting", {"inserter": 1, "transport_belt": 1}, 1, 6.0)

register_recipe("grenade", "crafting", {"coal":10, IRON:5}, 1, 8.0)
register_recipe("firearm_magazine", "crafting", {IRON:4}, 1, 1.0)
register_recipe("piercing_rounds_magazine", "crafting", {COPPER:2, "firearm_magazine":2, STEEL:1}, 2, 6.0)
register_recipe("stone_brick", "smelting", {"stone":2}, 1, 3.2)
register_recipe("wall", "crafting", {"stone_brick":5}, 1, 0.5)
register_recipe("military_science_pack", "crafting", {"grenade":1, "piercing_rounds_magazine":1, "wall":2}, 1, 10.0)

register_recipe("pipe", "crafting", {IRON:1}, 1, 0.5)
register_recipe("engine_unit", "crafting", {GEAR:1, PIPE:2, STEEL:1}, 1, 10.0)
register_recipe("chemical_science_pack", "crafting", {RED: 3, "engine_unit":2, SULFUR:1}, 1, 24.0)


register_recipe("petroleum_gas", "oil", {CRUDE:100, WATER:50}, 55, 5)
register_recipe("plastic_bar", "chemical", {COAL:1, PETROL:20}, 2, 1)
register_recipe("advanced_circuit", "crafting", {CABLE:4, GREEN:2, PLASTIC:2}, 1, 6)

register_recipe("sulfur", "chemical", {PETROL:30, WATER:30}, 2, 1)
register_recipe("sulfuric_acid", "chemical", {IRON:1, SULFUR:5, WATER:100}, 50, 1)
register_recipe("processing_unit", "crafting", {RED:2, GREEN:20, ACID:5}, 1, 10)

register_recipe("electric_furnace", "crafting", {RED:5, STEEL:10, BRICK:10}, 1, 5.0)
register_recipe("productivity_module", "crafting", {RED:5, GREEN:5}, 1, 15.0)
register_recipe("iron_stick", "crafting", {IRON:1}, 2, 0.5)
register_recipe("rail", "crafting", {"iron_stick":1, STEEL:1, STONE:1}, 2, 0.5)
register_recipe("production_science_pack", "crafting", {"electric_furnace":1, "productivity_module":1, "rail":30}, 3, 21.0)

register_recipe("battery", "crafting", {COPPER:1, IRON:1, ACID:20}, 1, 4.0)
register_recipe("heavy_oil", "oil", {CRUDE:100, WATER:50}, 25, 5.0)
register_recipe("lubricant", "chemical", {"heavy_oil":10}, 10, 1.0)
register_recipe("electric_engine_unit", "crafting", {GREEN:2, "engine_unit":1, "lubricant":15}, 1, 10.0)
register_recipe("flying_robot_frame", "crafting", {"battery":2, "electric_engine_unit":1, GREEN:3, STEEL:1}, 1, 20.0)
register_recipe("low_density_structure", "crafting", {COPPER:20, PLASTIC:5, STEEL:2}, 1, 15.0)
register_recipe("utility_science_pack", "crafting", {"flying_robot_frame":1, LDS:3, BLUE:2}, 3, 21.0)

for prefix in ["speed_", "productivity_", "efficiency_"]:
    register_recipe(prefix+"module", "crafting", {RED:5, GREEN:5}, 1, 15.0)
    register_recipe(prefix+"module_2", "crafting", {RED:5, BLUE:5, prefix+"module":4}, 1, 30.0)
    register_recipe(prefix+"module_3", "crafting", {RED:5, BLUE:5, prefix+"module_2":4}, 1, 60.0)



