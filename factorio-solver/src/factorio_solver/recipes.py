# recipes.py
from dataclasses import dataclass

@dataclass
class Recipe:
    name: str
    category: str
    inputs: dict[str, float]
    outputs: dict[str, float]
    time: float


# Hard-coded recipes for MVP
RecipeDB = {
    "iron_plate": Recipe(
        name= "iron_plate",
        category= "smelting",
        inputs= {"iron_ore": 1},
        outputs= {"iron_plate": 1},
        time= 3.2
    ),
    "copper_plate": Recipe(
        name= "copper_plate",
        category= "smelting",
        inputs= {"copper_ore": 1},
        outputs= {"copper_plate": 1},
        time= 3.2
    ),
    "copper_cable": Recipe(
        name= "copper_cable",
        category= "crafting",
        inputs= {"copper_plate": 1},
        outputs= {"copper_cable": 2},
        time= 0.5
    ),
    "electronic_circuit": Recipe(
        name= "electronic_circuit",
        category= "crafting",
        inputs= {"iron_plate": 1, "copper_cable": 3},
        outputs= {"electronic_circuit": 1},
        time= 0.5
    )
}


