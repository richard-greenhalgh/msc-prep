# test_solver.py

from factorio_solver.solver import solve

def test_electronic_circuit_item_rates():
    result = solve("electronic_circuit", 45)

    assert result.item_rates["electronic_circuit"] == 45
    assert result.item_rates["iron_plate"] == 45
    assert result.item_rates["copper_cable"] == 135
    assert result.item_rates["copper_plate"] == 67.5


def test_electronic_circuit_recipe_rates():
    result = solve("electronic_circuit", 45)

    assert result.recipe_rates["electronic_circuit"] == 45
    assert result.recipe_rates["copper_cable"] == 67.5


def test_electronic_circuit_raw_rates():
    result = solve("electronic_circuit", 45)

    assert result.raw_rates["iron_ore"] == 45
    assert result.raw_rates["copper_ore"] == 67.5


def test_electronic_circuit_machine_counts():
    result = solve("electronic_circuit", 45)

    assert result.machines_by_recipe["electronic_circuit"].rounded_count == 1
    assert result.machines_by_recipe["copper_cable"].rounded_count == 2
    assert result.machines_by_type["assembling_machine_1"] == 3
    assert result.machines_by_type["stone_furnace"] == 7