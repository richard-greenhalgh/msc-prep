# test_solver.py

from factorio_solver.solver import solve
from factorio_solver.matrix import verify_balance

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

def test_logistic_science_raw_rates():
    result = solve("logistic_science_pack", 60)

    assert result.raw_rates["iron_ore"] == 330
    assert result.raw_rates["copper_ore"] == 90


def test_logistic_science_recipe_rates():
    result = solve("logistic_science_pack", 60)

    assert result.recipe_rates["transport_belt"] == 30
    assert result.recipe_rates["copper_cable"] == 90
    assert result.recipe_rates["iron_gear_wheel"] == 90

def test_copper_cable_output_quantity():
    result = solve("electronic_circuit", 45)

    assert result.item_rates["copper_cable"] == 135
    assert result.recipe_rates["copper_cable"] == 67.5

def test_matrix_balance_for_electronic_circuits():
    result = solve("electronic_circuit", 45)
    residual = verify_balance(result, "electronic_circuit", 45)

    assert all(abs(value) < 1e-9 for value in residual.values())