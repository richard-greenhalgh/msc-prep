# graph.py
# generate dependency graph(s)
from dataclasses import dataclass
from math import ceil
from graphviz import Digraph

from factorio_solver.recipes import RAW_ITEMS, LIQUIDS, RecipeDB
from factorio_solver.solver import SolveResult, solve

@dataclass(frozen=True)
class GraphEdge:
    source_id: str
    target_id: str
    kind: str                 # "input" or "output"
    item_name: str
    recipe_name: str

def item_id(item: str) -> str:
    return f"item__{item}"

def recipe_id(recipe: str) -> str:
    return f"recipe__{recipe}"

def trace_recipe_dependencies(target_item: str):
    item_nodes: set[str] = set()
    recipe_nodes: set[str] = set()
    edges: set[GraphEdge] = set()
    visited_items: set[str] = set()

    def visit_item(item: str) -> None:
        item_nodes.add(item)

        if item in visited_items:
            return
        visited_items.add(item)

        if item in RAW_ITEMS:
            return

        if item not in RecipeDB:
            raise ValueError(f"Unknown item: {item}")

        recipe = RecipeDB[item]
        recipe_nodes.add(recipe.name)

        # Ingredients flow into the recipe.
        for ingredient in recipe.inputs:
            item_nodes.add(ingredient)
            #edges.add((item_id(ingredient), recipe_id(recipe.name), "input"))
            edges.add(
                GraphEdge(
                    source_id=item_id(ingredient),
                    target_id=recipe_id(recipe.name),
                    kind="input",
                    item_name=ingredient,
                    recipe_name=recipe.name,
                )
            )
            visit_item(ingredient)

        # Recipe produces its output item.
        #edges.add((recipe_id(recipe.name), item_id(item), "output"))
        edges.add(
            GraphEdge(
                source_id=recipe_id(recipe.name),
                target_id=item_id(item),
                kind="output",
                item_name=item,
                recipe_name=recipe.name,
            )
        )

    visit_item(target_item)

    return item_nodes, recipe_nodes, edges

def draw_production_graph(result:SolveResult, target_item:str, label_outputs=False):
    dot = Digraph(
        format="svg",
        graph_attr={"rankdir": "LR"},
    )

    item_nodes, recipe_nodes, edges = trace_recipe_dependencies(target_item)

    def recipe_label(recipe_name: str, result: SolveResult) -> str:
        req = result.machines_by_recipe[recipe_name]

        return (
            f"{{{{"
            f"{recipe_name.replace('_',' ').upper()}"
            f"|{req.machine_name} × {req.rounded_count}"
            f"|{result.recipe_rates[recipe_name]:.1f} crafts/min"
            f"|{result.item_rates[recipe_name]:.1f} output/min"
            f"}}}}"
        )

    def edge_label(edge: GraphEdge, result: SolveResult) -> str:
        recipe_name = edge.recipe_name
        recipe = RecipeDB[recipe_name]
        crafts_per_min = result.recipe_rates[recipe_name]

        if edge.kind == "output":
            if not label_outputs:
                return ""

            output_per_craft = recipe.outputs[edge.item_name]
            flow_per_min = crafts_per_min * output_per_craft
            return f"{flow_per_min:.0f}/min"

        input_per_craft = recipe.inputs[edge.item_name]
        flow_per_min = crafts_per_min * input_per_craft

        yellow_capacity = 15 * 60
        red_capacity = 30 * 60
        blue_capacity = 45 * 60

        if flow_per_min <= yellow_capacity:
            belt_suggestion = "yellow belt"
        elif flow_per_min <= red_capacity:
            belt_suggestion = "red belt"
        elif flow_per_min <= blue_capacity:
            belt_suggestion = "blue belt"
        else:
            blue_belts = ceil(flow_per_min / blue_capacity)
            belt_suggestion = f"{blue_belts} blue belts"

        if edge.item_name in LIQUIDS:
            return f"{flow_per_min:.0f}/min\n{flow_per_min / 60:.1f}/sec"

        return f"{flow_per_min:.0f}/min\nfits on {belt_suggestion}"

    for item in sorted(item_nodes):
        if item in RAW_ITEMS:
            shape = "box"
        elif item in LIQUIDS:
            shape = "diamond"
        else:
            shape = "ellipse"
        dot.node(
            item_id(item),
            label=item,
            shape=shape,
        )

    for recipe_name in sorted(recipe_nodes):
        dot.node(
            recipe_id(recipe_name),
            label=recipe_label(recipe_name, result),
            shape="record",
        )

    for edge in sorted(edges, key=lambda e: (e.source_id, e.target_id, e.kind)):
        dot.edge(
            edge.source_id,
            edge.target_id,
            label=edge_label(edge, result),
            fontsize="10",
        )

    return dot

if __name__ == "__main__":
    result = solve("advanced_circuit", 15*60, machine_policy="LATE_GAME")
    dot = draw_production_graph(result, "advanced_circuit", label_outputs=True)
    dot.render("outputs/TEST_graph", cleanup=True)
