# entry point for CLI

import argparse
import sys

from factorio_solver.solver import solve, print_solver_result
from factorio_solver.matrix import verify_balance

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Solve Factorio production requirements for a target item."
    )

    parser.add_argument("item")
    parser.add_argument("rate", type=float)

    parser.add_argument(
        "--machine-policy",
        default="EARLY_GAME",
        choices=["EARLY_GAME", "MID_GAME", "LATE_GAME"],
        help="Which default machines to use for machine counts.",
    )

    parser.add_argument(
        "--graph",
        metavar="PATH",
        help="Write a production graph to this path, e.g. outputs/green_science",
    )

    args = parser.parse_args()

    try:
        result = solve(
            args.item,
            args.rate,
            machine_policy=args.machine_policy,
        )

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # output results
    print_solver_result(result)
    verify_balance(result, args.item, args.rate)

    # output dependency graph?
    if args.graph is not None:
        from factorio_solver.graph import draw_production_graph

        dot = draw_production_graph(result, args.item)
        output_path = dot.render(args.graph, cleanup=True)

        print(f"\nGraph written to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())