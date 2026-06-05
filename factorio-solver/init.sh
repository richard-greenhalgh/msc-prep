#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="factorio-solver"
PACKAGE_NAME="factorio_solver"

create_file() {
    local path="$1"
    local content="$2"

    if [[ ! -e "$path" ]]; then
        printf "%s\n" "$content" > "$path"
    fi
}

# =============================================================================

mkdir -p "src/$PACKAGE_NAME"
mkdir -p tests

# =============================================================================

cat > pyproject.toml <<EOF
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "$PROJECT_NAME"
version = "0.1.0"
description = "A Python production-chain solver for Factorio"
requires-python = ">=3.11"
dependencies = []

[project.optional-dependencies]
dev = ["pytest"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
EOF

# =============================================================================

for f in __init__.py recipes.py solver.py machines.py; do
    create_file "src/$PACKAGE_NAME/$f" "# $f"
done

create_file "tests/test_solver.py" "# test_solver.py"

# =============================================================================

cat <<'EOF'

Setup complete.

Next commands to run:

  python -m venv .venv
  source .venv/bin/activate
  pip install -e ".[dev]"
  pytest

Later, when returning to the project:

  source .venv/bin/activate
  pytest

EOF

