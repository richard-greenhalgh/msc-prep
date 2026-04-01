#!/usr/bin/env bash
set -e

req=requirements.txt

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r $req

python -m ipykernel install --user --name=mnist --display-name "Python (mnist)"

echo "Setup complete, installed:"
cat $req
echo "Activate venv with: source .venv/bin/activate"

