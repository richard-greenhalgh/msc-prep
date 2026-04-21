#!/usr/bin/env bash
set -e

req=requirements.txt

[ -f "$req" ] || { echo "Missing $req"; exit 1; }

sudo apt update
sudo apt install -y python3-tk

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r "$req"

python -m ipykernel install --user --name=mnist --display-name "Python (mnist)"

echo "Setup complete, installed:"
cat $req
echo "Installed system package: python3-tk"
echo "Activate venv with: source .venv/bin/activate"

