#!/bin/bash
# Setup a Python virtual environment and install required dependencies.
# Requires Python > 3.12

REQUIRED_PYTHON_VERSION="3.12"
INSTALLED_PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')

install_venv() {
    python -m venv .venv
}

install_packages() {
    pip install -r python_envs/requirements.txt
    pip install .
    cd src/envs/decentralized-envs && pip install .
    cd ../ic3net-envs && pip install .
}

if ! command -v python3 &> /dev/null; then
    echo "Error: Python is not installed"
    exit 1
fi
if [ "$INSTALLED_PYTHON_VERSION" != "$REQUIRED_PYTHON_VERSION" ]; then
    echo "Error: Python $REQUIRED_PYTHON_VERSION is required, but $INSTALLED_PYTHON_VERSION is installed."
    exit 1
fi

cd ..
if [ -d ".venv" ]; then
    read -p "Virtual environment already exists. Do you want to re-create it? [y/n] " answer
    if [[ $answer == [Yy]* ]]; then
        rm -rf .venv
        install_venv
    fi
else
    install_venv
fi

source .venv/bin/activate
install_packages
