#!/bin/bash
# Setup a Python virtual environment and install required dependencies.
# Requires Python > 3.12

MINIMUM_REQUIRED_PYTHON_VERSION="3.12"

install_venv() {
    python3 -m venv .venv
}

install_packages() {
    pip install --upgrade pip
    pip install -r python_envs/requirements.txt
    pip install -e .
    cd learning_envs/decentralised-envs && pip install .
    cd ../ic3net-envs && pip install .
}

if command -v python3.12 &> /dev/null; then
    INSTALLED_PYTHON_VERSION=312
elif command -v python3 &> /dev/null; then
    INSTALLED_PYTHON_VERSION=$(python3 -c 'import sys; print(sys.version_info[0]*100 + sys.version_info[1])')
    if [[ $INSTALLED_PYTHON_VERSION -lt $MINIMUM_REQUIRED_PYTHON_VERSION ]]; then
        echo "Error: At least Python 3.12 is required, but found Python $INSTALLED_PYTHON_VERSION."
        exit 1
    fi
else
    echo "Error: Python3 is not installed."
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
