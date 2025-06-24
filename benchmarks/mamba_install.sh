#!/bin/bash

# Script to create a Mamba environment named 'gene' with numpy and scipy
# Only creates the environment if it does not already exist

ENV_NAME="gene"

# Function to check if the environment exists
environment_exists() {
    # List all environments and check if ENV_NAME is among them
    mamba env list | awk '{print $1}' | grep -q "^${ENV_NAME}$"
}

# Check if the environment exists
if environment_exists; then
    echo "The environment '${ENV_NAME}' already exists."
else
    echo "Creating the '${ENV_NAME}' environment with numpy and scipy..."
    
    # Create the environment with numpy and scipy
    mamba create -n "${ENV_NAME}" numpy scipy matplotlib ipykernel scikit-learn -y

    # Regdiffusion and its dependencies
    pip install --no-deps regdiffusion --no-input
    mamba install pyvis anndata -y
    
    # Check if the environment was created successfully
    if [ $? -eq 0 ]; then
        echo "Environment '${ENV_NAME}' created successfully."
    else
        echo "Failed to create the environment '${ENV_NAME}'."
        exit 1
    fi
fi