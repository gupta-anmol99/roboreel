#!/bin/bash
set -e

# Activate tasks env by default (matches what .bashrc does for interactive shells)
source /opt/conda/etc/profile.d/conda.sh
conda activate tasks

exec "$@"
