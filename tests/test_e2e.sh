#!/bin/bash
echo "Testing end-to-end optimization workflow..."
python3 ./scripts/config.py && \
python3 ./scripts/demo.py && \
echo "Optimization workflow completed."
