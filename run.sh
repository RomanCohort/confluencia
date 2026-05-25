#!/bin/bash
# Confluencia Launcher
# Choose between Drug Module or circRNA Module

echo "=========================================="
echo "  Confluencia Platform Launcher"
echo "=========================================="
echo ""
echo "Select Module:"
echo "  1. Drug Module (Small Molecules)"
echo "  2. circRNA Module (RNA Vaccines)"
echo "  3. Exit"
echo ""

read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo ""
        echo "Starting Drug Module..."
        echo "Location: confluencia-2.0-drug/app_drug.py"
        cd confluencia-2.0-drug
        streamlit run app_drug.py
        ;;
    2)
        echo ""
        echo "Starting circRNA Module..."
        echo "Location: confluencia_circrna/app.py"
        cd confluencia_circrna
        streamlit run app.py
        ;;
    3)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Please enter 1, 2, or 3."
        exit 1
        ;;
esac