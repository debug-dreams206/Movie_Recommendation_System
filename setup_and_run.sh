#!/usr/bin/env bash
# =============================================================================
# setup_and_run.sh
# One-shot script: creates venv, installs deps, and runs the project.
# Usage:  bash setup_and_run.sh
# =============================================================================
set -e   # exit immediately on any error

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   🎬  Movie Recommendation System — Setup & Run     ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── 1. Make sure we're in the project root ───────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "📂 Working directory: $SCRIPT_DIR"

# ── 2. Create virtual environment (if not already present) ───────────────────
if [ ! -d "venv" ]; then
    echo "🐍 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created."
else
    echo "✅ Virtual environment already exists."
fi

# ── 3. Activate venv ─────────────────────────────────────────────────────────
# Detect OS (Windows Git Bash uses Scripts/, Linux/Mac uses bin/)
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi
echo "✅ Virtual environment activated."

# ── 4. Install / upgrade dependencies ────────────────────────────────────────
echo "📦 Installing dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "✅ Dependencies installed."

# ── 5. Run the project ───────────────────────────────────────────────────────
echo ""
echo "🚀 Launching main.py ..."
echo ""
python main.py
