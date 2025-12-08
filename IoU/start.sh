#!/bin/bash

# --- Config ---
CONDA_ENV="iou"
PROJECT_ROOT="/home/ubuntu/chen/IoU"
BACKEND_DIR="$PROJECT_ROOT/backend"
HOST="0.0.0.0"
PORT="8001"

echo "============================================"
echo "Starting IoU OCR Service"
echo "============================================"

# 1. Check Conda Environment
echo "[*] Checking Conda Environment..."
# Source conda.sh to enable 'conda' command if not already in PATH
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

if conda info --envs | grep -q "$CONDA_ENV"; then
    echo "    -> Environment '$CONDA_ENV' found."
else
    echo "    -> Error: Environment '$CONDA_ENV' not found!"
    exit 1
fi

# 2. Activate Environment
echo "[*] Activating Environment..."
conda activate "$CONDA_ENV"

# 3. Check Dependencies
echo "[*] Checking Dependencies..."
# Simple check for python packages
if python -c "import fastapi, uvicorn, pdf2image" &> /dev/null; then
    echo "    -> Core Python packages found."
else
    echo "    -> Installing missing packages..."
    pip install fastapi uvicorn python-multipart jinja2 aiofiles docx2pdf pdf2image opencv-python-headless pillow shapely modelscope
fi

# 4. Check System Dependencies (LibreOffice for Word)
# This is optional but recommended for Word support
if command -v libreoffice &> /dev/null; then
    echo "    -> LibreOffice found (Word support enabled)."
else
    echo "    -> Warning: LibreOffice not found. Word (.doc/.docx) files may fail to convert."
    # Optionally try to install or just warn
fi

# 5. Start Service
echo "[*] Starting FastAPI Server..."
cd "$BACKEND_DIR" || exit 1
echo "    -> Listening on http://$HOST:$PORT"

# Use exec to replace shell with python process
# Fix: reload causes watch file limit error on some systems, disable reload for production stability
exec uvicorn main:app --host "$HOST" --port "$PORT"
