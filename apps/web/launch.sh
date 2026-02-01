#!/bin/bash
# Launch NexusMind Web Interface

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing..."
    pip install streamlit plotly
fi

# Enable academic accelerator if available
if [ -f /etc/network_turbo ]; then
    echo "🚀 Enabling academic accelerator..."
    source /etc/network_turbo
fi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Launch Streamlit
echo "🌐 Starting NexusMind Web Interface..."
echo "📱 URL: http://localhost:8501"
echo ""
streamlit run apps/web/app.py "$@"