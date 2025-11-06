#!/bin/bash

# ParlayDesk Enhanced - Setup Script
# This script installs all required dependencies

echo "🎯 ParlayDesk Enhanced - Setup Script"
echo "======================================"
echo ""

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Found Python $python_version"
echo ""

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 not found. Please install pip first."
    exit 1
fi

echo "📦 Installing required packages..."
echo ""

# Install core packages
echo "Installing streamlit..."
pip3 install streamlit --quiet

echo "Installing pandas..."
pip3 install pandas --quiet

echo "Installing numpy..."
pip3 install numpy --quiet

echo "Installing requests..."
pip3 install requests --quiet

echo "Installing pytz..."
pip3 install pytz --quiet

echo ""
echo "🧠 Installing ML packages..."
echo ""

echo "Installing scikit-learn..."
pip3 install scikit-learn --quiet

echo ""
echo "✅ Installation complete!"
echo ""
echo "📊 Verifying installations..."
python3 << EOF
import sys
packages = {
    'streamlit': 'streamlit',
    'pandas': 'pandas',
    'numpy': 'numpy',
    'requests': 'requests',
    'pytz': 'pytz',
    'scikit-learn': 'sklearn'
}

all_good = True
for name, import_name in packages.items():
    try:
        __import__(import_name)
        print(f"   ✅ {name}")
    except ImportError:
        print(f"   ❌ {name} - FAILED")
        all_good = False

if all_good:
    print("\n🎉 All packages installed successfully!")
else:
    print("\n⚠️  Some packages failed to install. Please install manually.")
    sys.exit(1)
EOF

echo ""
echo "🚀 Ready to run!"
echo ""
echo "To start the application:"
echo "   streamlit run streamlit_app_enhanced.py"
echo ""
echo "📚 For help, read:"
echo "   - README_ENHANCED.md (full documentation)"
echo "   - COMPARISON.md (feature comparison)"
echo ""
echo "🔑 Don't forget to get your API key from:"
echo "   https://the-odds-api.com/account/"
echo ""
echo "   Make sure your subscription includes historical data access!"
echo ""
