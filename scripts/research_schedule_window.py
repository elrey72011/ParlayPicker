"""Print the Actions gate without importing application dependencies."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app_core.research_schedule import is_open
if __name__=="__main__":print("true" if is_open() else "false")
