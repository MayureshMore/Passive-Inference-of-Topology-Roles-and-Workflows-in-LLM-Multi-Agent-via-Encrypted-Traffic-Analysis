"""Pytest path setup so tests can import the project packages."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
