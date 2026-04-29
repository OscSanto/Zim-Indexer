#!/usr/bin/env python3
"""ZIM Indexer — standalone GUI for indexing and browsing ZIM files."""
import sys

if sys.version_info < (3, 10):
    print("Python 3.10+ required.")
    sys.exit(1)

from gui.app import launch

if __name__ == "__main__":
    launch()
