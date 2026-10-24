"""Launch from any MCP host working directory, using this project's configuration."""

import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)

from troubleshooter.mcp_server import main

if __name__ == "__main__":
    main()
