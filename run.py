r"""Launch the ResumeMatch AI Discord bot.

Usage (Windows PowerShell / cmd):
    .venv\Scriptsctivate
    python run.py

Usage (macOS / Linux):
    source .venv/bin/activate
    python run.py
"""

import sys

from app.main import main

if __name__ == "__main__":
    sys.exit(main())
