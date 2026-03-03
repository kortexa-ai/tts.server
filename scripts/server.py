#!/usr/bin/env python3

from pathlib import Path

project_root = Path(__file__).parent.parent

import sys

sys.path.insert(0, str(project_root / "src"))

from kortexa.tts.cli import main


if __name__ == "__main__":
    main()
