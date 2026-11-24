"""``python -m phasor ...``: the command line (the desktop app runs exports this way)."""

import sys

from phasor.cli import main

sys.exit(main())
