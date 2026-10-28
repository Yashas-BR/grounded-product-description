"""groundedgen: Extract, Abstain, Generate (Team 10, RV University).

Modules are imported explicitly (``from groundedgen import taxonomy``) so that
CPU-only code never pulls in GPU dependencies.
"""

import sys

if sys.platform == "win32":
    # On Windows, loading pyarrow (via pandas) before torch makes later model
    # loading segfault (native DLL clash). Importing torch first avoids it.
    # See DECISIONS.md, M4 #1.
    try:
        import torch  # noqa: F401
    except ImportError:
        pass

__version__ = "0.1.0"
