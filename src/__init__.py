"""ShopHub Agent core package.

Ensures src directory is automatically available on sys.path so intra-package
imports (e.g. `from config import ...`) work seamlessly whether imported as
`import orchestrator` or `import src.orchestrator`.
"""

from __future__ import annotations

import sys
from pathlib import Path

_src_dir = str(Path(__file__).resolve().parent)
_root_dir = str(Path(__file__).resolve().parent.parent)
for _p in (_src_dir, _root_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)
