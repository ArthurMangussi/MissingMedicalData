"""
Vendored copy of MAT (Mask-Aware Transformer for Large Hole Image Inpainting,
Li et al., CVPR 2022 - https://github.com/fenglinglwb/MAT).

The original repo assumes its own root is on sys.path (it imports internal
modules as top-level packages, e.g. `import dnnlib`, `from networks.mat import
Generator`). Rather than rewriting every internal import, this package's
directory is added to sys.path on first import so the vendored code runs
unmodified. Import `algorithms.mat` before importing any of its submodules
by absolute name (e.g. `import dnnlib`, `from networks.mat import Generator`).

License: NVIDIA Source Code License-NC (research/non-commercial use only,
see algorithms/mat/LICENSE).
"""

import os
import sys

_MAT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _MAT_ROOT not in sys.path:
    sys.path.insert(0, _MAT_ROOT)
