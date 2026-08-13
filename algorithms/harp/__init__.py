"""
Vendored subset of HARP (Histopathological Artifact Restoration Pipeline,
Fuchs et al., MIDL 2024 - https://github.com/MECLabTUDA/HARP, PyPI: HARPipe).

Only the diffusion restoration model is vendored here (network.py,
guided_diffusion_modules/, base_network.py - a "Palette"-style conditional
DDPM), not the full HARP pipeline (artifact detection via anomalib, SAM/DBSCAN
segmentation, mask ranking). Those stages exist upstream to *find* artifacts
and their masks automatically; this project already has its own missing-data
masks (see codes/data_amputation.py), so only the restoration step is useful
here. Skipping the rest also avoids anomalib/FrEIA/antlr4/stringzilla, which
pull in a C-extension build step that fails on Windows without MSVC build
tools.

See algorithms/harp/inference.py for the project-facing wrapper
(HARPInpainter), matching the same .transform(x_md, missing_mask) convention
as algorithms/mat/inference.py::MATInpainter.

License: CC BY 4.0 (attribution required), see algorithms/harp/LICENSE.txt.
"""
