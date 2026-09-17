# Vendored dependency: torchreid (from bpbreid)

Vendored from `/home/lakshh/workspace/reid/bpbreid` (Somers, V. et al., "BPBreID: Body Part-based
Representation Learning for Occluded Person Re-Identification", WACV 2023), commit-state as of
2026-08-22, for local/internal modification and debugging without a separate `pip install -e
/path/to/bpbreid` step.

**License note:** `pip show torchreid` on the original editable install reports `License: MIT`,
but the actual `LICENSE` file shipped in the bpbreid repository (copied alongside this notice) is
the **Hippocratic License 3.0**, not MIT -- an ethical-use license with additional restrictions
beyond a standard permissive license. The metadata string is not authoritative; the LICENSE file
is. Consult it (and the upstream repository) before any use beyond internal debugging.

Includes a prebuilt Cython extension (`metrics/rank_cylib/rank_cy.*.so`) compiled for
`cpython-312-x86_64-linux-gnu`. If this vendored copy is ever run under a different Python
version or platform, that extension needs rebuilding from `rank_cy.pyx` (via `cythonize` + a C
compiler, see bpbreid's own `setup.py` for the exact build invocation) -- it will not load as-is
on a mismatched interpreter/ABI.

Upstream: https://github.com/VlSomers/bpbreid
