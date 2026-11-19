"""Multi-agent reinforcement learning (Issues #5–#6; architecture §5.6, ADR-3, ADR-6).

NumPy-only modules (importable without the ``marl`` extra): ``config``, ``observation``,
``reward``, ``global_state``, ``env``, ``vector_env``, ``replay``, ``scenarios``, ``evaluation``.
Modules that need PyTorch (``networks``, ``madql``, ``trainer``, ``checkpoint``, ``policy``) import
it themselves; install it with ``pip install -e ".[marl]" --extra-index-url
https://download.pytorch.org/whl/cpu`` (``03`` §3.4). This package never imports torch at import
time.
"""
