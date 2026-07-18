The frozen reference implementation IS the default in `adawave/`
(`geometry.py`, `patch.py`, `directional.py` are verbatim copies of the
loops used for every result in the paper). `adawave/fast_geometry.py` is
the vectorised alternative; `tests/test_fast_equivalence.py` certifies
end-to-end deviation < 1e-12 m before it may be enabled via
`fast_geometry.patch()`.
