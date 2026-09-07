"""Cross-family test of the receipt-transport classification (Corollary 1).

Each adapter in :mod:`recurrent_families.adapters` exposes one released
recurrent language model through the same small interface: prefill a context,
advance one token while recording the per-layer transition and write terms,
snapshot and restore the recurrent state, and apply a recorded transition to an
arbitrary state. :mod:`recurrent_families.audit` runs the same record-present /
record-omitted protocol over every adapter and reports, per family, whether the
static, diagonal-ledger, and full-matrix receipt classes transport exactly
under frozen inputs, how large the native forcing is, and how much of the
record a teacher-forced probe can still recover from each corrected state.
"""
