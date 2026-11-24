# Archived tests

These files are retained for historical context, not as part of the active
regression suite. Default pytest discovery is restricted to `tests/` by
`pyproject.toml`. Do not include this directory in routine test runs.

| File or directory | Reason for archival |
| --- | --- |
| `test_topology_opls.py` | Superseded by `tests/test_lipf6.py`, which exercises the same cases using current imports. |
| `test_reaction.py` | Uses the obsolete `propose_reaction_trial` API. Current LiPF6 reaction tests exercise the replacement workflow. |
| `legacy_fixed_coordinate_mh/` | Historical production-kernel checks for the earlier fixed-coordinate A2 MH workflow. Not a validation of the current geometry-changing rs@md[MC] algorithm. |

Independent numerical and statistical reference tests remain active under
`tests/` and can be selected with `-m a2_reference`. They retain scientific
value without implying that the current production algorithm samples the
same equilibrium distribution.
