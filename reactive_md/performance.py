"""Opt-in, nested, process-local wall-clock diagnostics."""
from contextlib import contextmanager
from contextvars import ContextVar
from itertools import count
from uuid import uuid4
import json
import os
from time import perf_counter

import jax


_active = ContextVar("performance_scope", default=None)
_ids = count()
_run_id = uuid4().hex


class PerformanceTimer:
    def __init__(self):
        self.enabled = os.environ.get('REACTIVE_MD_PERFORMANCE') == '1'

    @staticmethod
    def _noop(*values):
        pass

    @staticmethod
    def _wait(*values):
        # Wait on array leaves, without transferring them to the host.
        for leaf in jax.tree_util.tree_leaves(values):
            wait = getattr(leaf, 'block_until_ready', None)
            if callable(wait):
                wait()

    @contextmanager
    def measure(self, stage, step=None, **fields):
        if not self.enabled:
            yield self._noop
            return
        parent = _active.get()
        if step is None and parent is not None:
            step = parent[1]
        span = next(_ids)
        token = _active.set((span, step))
        start = perf_counter()
        status = 'ok'
        try:
            yield self._wait
        except BaseException:
            status = 'error'
            raise
        finally:
            elapsed = perf_counter() - start
            _active.reset(token)
            record = dict(stage=stage, step=None if step is None else int(step),
                          elapsed_s=elapsed, status=status, run_id=_run_id,
                          span_id=span, parent_id=None if parent is None else parent[0],
                          **fields)
            if status == 'ok' and stage in ('md_chunk', 'langevin_chunk') and elapsed > 0:
                record['steps_per_s'] = fields['chunk_steps'] / elapsed
            print('[REACTIVE_MD PERF] ' + json.dumps(record, sort_keys=True),
                  flush=True)
