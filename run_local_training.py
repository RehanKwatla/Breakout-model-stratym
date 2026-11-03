from __future__ import annotations
import contextlib
import io
import os
import sys
import time
import traceback
from pathlib import Path
import nbformat
from IPython.core.interactiveshell import InteractiveShell

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "04_breakout_policy_local.ipynb"
EXECUTED = ROOT / "04_breakout_policy_local_executed.ipynb"
LOG = ROOT / "training_execution.log"
os.chdir(ROOT)
os.environ.setdefault("MPLBACKEND", "Agg")
nb = nbformat.read(NOTEBOOK, as_version=4)
shell = InteractiveShell.instance()
shell.user_ns["__name__"] = "__main__"

class Tee(io.TextIOBase):
    def __init__(self, *targets): self.targets = targets
    def write(self, text):
        for target in self.targets:
            target.write(text)
            target.flush()
        return len(text)
    def flush(self):
        for target in self.targets: target.flush()

with LOG.open("w", encoding="utf-8", buffering=1) as log:
    start = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"Notebook: {NOTEBOOK.name}\nStarted: {start}\n", flush=True)
    print(f"Notebook: {NOTEBOOK.name}\nStarted: {start}\n", file=log, flush=True)
    for index, cell in enumerate(nb.cells):
        if cell.cell_type != "code": continue
        cell.execution_count = index + 1
        out_buf, err_buf = io.StringIO(), io.StringIO()
        print(f"\n===== CELL {index} =====", flush=True)
        print(f"\n===== CELL {index} =====", file=log, flush=True)
        try:
            with contextlib.redirect_stdout(Tee(sys.__stdout__, log, out_buf)), contextlib.redirect_stderr(Tee(sys.__stderr__, log, err_buf)):
                result = shell.run_cell(cell.source, store_history=False, silent=False)
            out_text, err_text = out_buf.getvalue(), err_buf.getvalue()
            combined = out_text + ("\n[stderr]\n" + err_text if err_text else "")
            cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text=combined)] if combined else []
            if result.error_before_exec is not None or result.error_in_exec is not None:
                error = result.error_before_exec or result.error_in_exec
                tb = "".join(traceback.format_exception(type(error), error, error.__traceback__))
                cell.outputs.append(nbformat.v4.new_output("error", ename=type(error).__name__, evalue=str(error), traceback=tb.splitlines()))
                nbformat.write(nb, EXECUTED)
                raise error
        except BaseException as exc:
            print(f"\nFAILED in cell {index}: {exc!r}", file=log, flush=True)
            nbformat.write(nb, EXECUTED)
            raise
        nbformat.write(nb, EXECUTED)
    end = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"\nFinished: {end}", flush=True)
    print(f"\nFinished: {end}", file=log, flush=True)
print(f"Executed notebook saved to {EXECUTED}")
print(f"Execution log saved to {LOG}")
