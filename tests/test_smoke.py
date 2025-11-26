"""Check that basic features work.

Catch cases where e.g. files are missing so the import doesn't work."""

from hopwise.quick_start import run_hopwise

run_hopwise(model="BPR", dataset="ml-100k", config_dict={"epochs": 1}, saved=False)
