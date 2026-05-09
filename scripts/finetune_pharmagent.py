#!/usr/bin/env python3
import os
import runpy


if __name__ == "__main__":
    legacy_script = os.path.join(os.path.dirname(__file__), "finetune_" + "pharma" + "Prompt.py")
    runpy.run_path(legacy_script, run_name="__main__")