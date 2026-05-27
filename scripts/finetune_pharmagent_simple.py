import os
import runpy

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
runpy.run_path(os.path.join(base_path, 'scripts', 'finetune_pharmaPrompt_simple.py'), run_name='__main__')
