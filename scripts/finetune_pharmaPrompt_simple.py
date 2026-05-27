import os
import runpy
import sys

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import src.model.simple_fusion as simple_fusion
import src.model.prompt_fusion_simple as prompt_fusion_simple

sys.modules['src.model.ban'] = simple_fusion
sys.modules['src.model.prompt_fusion'] = prompt_fusion_simple

runpy.run_path(os.path.join(base_path, 'scripts', 'finetune_pharmaPrompt.py'), run_name='__main__')
