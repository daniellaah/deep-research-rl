import json
import subprocess
import sys


def test_core_import_does_not_load_training_frameworks() -> None:
    code = """
import json
import sys
import deep_research_rl.core
print(json.dumps(sorted(name for name in sys.modules if name.split('.')[0] in {
    'torch', 'transformers', 'verl', 'agent_r1'
})))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == []
