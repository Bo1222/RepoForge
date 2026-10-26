"""Non-executing candidate compilation in the explicitly selected interpreter."""

import json

SCRIPT = """import json,sys
source=sys.stdin.buffer.read().decode('utf-8')
result={'status':'valid','interpreter':{'executable':sys.executable,'version':sys.version}}
try:
    compile(source,sys.argv[1],'exec',dont_inherit=True)
except (SyntaxError,ValueError) as exc:
    result.update(status='invalid',message=str(exc),line=getattr(exc,'lineno',None),column=getattr(exc,'offset',None))
print('REPOFORGE_SYNTAX='+json.dumps(result))
"""


def arguments(path):
    return ["-c", SCRIPT, path]


def parse_output(stdout, code):
    for line in reversed(stdout.splitlines()):
        if line.startswith("REPOFORGE_SYNTAX=") and code == 0:
            try:
                result = json.loads(line.split("=", 1)[1])
                if result.get("status") in {"valid", "invalid"} and result.get("interpreter"):
                    return result
            except (ValueError, TypeError):
                pass
    return {"status": "unavailable", "reason": "target_compiler_failed", "return_code": code}
