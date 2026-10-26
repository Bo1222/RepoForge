"""Verify native filesystem and tool-process isolation without reading secrets."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import importlib.util
import urllib.request

from repoforge.evaluation.experiment import read_json, write_json
from repoforge.runtime.native_tools import tool_process


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    access = {}
    for name in config["blocked_paths"]:
        if Path(name).is_dir():
            try:
                list(Path(name).iterdir())
            except OSError:
                access[name] = False
            else:
                access[name] = True
            continue
        try:
            with open(name, "rb"):
                pass  # Open/close only; never read credential or hidden bytes.
        except OSError:
            access[name] = False
        else:
            access[name] = True
    public = {}
    for name in config["public_paths"]:
        try:
            with open(name, "rb"):
                pass
        except OSError:
            public[name] = False
        else:
            public[name] = True
    code = """
import json,os,socket
from pathlib import Path
visible_parent_credential=False
for path in Path('/proc').glob('[0-9]*/environ'):
    try:
        visible_parent_credential |= b'DEEPSEEK_API_KEY=native-offline-fixture' in path.read_bytes()
    except OSError:
        pass
sock=socket.socket()
sock.settimeout(1)
connection=sock.connect_ex(('127.0.0.1',8787))
sock.close()
print(json.dumps({'net_namespace':os.readlink('/proc/self/ns/net'),
 'pid_namespace':os.readlink('/proc/self/ns/pid'),
 'model_keys_present':any(os.environ.get(n) for n in ('JINA_API_KEY','DEEPSEEK_API_KEY','OPENAI_API_KEY')),
 'parent_credential_visible_in_proc':visible_parent_credential,
 'host_proxy_connect_code':connection}))
"""
    argv, env = tool_process([sys.executable, "-c", code])
    process = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=20)
    child = json.loads(process.stdout) if process.returncode == 0 else {}
    parent_net = os.readlink("/proc/self/ns/net")
    parent_pid = os.readlink("/proc/self/ns/pid")
    service = config.get('embedding_service')
    service_check = None
    if service:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(service['url'], timeout=5) as response:
            health = json.load(response)
        service_check = {'health': health,
                         'heavy_packages': {name: importlib.util.find_spec(name) is not None
                                            for name in ('torch', 'sentence_transformers')},
                         'gpu_devices': [str(p) for p in Path('/dev').glob('nvidia*')],
                         'cloud_embedding_keys_present': any(os.environ.get(name) for name in
                             ('JINA_API_KEY', 'OPENAI_API_KEY', 'VOYAGE_API_KEY'))}
        service_check['passed'] = (health.get('status') == 'ok' and health.get('model') == service['model']
                                   and health.get('revision') == service['revision']
                                   and not any(service_check['heavy_packages'].values())
                                   and not service_check['gpu_devices']
                                   and not service_check['cloud_embedding_keys_present'])
    passed = (all(not available for available in access.values()) and all(public.values())
              and process.returncode == 0 and child.get("net_namespace") != parent_net
              and child.get("pid_namespace") != parent_pid and not child.get("model_keys_present", True)
              and not child.get("parent_credential_visible_in_proc", True)
              and child.get("host_proxy_connect_code") != 0
              and not os.environ.get("HOST_ONLY_SECRET")
              and (service_check is None or service_check['passed']))
    write_json(args.output / "result.json", {"manifest_hash": config["manifest_hash"],
        "passed": passed, "blocked_paths_readable": access, "public_paths_readable": public,
        "parent_net_namespace": parent_net, "parent_pid_namespace": parent_pid,
        "tool_result": child, "tool_return_code": process.returncode,
        "tool_stderr": process.stderr, "embedding_service": service_check,
        "provider_calls": 0, "official_harness": "not_run"})
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
