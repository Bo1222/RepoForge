"""Keep generated test/bash processes offline inside the native allowlist."""

import os
import shutil


def tool_process(argv, environment=None):
    env = dict(environment or os.environ)
    if env.get("GENERATION_TOOL_ISOLATION") != "native-offline-v1":
        return argv, env
    executable = shutil.which("bwrap", path=env.get("PATH"))
    if not executable:
        raise RuntimeError("Native tool isolation requires Bubblewrap; no network fallback")
    for name in tuple(env):
        if name.endswith("_API_KEY") or name.upper() in {
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "OPENAI_API_BASE", "OPENAI_BASE_URL",
        }:
            del env[name]
    # '/' here is the already allowlisted generation namespace, not host root.
    # A fresh proc filesystem also hides the credential-bearing parent process.
    command = [executable, "--die-with-parent", "--unshare-user", "--unshare-pid",
               "--unshare-net", "--unshare-ipc", "--unshare-uts", "--bind", "/", "/",
               "--proc", "/proc", "--dev", "/dev", "--", *argv]
    return command, env
