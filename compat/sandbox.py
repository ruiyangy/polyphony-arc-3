"""
sandbox.py — execution-layer safety for bash_exec and workspace tools.

Two modes:
  - linux-strace (production/cloud): `strace -f -e trace=file,network` to
    enforce path/network restriction, plus process-group kill, output cap,
    timeout.
  - dev-degraded (macOS local): Python-layer path/env restriction + output
    cap + timeout + process-group kill. NOT a substitute for the Linux gate.

Implementation report MUST distinguish `dev-degraded sandbox` from
`linux-strace sandbox`. macOS cannot claim sandbox gate pass.
"""
from __future__ import annotations

import os
import platform
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_TIMEOUT = 40
OUTPUT_CAP_BYTES = 16000  # head+tail preserved beyond this

# Credential-bearing paths the agent must never read (enforced on Linux via
# strace). Threat-model-aligned: only genuine credential assets are listed.
# /etc/passwd is a world-readable uid->name map (no password hashes — those live
# in /etc/shadow), and glibc NSS reads it incidentally for legitimate uid->name
# resolution (`ls -la` / `id` / `getpass.getuser()`); at the syscall layer that
# is indistinguishable from a malicious `cat /etc/passwd`, so it is NOT denied
# (denying it would break ordinary tooling for no real secret). We keep the truly
# sensitive assets, narrow /etc/ssh to the host-key prefix (sshd_config is not
# sensitive), and omit /proc/sys (not a credential; writes are already blocked by
# the write policy).
#
# Two entry kinds (matcher is _is_denied_read_path, doing path-boundary matching
# so a prefix like `/etc/shadowed` does not falsely match `/etc/shadow`):
#   DENY_EXACT_OR_DIR — exact file or directory: matches path == d or path under d/;
#   DENY_FILE_PREFIX  — filename prefix (intentional): matches a basename in the
#                       same directory that starts with the prefix.
DENY_EXACT_OR_DIR = (
    "/etc/shadow",          # password hashes
    "/etc/sudoers",         # sudo rules (file)
    "/etc/sudoers.d",       # sudo rules (drop-in dir)
    "/root/.ssh",           # root SSH private-key dir (id_rsa / authorized_keys ...)
)
DENY_FILE_PREFIX = (
    "/etc/ssh/ssh_host_",   # ssh_host_*_key host private keys (prefix; sshd_config stays readable)
)
# Backward-compat flat tuple (some callers/tests reference READ_DENYLIST).
READ_DENYLIST = DENY_EXACT_OR_DIR + DENY_FILE_PREFIX


def _is_denied_read_path(pt: str) -> bool:
    """Path-aware denylist match. Avoids naive prefix false-hits like
    `/etc/shadowed_dir/x` matching `/etc/shadow`."""
    if not pt:
        return False
    for d in DENY_EXACT_OR_DIR:
        if pt == d or pt.startswith(d + "/"):
            return True
    for d in DENY_FILE_PREFIX:
        # only the basename component carries the prefix intent
        slash = d.rfind("/")
        ddir, dpref = d[:slash], d[slash + 1:]
        if pt.startswith(ddir + "/"):
            base = pt[len(ddir) + 1:]
            # the matched component must be a direct file under ddir
            if "/" not in base and base.startswith(dpref):
                return True
    return False


@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    truncated: bool
    mode: str
    duration_sec: float
    error: Optional[str] = None
    violation: Optional[str] = None  # set when a strace policy violation killed the cmd
    # Structured violation detail for diagnostics (rule/syscall/path_or_addr/
    # raw_strace_line). None unless a strace violation fired.
    violation_detail: Optional[dict] = None


def sandbox_mode() -> str:
    """Return the sandbox mode available on this host."""
    if platform.system() == "Linux" and shutil.which("strace"):
        return "linux-strace"
    return "dev-degraded"


def _truncate(text: str, cap: int = OUTPUT_CAP_BYTES) -> tuple[str, bool]:
    if len(text) <= cap:
        return text, False
    head = text[: cap // 2]
    tail = text[-cap // 2:]
    removed = len(text) - cap
    return f"{head}\n[truncated {removed} chars]\n{tail}", True


def run_supervised(command: str, exec_dir: str, workspace_root: str,
                   runtime_port: Optional[int] = None,
                   timeout: int = DEFAULT_TIMEOUT,
                   env_overrides: Optional[dict] = None,
                   run_root: Optional[str] = None) -> SandboxResult:
    """Run a shell command under the available sandbox.

    Always: process-group kill on timeout, output cap, restricted env.
    Linux+strace: also file/network syscall restriction (best-effort).

    `run_root` (the canonical run dir holding session/ recorder/ *.jsonl) is
    used to deny agent writes to canonical artifacts even when run_root is
    itself under /tmp.
    """
    mode = sandbox_mode()
    exec_dir = str(exec_dir)
    workspace_root = str(Path(workspace_root).resolve())
    run_root = str(Path(run_root).resolve()) if run_root else None

    # Restricted environment. We START from the parent environment and OVERRIDE
    # only the security-relevant keys, rather than handing Popen a fully minimal
    # dict. A fully-stripped env (env -i style) makes glibc NSS do an incidental
    # openat("/etc/passwd") at bash/python startup to resolve the uid→name —
    # indistinguishable from a malicious read and a false-positive for the
    # read-denylist enforcement. With an inherited+overridden env, the only
    # /etc/passwd opens that remain are genuine agent reads (verified: allowed
    # script → 0 incidental opens, leak probe → blocked).
    env = dict(os.environ)
    env.update({
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": exec_dir,
        "PWD": exec_dir,
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "PYTHONPATH": workspace_root,
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    if runtime_port:
        env["RUNTIME_API_BASE"] = f"http://127.0.0.1:{runtime_port}"
    if env_overrides:
        env.update(env_overrides)

    if mode == "linux-strace":
        return _run_linux_strace(command, exec_dir, workspace_root,
                                 runtime_port, timeout, env, run_root)

    # dev-degraded path (macOS / no strace): plain supervised subprocess.
    t0 = time.monotonic()
    try:
        proc = subprocess.Popen(
            command, shell=True, cwd=exec_dir, env=env,
            # stdin=DEVNULL so the child never inherits a parent PTY. A child
            # that inherits a PTY on fd0 probes openat("/dev/pts/N", O_RDWR) at
            # interpreter startup, which the strace path would flag as a write
            # outside workspace (FP-2). Detaching stdin makes bash_exec a pure
            # non-interactive single-command executor (no interactive TTY).
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,  # own process group for killpg
            text=True,
        )
    except Exception as e:
        return SandboxResult("", f"failed to launch: {e}", 127, False, False,
                             mode, time.monotonic() - t0, error=str(e))

    timed_out = False
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process_group(proc)
        try:
            out, err = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            out, err = "", ""
    dur = time.monotonic() - t0

    out_t, t1 = _truncate(out or "")
    err_t, t2 = _truncate(err or "")
    return SandboxResult(
        stdout=out_t, stderr=err_t,
        exit_code=proc.returncode if not timed_out else -signal.SIGKILL,
        timed_out=timed_out, truncated=t1 or t2, mode=mode, duration_sec=dur,
    )


# ── Linux strace enforcement (cloud production path) ────────────────────────
#
# Real enforcement: run the command under
# `strace -f -e trace=file,network`, stream strace's stderr line by line in a
# watcher thread, and HARD-KILL the process group the instant a forbidden
# syscall (write outside workspace/tmp, read of a denylisted path, or a
# non-localhost network connect) appears. This is NOT just an strace
# invocation that logs to a file — the watcher actively kills on violation.

def _run_linux_strace(command, exec_dir, workspace_root, runtime_port,
                      timeout, env, run_root=None) -> SandboxResult:
    mode = "linux-strace"
    # The child's real stderr would otherwise be merged into strace's own
    # stderr stream (the trace), making it unrecoverable (observability gap
    # (some exit=1 commands had no stderr on disk). Redirect the
    # child's fd2 to a dedicated /tmp capture file via a leading `exec 2>...`
    # in the script. /tmp is already in the write allowlist, so this does not
    # trip the write policy; the strace trace still flows on the real stderr
    # PIPE for the enforcement watcher.
    stderr_cap_path = None
    script_path = None
    try:
        fd, stderr_cap_path = tempfile.mkstemp(prefix=".v11_stderr_",
                                               suffix=".log", dir="/tmp")
        os.close(fd)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=".v11_bash_", suffix=".sh",
            delete=False, dir="/tmp",
        ) as sf:
            sf.write(f'exec 2>"{stderr_cap_path}"\n')
            sf.write(command)
            if not command.endswith("\n"):
                sf.write("\n")
            script_path = sf.name
    except Exception as e:
        return SandboxResult("", f"failed to prepare script: {e}", 127, False,
                             False, mode, 0.0, error=str(e))

    def _read_stderr_cap() -> str:
        if not stderr_cap_path:
            return ""
        try:
            with open(stderr_cap_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception:
            return ""

    strace_path = shutil.which("strace") or "strace"
    # -s 256 keeps quoted path args long enough to policy-check; -f follows children.
    cmd = [strace_path, "-f", "-e", "trace=file,network", "-s", "256",
           "bash", script_path]

    t0 = time.monotonic()
    try:
        proc = subprocess.Popen(
            cmd, cwd=exec_dir, env=env,
            # stdin=DEVNULL: do NOT inherit the parent PTY. This is the FP-2
            # root-cause fix — when the runner is launched from a tty/tmux/ssh -t,
            # an inherited PTY on fd0 makes python/bash probe
            # openat("/dev/pts/N", O_RDWR|O_NONBLOCK) at startup, which the
            # write-outside-workspace rule below would kill (every command, from
            # the first). bash_exec is a non-interactive single-command executor.
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True, text=True, bufsize=1,
        )
    except Exception as e:
        return SandboxResult("", f"failed to launch: {e}", 127, False, False,
                             mode, time.monotonic() - t0, error=str(e))

    stdout_chunks: list[str] = []
    violation: dict = {}

    def _read_stdout():
        if proc.stdout is None:
            return
        for line in proc.stdout:
            stdout_chunks.append(line)

    def _watch_strace_and_enforce():
        if proc.stderr is None:
            return
        for line in proc.stderr:
            if violation:
                continue
            v = _check_strace_violation(line, workspace_root, runtime_port, run_root)
            if v:
                violation.update(v)
                _kill_process_group(proc)

    t_out = threading.Thread(target=_read_stdout, daemon=True)
    t_err = threading.Thread(target=_watch_strace_and_enforce, daemon=True)
    t_out.start()
    t_err.start()

    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process_group(proc)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    finally:
        t_out.join(timeout=1.0)
        t_err.join(timeout=1.0)
        child_stderr = _read_stderr_cap()
        for pth in (script_path, stderr_cap_path):
            if pth:
                try:
                    os.unlink(pth)
                except Exception:
                    pass
    dur = time.monotonic() - t0

    out_t, trunc = _truncate("".join(stdout_chunks))
    if violation:
        # The process was killed for a policy violation. strace is a STREAMING
        # supervisor: it prints a syscall line only AFTER the syscall returns,
        # so a fast C program (e.g. `cat /etc/shadow`) can open→read→write its
        # stdout before the watcher's killpg lands — intermittently leaking
        # secret bytes. Since a violating process is untrusted by definition, we
        # deterministically SUPPRESS its stdout (and its captured stderr). This
        # closes the read-denylist exfiltration race (gate §6.2: 0-byte leak)
        # without weakening any boundary. Legitimate commands never get here.
        return SandboxResult(
            stdout="", stderr=f"[sandbox] {violation['msg']}",
            exit_code=126, timed_out=False, truncated=False, mode=mode,
            duration_sec=dur, violation=violation["msg"],
            violation_detail=dict(violation),
        )
    err_t, _ = _truncate(child_stderr)
    if timed_out:
        timeout_note = "[sandbox] command timed out"
        return SandboxResult(
            stdout=out_t,
            stderr=(err_t + "\n" + timeout_note) if err_t else timeout_note,
            exit_code=124, timed_out=True, truncated=trunc, mode=mode,
            duration_sec=dur,
        )
    return SandboxResult(
        stdout=out_t, stderr=err_t, exit_code=int(proc.returncode or 0),
        timed_out=False, truncated=trunc, mode=mode, duration_sec=dur,
    )


def _extract_strace_quoted_strings(line: str) -> list[str]:
    vals = re.findall(r'"((?:[^"\\]|\\.)*)"', line)
    out = []
    for v in vals:
        try:
            out.append(bytes(v, "utf-8").decode("unicode_escape"))
        except Exception:
            out.append(v)
    return out


def _check_strace_violation(line: str, workspace_root: str,
                            runtime_port: Optional[int],
                            run_root: Optional[str] = None) -> Optional[dict]:
    """Return a structured violation dict if this strace line breaks policy.

    Returns None when the line is policy-clean. On violation returns:
      {"rule", "syscall", "path_or_addr", "msg", "raw_strace_line"}
    where `msg` is the short human string (kept for backward compat / stderr)
    and `raw_strace_line` is the original strace stderr line for diagnostics.

    Three policies enforced:
      1. network: only localhost:<runtime_port> connect allowed;
      2. read: denylisted sensitive paths (/etc/passwd ...) blocked on any open;
      3. write: file mutation outside workspace (and never into canonical
         run_root artifacts) blocked.
    """
    padded = f" {line}"

    def _syscall_name() -> str:
        m = re.search(r"\b([a-z_][a-z0-9_]*)\(", line)
        return m.group(1) if m else ""

    def _v(rule: str, path_or_addr: str, msg: str) -> dict:
        return {"rule": rule, "syscall": _syscall_name(),
                "path_or_addr": path_or_addr, "msg": msg,
                "raw_strace_line": line.rstrip("\n")}

    # 1. network — block non-localhost AF_INET/AF_INET6 connect.
    if " connect(" in padded and ("AF_INET" in line or "AF_INET6" in line):
        host_ok = ('"127.0.0.1"' in line) or ('"::1"' in line)
        port_ok = (runtime_port is not None) and (f"htons({runtime_port})" in line)
        if not (host_ok and port_ok):
            return _v("network", "non-localhost",
                      "network policy violation: only localhost runtime port allowed")

    # 2. read denylist — any open/openat/stat of a sensitive path.
    if any(t in padded for t in (" open(", " openat(", " stat(", " lstat(",
                                  " newfstatat(", " access(")):
        for pt in _extract_strace_quoted_strings(line):
            if _is_denied_read_path(pt):
                return _v("read_denylist", pt,
                          f"read policy violation: {pt} is denylisted")

    # 2b. chdir containment. The write checker resolves
    # RELATIVE paths against workspace_root; a process that chdir'd elsewhere
    # (the agent ran `cd /workspace`) made relative writes land outside the
    # workspace while being judged as inside it. Containing chdir itself closes
    # that hole: absolute chdir is allowed only into workspace//tmp//var/tmp
    # subtrees, and relative chdir must not contain `..` (cannot climb out of
    # an allowed subtree without it). bash_exec always starts in the workspace.
    if " chdir(" in padded:
        for pt in _extract_strace_quoted_strings(line):
            if pt.startswith("/"):
                if not _is_allowed_chdir(pt, workspace_root):
                    return _v("chdir_outside_workspace", pt,
                              f"filesystem policy violation: cd outside the "
                              f"workspace is blocked ({pt}); stay in the "
                              f"workspace and use relative paths")
            elif ".." in pt.split("/"):
                return _v("chdir_relative_escape", pt,
                          f"filesystem policy violation: cd with '..' is "
                          f"blocked ({pt}); use paths inside the workspace")

    # 3. write outside workspace / into canonical run artifacts.
    write_syscall = any(t in padded for t in (
        " open(", " openat(", " creat(", " unlink(", " unlinkat(", " rename(",
        " renameat(", " renameat2(", " mkdir(", " rmdir(", " truncate(",
        " chmod(", " chown(",
    ))
    if write_syscall:
        writes_via_open = (
            ((" open(" in padded) or (" openat(" in padded))
            and any(f in line for f in
                    ("O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC", "O_APPEND"))
        )
        destructive = any(x in line for x in (
            " creat(", " unlink(", " unlinkat(", " rename(", " renameat(",
            " renameat2(", " mkdir(", " rmdir(", " truncate(", " chmod(", " chown("))
        if writes_via_open or destructive:
            for pt in _extract_strace_quoted_strings(line):
                if pt.startswith("/") or pt.startswith(".") or "/" in pt:
                    if not _is_allowed_write_path(pt, workspace_root, run_root):
                        return _v("write_outside_workspace", pt,
                                  f"filesystem policy violation: write outside workspace blocked ({pt})")
    return None


def _is_allowed_chdir(raw_path: str, workspace_root: str) -> bool:
    """Absolute chdir targets allowed only inside workspace / scratch tmp."""
    try:
        resolved = Path(raw_path).resolve()
    except Exception:
        return False
    for root in (Path(workspace_root).resolve(), Path("/tmp"), Path("/var/tmp")):
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _is_allowed_write_path(raw_path: str, workspace_root: str,
                           run_root: Optional[str] = None) -> bool:
    """Writes allowed only under workspace_root (minus arc_utils.py) or scratch
    temp dirs. Canonical run artifacts under run_root are NEVER writable, even
    when run_root itself lives under /tmp."""
    if not raw_path:
        return True
    try:
        ws = Path(workspace_root).resolve()
        p = Path(raw_path)
        resolved = p.resolve() if p.is_absolute() else (ws / p).resolve()
    except Exception:
        return False
    # 1) Inside workspace → allowed, except read-only arc_utils.py.
    try:
        rel = resolved.relative_to(ws)
        if rel == Path("arc_utils.py"):
            return False
        return True
    except ValueError:
        pass
    # 2) Canonical run artifacts (session/recorder/*.jsonl/metadata) → denied,
    #    checked BEFORE the /tmp allowlist so a run_root under /tmp is protected.
    if run_root:
        try:
            run = Path(run_root).resolve()
            rel_run = resolved.relative_to(run)
            parts = rel_run.parts
            # workspace under run_root is handled by case (1); everything else
            # under run_root is canonical and read-only to the agent.
            if not (parts and parts[0] == "workspace"):
                return False
        except ValueError:
            pass
    # 3) Scratch temp dirs allowed.
    for tmp in ("/var/tmp", "/dev/null", "/dev/urandom", "/dev/random"):
        try:
            resolved.relative_to(Path(tmp))
            return True
        except ValueError:
            continue
    # /tmp allowed only when NOT inside run_root (handled above).
    try:
        resolved.relative_to(Path("/tmp"))
        return True
    except ValueError:
        pass
    if str(resolved) in ("/dev/null", "/dev/urandom", "/dev/random", "/dev/tty"):
        return True
    return False



def _kill_process_group(proc) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except Exception:
            pass


# ── Path policy (dev-degraded enforcement + shared logic) ───────────────────
CANONICAL_DIRS = {"session", "recorder", "compact"}
CANONICAL_FILES = {"events.jsonl", "actions.jsonl",
                   "run_metadata.json", "counters.json"}


def is_path_allowed_for_write(path: str, workspace_root: str, run_root: str) -> bool:
    """Return True if `path` is a permitted agent write target.

    Allowed: anywhere under workspace/ EXCEPT arc_utils.py.
    Forbidden: canonical run artifacts (session/, recorder/, *.jsonl, etc.),
    and arc_utils.py.
    """
    p = Path(path).resolve()
    ws = Path(workspace_root).resolve()
    run = Path(run_root).resolve()

    # must be inside workspace
    try:
        rel = p.relative_to(ws)
    except ValueError:
        return False
    # arc_utils.py is read-only
    if rel.parts and rel.parts[-1] == "arc_utils.py" and len(rel.parts) == 1:
        return False
    return True


def is_path_forbidden_read(path: str) -> bool:
    """Coarse dev-degraded read denylist (credential-bearing files only).

    Aligned with the strace path (READ_DENYLIST / _is_denied_read_path):
    /etc/passwd is world-readable and NOT a credential, so it is not forbidden.
    Only genuine credential paths are denied, with path-boundary matching (no
    naive-prefix false hits)."""
    return _is_denied_read_path(str(Path(path)))
