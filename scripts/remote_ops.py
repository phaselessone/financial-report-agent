from __future__ import annotations

import argparse
import fnmatch
import os
from pathlib import Path
from stat import S_ISDIR
from typing import Iterable

import paramiko


def _env(name: str, *, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def _configure_host_keys(client: paramiko.SSHClient) -> None:
    client.load_system_host_keys()
    known_hosts_path = os.environ.get("REMOTE_KNOWN_HOSTS_PATH")
    if known_hosts_path:
        client.load_host_keys(str(Path(known_hosts_path)))
    if os.environ.get("REMOTE_ALLOW_INSECURE_HOSTKEY", "").lower() in {"1", "true", "yes"}:
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        return
    client.set_missing_host_key_policy(paramiko.RejectPolicy())


def _connect(auth_mode: str) -> paramiko.SSHClient:
    host = _env("REMOTE_HOST")
    port = int(_env("REMOTE_PORT", default="22"))
    user = _env("REMOTE_USER")

    client = paramiko.SSHClient()
    _configure_host_keys(client)

    kwargs = {"hostname": host, "port": port, "username": user, "timeout": 30}
    if auth_mode == "password":
        kwargs["password"] = _env("REMOTE_PASSWORD")
    elif auth_mode == "key":
        key_path = Path(_env("REMOTE_KEY_PATH"))
        kwargs["key_filename"] = str(key_path)
        passphrase = os.environ.get("REMOTE_KEY_PASSPHRASE")
        if passphrase:
            kwargs["passphrase"] = passphrase
    else:
        raise SystemExit(f"Unsupported auth mode: {auth_mode}")

    client.connect(**kwargs)
    return client


def _exec_command(client: paramiko.SSHClient, command: str) -> int:
    stdin, stdout, stderr = client.exec_command(command)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    if out:
        print(out, end="")
    if err:
        print(err, end="")
    return stdout.channel.recv_exit_status()


def _mkdir_p(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
    parts = Path(remote_dir).as_posix().strip("/").split("/")
    current = ""
    for part in parts:
        current = f"{current}/{part}" if current else f"/{part}"
        try:
            sftp.stat(current)
        except FileNotFoundError:
            sftp.mkdir(current)


def _remote_exists(sftp: paramiko.SFTPClient, remote_path: str) -> bool:
    try:
        sftp.stat(remote_path)
    except FileNotFoundError:
        return False
    return True


def _read_remote_file(sftp: paramiko.SFTPClient, remote_path: str) -> str:
    with sftp.open(remote_path, "r") as handle:
        return handle.read().decode("utf-8")


def install_key(pubkey_path: Path) -> int:
    client = _connect("password")
    try:
        with client.open_sftp() as sftp:
            _mkdir_p(sftp, "/root/.ssh")
            authorized_keys = "/root/.ssh/authorized_keys"
            key_text = pubkey_path.read_text(encoding="utf-8").strip()
            existing = _read_remote_file(sftp, authorized_keys) if _remote_exists(sftp, authorized_keys) else ""
            if key_text not in existing:
                combined = (existing.rstrip() + "\n" + key_text + "\n") if existing.strip() else key_text + "\n"
                with sftp.open(authorized_keys, "w") as handle:
                    handle.write(combined)
        exit_code = _exec_command(client, "chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys")
        return exit_code
    finally:
        client.close()


def _iter_local_files(local_root: Path, exclude: Iterable[str]) -> Iterable[tuple[Path, str]]:
    default_excludes = (
        "__pycache__",
        "**/__pycache__/*",
        "*.pyc",
        ".venv",
        ".venv/*",
        ".ops-venv",
        ".ops-venv/*",
    )
    exclude_patterns = default_excludes + tuple(exclude)
    for path in sorted(local_root.rglob("*")):
        rel = path.relative_to(local_root).as_posix()
        if any(fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(path.name, pattern) for pattern in exclude_patterns):
            continue
        yield path, rel


def upload(local_root: Path, remote_root: str, exclude: Iterable[str]) -> int:
    client = _connect("key")
    try:
        with client.open_sftp() as sftp:
            _mkdir_p(sftp, remote_root)
            for path, rel in _iter_local_files(local_root, exclude):
                remote_path = f"{remote_root.rstrip('/')}/{rel}"
                if path.is_dir():
                    _mkdir_p(sftp, remote_path)
                    continue
                _mkdir_p(sftp, str(Path(remote_path).parent).replace("\\", "/"))
                sftp.put(str(path), remote_path)
                print(f"Uploaded {rel}")
        return 0
    finally:
        client.close()


def exec_remote(command: str, auth_mode: str) -> int:
    if auth_mode == "password" and os.environ.get("REMOTE_ALLOW_PASSWORD_AUTH", "").lower() not in {"1", "true", "yes"}:
        raise SystemExit(
            "Password auth is disabled for routine remote operations. "
            "Use key auth, or set REMOTE_ALLOW_PASSWORD_AUTH=1 explicitly for a one-off bootstrap."
        )
    client = _connect(auth_mode)
    try:
        return _exec_command(client, command)
    finally:
        client.close()


def download(remote_path: str, local_root: Path) -> int:
    client = _connect("key")
    try:
        with client.open_sftp() as sftp:
            remote_stat = sftp.stat(remote_path)
            local_root.mkdir(parents=True, exist_ok=True)
            if not S_ISDIR(remote_stat.st_mode):
                target = local_root / Path(remote_path).name
                sftp.get(remote_path, str(target))
                print(f"Downloaded {target}")
                return 0
            _download_dir(sftp, remote_path, local_root)
        return 0
    finally:
        client.close()


def _download_dir(sftp: paramiko.SFTPClient, remote_root: str, local_root: Path) -> None:
    local_root.mkdir(parents=True, exist_ok=True)
    for entry in sftp.listdir_attr(remote_root):
        remote_child = f"{remote_root.rstrip('/')}/{entry.filename}"
        local_child = local_root / entry.filename
        if S_ISDIR(entry.st_mode):
            _download_dir(sftp, remote_child, local_child)
        else:
            sftp.get(remote_child, str(local_child))
            print(f"Downloaded {local_child}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remote SSH/SFTP helper.")
    subparsers = parser.add_subparsers(dest="action", required=True)

    install_key_parser = subparsers.add_parser("install-key")
    install_key_parser.add_argument("--pubkey", type=Path, required=True)

    upload_parser = subparsers.add_parser("upload")
    upload_parser.add_argument("--local", type=Path, required=True)
    upload_parser.add_argument("--remote", required=True)
    upload_parser.add_argument("--exclude", action="append", default=[])

    exec_parser = subparsers.add_parser("exec")
    exec_parser.add_argument("--auth", choices=("password", "key"), default="key")
    exec_parser.add_argument("--remote-command", required=True)

    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("--remote", required=True)
    download_parser.add_argument("--local", type=Path, required=True)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.action == "install-key":
        return install_key(args.pubkey)
    if args.action == "upload":
        return upload(args.local, args.remote, args.exclude)
    if args.action == "exec":
        return exec_remote(args.remote_command, args.auth)
    if args.action == "download":
        return download(args.remote, args.local)
    raise SystemExit("Unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
