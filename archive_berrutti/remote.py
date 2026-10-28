"""Named destinations: local UI, SSH metadata and Slurm data processing."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from archive_berrutti import library, manage as core

RUNTIME_ENV = "ARCHIVE_BERRUTTI_RUNTIME"
CONFIG_ENV = "ARCHIVE_BERRUTTI_CONFIG"
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def config_path():
    return Path(
        os.environ.get(
            CONFIG_ENV, core.REPO / ".private/archive_berrutti/profiles.json"
        )
    ).expanduser()


def configuration():
    path = config_path()
    value = (
        core.load(path)
        if path.exists()
        else {"schema_version": 1, "default": "local", "profiles": {}}
    )
    if value.get("schema_version") != 1 or not isinstance(value.get("profiles"), dict):
        raise ValueError("Configuración de perfiles inválida")
    if value["default"] != "local" and value["default"] not in value["profiles"]:
        raise ValueError("El perfil predeterminado no está configurado")
    return value


def validate(profile, *, deployed=False):
    for field in ("name", "host", "account"):
        if not NAME.fullmatch(profile[field]):
            raise ValueError(f"{field}: usar un nombre simple; host es un alias SSH")
    workdir = Path(profile["workdir"])
    for field in ("workdir", "root", "python"):
        path = Path(profile[field])
        if (
            not path.is_absolute()
            or ".." in path.parts
            or any(char in profile[field] for char in ("\n", "\r", "\x00"))
        ):
            raise ValueError(f"{field}: requiere una ruta absoluta sin saltos de línea")
    if workdir == Path("/") or not Path(profile["root"]).is_relative_to(workdir):
        raise ValueError("La biblioteca debe estar dentro del WORKDIR configurado")
    if Path(profile["root"]) == workdir:
        raise ValueError("La biblioteca necesita su propia subcarpeta de WORKDIR")
    for kind in ("session", "worker"):
        shape = profile[kind]
        if not NAME.fullmatch(shape["partition"]):
            raise ValueError("Partición inválida")
        if not re.fullmatch(r"(?:\d+-)?\d{1,3}:\d{2}:\d{2}", shape["time"]):
            raise ValueError("Tiempo Slurm inválido")
        if not re.fullmatch(r"[1-9]\d*[MGT]?", shape["mem"]):
            raise ValueError("Memoria Slurm inválida")
        if not isinstance(shape["cpus"], int) or not 1 <= shape["cpus"] <= 16:
            raise ValueError("CPU por trabajo debe estar entre 1 y 16")
    if deployed:
        for field in ("code_root", "runtime_file"):
            path = Path(profile.get(field, ""))
            if (
                not path.is_absolute()
                or not path.is_relative_to(workdir)
                or ".." in path.parts
            ):
                raise ValueError("Perfil sin desplegar: ejecutar profile deploy NOMBRE")
    return profile


def runtime():
    path = os.environ.get(RUNTIME_ENV)
    if not path:
        return None
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("El perfil remoto requiere una asignación Slurm")
    return validate(core.load(Path(path)), deployed=True)


def check_root(root, profile):
    path = Path(root).expanduser().resolve()
    workdir = Path(profile["workdir"]).resolve()
    if path == workdir or not path.is_relative_to(workdir):
        raise ValueError("La biblioteca remota debe quedar dentro de WORKDIR")
    code = Path(profile["code_root"]).resolve().parent
    if path.is_relative_to(code) or code.is_relative_to(path):
        raise ValueError("Usar una biblioteca separada de las versiones de código")
    return path


def location(root):
    profile = runtime()
    return {
        "profile": profile["name"] if profile else "local",
        "host": profile["host"] if profile else None,
        "root": str(root) if root is not None else None,
        "scheduler": "slurm" if profile else None,
    }


def label(root):
    profile = runtime()
    return f"{profile['name']} · {profile['host']}:{root}" if profile else str(root)


def ssh_base(host, *, interactive=False):
    if not NAME.fullmatch(host):
        raise ValueError("Alias SSH inválido")
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        *(["-tt"] if interactive else ["-T"]),
        host,
    ]


def ssh_script(profile, script, **kwargs):
    return subprocess.run(
        [*ssh_base(profile["host"]), "bash -l -c " + shlex.quote(script)],
        check=True,
        text=True,
        **kwargs,
    )


def extract_destination(arguments):
    """Destination flags work before or after subcommands; explicit roots stay local."""
    remaining, name = [], None
    iterator = iter(arguments)
    for argument in iterator:
        if argument == "--local":
            selected = "local"
        elif argument == "--profile" or argument.startswith("--profile="):
            selected = (
                argument.split("=", 1)[1] if "=" in argument else next(iterator, "")
            )
            if not selected:
                raise ValueError("--profile requiere un nombre")
        else:
            remaining.append(argument)
            continue
        if name is not None and selected != name:
            raise ValueError("Elegir --local o un único --profile")
        name = selected
    if os.environ.get(RUNTIME_ENV):
        runtime()  # Validate the allocation before trusting a forwarded invocation.
        if name not in (None, "local"):
            raise ValueError("Cambiar servidor desde el cliente local")
        return remaining, None
    if name is None and any(
        arg == "--root" or arg.startswith("--root=") for arg in remaining
    ):
        name = "local"
    if name == "local" or any(arg in ("--help", "-h") for arg in remaining):
        return remaining, None
    config = configuration()
    name = name or config["default"]
    if name == "local":
        return remaining, None
    if name not in config["profiles"]:
        raise ValueError(f"Perfil desconocido: {name}. Ver profile show.")
    return remaining, validate(config["profiles"][name], deployed=True)


def resource_args(profile, kind):
    shape = profile[kind]
    return [
        "--account",
        profile["account"],
        "--partition",
        shape["partition"],
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={shape['cpus']}",
        f"--mem={shape['mem']}",
        f"--time={shape['time']}",
    ]


def execution_script(profile, arguments, *, interactive=False):
    arguments = arguments or ["browse"]
    if not any(arg == "--root" or arg.startswith("--root=") for arg in arguments):
        arguments = [*arguments, "--root", profile["root"]]
    command = (
        arguments[1] if arguments[0] == "agent" and len(arguments) > 1 else arguments[0]
    )
    kind = (
        "worker"
        if command in {"complete", "download", "extract", "render", "prepare", "verify"}
        and "--background" not in arguments
        and "--dry-run" not in arguments
        else "session"
    )
    options = resource_args(profile, kind)
    if not interactive and kind == "session":
        options = [arg for arg in options if not arg.startswith("--time=")]
        options.append("--time=00:10:00")
    terminal = os.environ.get("TERM", "xterm-256color")
    environment = [
        "env",
        f"{RUNTIME_ENV}={profile['runtime_file']}",
        f"WORKDIR={profile['workdir']}",
        "PYTHONDONTWRITEBYTECODE=1",
        f"OMP_NUM_THREADS={profile[kind]['cpus']}",
        f"TERM={terminal}",
        "LANG=C.UTF-8",
    ]
    # SSH does not forward these automatically; preserve viewer capabilities,
    # including conservative fallback when the local terminal is multiplexed.
    for name in (
        "COLORTERM",
        "TERM_PROGRAM",
        "KITTY_WINDOW_ID",
        "ITERM_SESSION_ID",
        "TMUX",
        "STY",
        "ARCHIVE_BERRUTTI_VIEWER",
    ):
        if value := os.environ.get(name):
            environment.append(f"{name}={value}")
    return (
        "set -eu\numask 077\ncd "
        + shlex.quote(profile["code_root"])
        + "\nexec "
        + shlex.join(
            [
                "srun",
                "--job-name=berrutti-browser"
                if interactive
                else "--job-name=berrutti-query",
                *options,
                "--unbuffered",
                *(["--pty"] if interactive else []),
                *environment,
                profile["python"],
                "-u",
                "-m",
                "archive_berrutti",
                *arguments,
            ]
        )
    )


def launch(profile, arguments):
    from archive_berrutti import cli, remote_library, tui

    arguments = arguments or ["browse"]
    if (
        arguments[0] == "agent"
        and len(arguments) > 1
        and (
            arguments[1] in {"list", "info", "files", "plan", "job"}
            or "--dry-run" in arguments
        )
    ):
        return remote_library.query(profile, arguments)
    if arguments[0] == "browse":
        args = cli.parser().parse_args(arguments)
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            raise ValueError(
                "El explorador necesita una terminal; para agentes usar agent list"
            )
        client = remote_library.Client(profile, args.root)
        try:
            tui.run(args, backend=client)
        finally:
            client.close()
        return 0
    return launch_compute(profile, arguments)


def launch_compute(profile, arguments):
    from archive_berrutti import agent, cli

    arguments = arguments or ["browse"]
    machine = arguments[0] == "agent"
    (agent.parser() if machine else cli.parser()).parse_args(
        arguments[1:] if machine else arguments
    )
    interactive = arguments[0] in {"browse", "view"}
    if interactive and (not sys.stdin.isatty() or not sys.stdout.isatty()):
        raise ValueError(
            "El explorador necesita una terminal; para agentes usar agent list"
        )
    script = execution_script(profile, arguments, interactive=interactive)
    if interactive:
        print(
            f"Abriendo {profile['name']} · {profile['host']}:{profile['root']}\nEsperando asignación CPU de Slurm…",
            file=sys.stderr,
            flush=True,
        )
    command = [
        *ssh_base(profile["host"], interactive=interactive),
        "bash -l -c " + shlex.quote(script),
    ]
    if not machine:
        return subprocess.run(
            command, stdin=None if interactive else subprocess.DEVNULL
        ).returncode
    # Capture one JSON response, keeping SSH/Slurm diagnostics off stdout.
    result = subprocess.run(
        command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True
    )
    try:
        response = json.loads(result.stdout)
        if response.get("schema_version") != 1 or "ok" not in response:
            raise ValueError("Contrato remoto desconocido")
    except (ValueError, AttributeError) as error:
        raise ValueError(
            f"No se recibió JSON del servidor (salida SSH {result.returncode}); ver stderr"
        ) from error
    print(json.dumps(response, ensure_ascii=False))
    return result.returncode


def deploy(profile):
    """Copy only reviewed package files to a new release; never install on login."""
    profile = validate(dict(profile))
    release = library.identifier()
    code = Path(profile["workdir"]) / "berrutti-data" / "code" / release
    profile.update(
        code_root=str(code), runtime_file=str(code / "profile.json"), release=release
    )
    source = Path(__file__).parent
    with tempfile.TemporaryDirectory() as temporary:
        staging = Path(temporary)
        package = staging / "archive_berrutti"
        package.mkdir()
        hashes = {}
        for path in sorted(source.iterdir()):
            if path.suffix not in {".py", ".md", ".sbatch"}:
                continue
            shutil.copy2(path, package / path.name)
            hashes[path.name] = core.hashes(path)["sha256"]
        (staging / "profile.json").write_text(json.dumps(profile, indent=2) + "\n")
        (staging / "release.json").write_text(
            json.dumps(
                {"release": release, "files": hashes, "created": core.now()}, indent=2
            )
            + "\n"
        )
        ssh_script(
            profile,
            "set -eu\numask 077\nmkdir -p "
            + shlex.quote(str(code.parent))
            + "\nmkdir "
            + shlex.quote(str(code)),
            capture_output=True,
            timeout=45,
        )
        subprocess.run(
            [
                "rsync",
                "-rt",
                "--protect-args",
                "--chmod=D700,F600",
                "-e",
                shlex.join(ssh_base(profile["host"])[:-1]),
                str(staging) + "/",
                profile["host"] + ":" + str(code) + "/",
            ],
            check=True,
            timeout=120,
        )
    return profile


def profile_command(arguments):
    parser = argparse.ArgumentParser(
        prog="archive-berrutti profile",
        description="Perfiles privados de biblioteca local o SSH + Slurm",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "show", help="Mostrar destino predeterminado y perfiles, en JSON"
    )
    use = commands.add_parser("use", help="Elegir el destino predeterminado")
    use.add_argument("name")
    configure = commands.add_parser(
        "configure", help="Registrar un servidor; no despliega ni descarga datos"
    )
    configure.add_argument("name")
    for field in ("host", "root", "workdir", "python", "account"):
        configure.add_argument("--" + field, required=True)
    for kind in ("session", "worker"):
        configure.add_argument(f"--{kind}-partition", default="cpu_med")
        configure.add_argument(f"--{kind}-time", default="04:00:00")
    deploy_parser = commands.add_parser(
        "deploy", help="Transferir solo código a una versión nueva en el servidor"
    )
    deploy_parser.add_argument("name")
    args = parser.parse_args(arguments)
    config = configuration()
    if args.command == "configure":
        if args.name == "local":
            raise ValueError("local es un nombre reservado")
        profile = {
            field: getattr(args, field)
            for field in ("name", "host", "root", "workdir", "python", "account")
        }
        for kind, cpus, mem in (("session", 1, "2G"), ("worker", 2, "8G")):
            profile[kind] = {
                "partition": getattr(args, kind + "_partition"),
                "time": getattr(args, kind + "_time"),
                "cpus": cpus,
                "mem": mem,
            }
        config["profiles"][args.name] = validate(profile)
    elif args.command == "use":
        if args.name != "local":
            if args.name not in config["profiles"]:
                raise ValueError("Perfil desconocido")
            validate(config["profiles"][args.name], deployed=True)
        config["default"] = args.name
    elif args.command == "deploy":
        if args.name not in config["profiles"]:
            raise ValueError("Configurar el perfil antes de desplegarlo")
        config["profiles"][args.name] = deploy(config["profiles"][args.name])
    if args.command != "show":
        library.replace_json(config_path(), config)
    print(
        json.dumps(
            {"config_path": str(config_path()), **config}, ensure_ascii=False, indent=2
        )
    )
