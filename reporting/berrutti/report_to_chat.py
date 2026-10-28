"""Deliver monitor reports through the app server that owns the chat.

A second ``codex exec resume`` process cannot write a chat loaded in the app.
Use the existing Unix-socket server and defer reports while the user is active.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4


def report_prompt(snapshot: dict | None) -> str:
    checked_at = None if snapshot is None else snapshot.get("checked_at_utc")
    age_minutes = None
    if checked_at:
        checked = datetime.fromisoformat(checked_at)
        if checked.tzinfo is None:
            raise ValueError("Snapshot timestamp must include a timezone")
        age_minutes = round(
            (
                datetime.now(timezone.utc) - checked.astimezone(timezone.utc)
            ).total_seconds()
            / 60,
            1,
        )
    payload = {"age_minutes": age_minutes, "snapshot": snapshot}
    return (
        "Informe programado Qwen–Berrutti. No reemplaza solicitudes en curso. "
        "Informá brevemente en español en ESTE MISMO chat, sólo con este JSON; no uses herramientas ni hagas cambios. "
        "Reportá en cada ejecución los tres experimentos Berrutti: sólo páginas, sólo líneas y combinación intercalada. "
        "Cada uno usa las 125 páginas train o sus 5212 líneas, durante tres épocas completas. "
        "PLANNED significa definido pero todavía no lanzado, sin entrenamiento ni CER. "
        "La línea base compartida se mide una sola vez antes de la primera época; no es entrenamiento. "
        "Si existe shared_baseline, informá su estado y avance como evaluación previa, sin contarlo como cuarto experimento. "
        "PENDING significa enviado pero en cola o esperando dependencias, todavía sin entrenamiento. "
        "Si hay queue, explicá reason (Resources: faltan recursos; Dependency: espera una fase previa); "
        "estimated_start_utc es una estimación variable, nunca una hora garantizada; convertí a Europe/Paris si la mostrás. "
        "Después de cada época se evalúan las 25 páginas de validación; el test de 25 páginas "
        "se hace sólo al final para el checkpoint elegido de cada modelo. "
        "Reportá estado, época, avance y CER disponibles; selección por validación completa y test descriptivo. "
        "Informá failed_pages y retried_pages; CER incompleto es provisional y no acredita mejoras. "
        "LUISA queda fuera; su ampliación futura tiene dos variantes: añadir líneas al mecanismo ganador "
        "o construir páginas sintéticas con esas imágenes de líneas, después de auditar exclusiones. "
        "Es una propuesta, no un entrenamiento lanzado. Los experimentos anteriores se retiraron y "
        "sus resultados se borraron por pedido del usuario; no los vuelvas a presentar como corridas actuales. "
        "Si hay performance_checks, resumilos juntos en un único párrafo como comprobaciones de optimización, "
        "sin enumerar cada trabajo o reintento, "
        "sin contarlos como experimentos ni épocas y sin extrapolar CER o velocidad. "
        "Sólo están completos con Slurm COMPLETED y recibo exitoso; un runtime GPU exitoso "
        "no anula un corte posterior. No declares ausencia de GPU por un corte de importación CPU. "
        "Un diagnóstico completado con optimization_accepted=false no acredita la optimización; "
        "generation_checked=false significa que todavía no se comprobó generación. "
        "No prometas que la combinación ganará. Señalá snapshot ausente o de más de 30 minutos. "
        "No reveles rutas, IDs de trabajo ni texto OCR.\n\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


class Deferred(Exception):
    """Retry later without changing or interrupting the active chat."""


class RpcError(Exception):
    pass


class AppServer:
    def __init__(self, websocket):
        self.websocket = websocket
        self.sequence = 0
        self.completed: dict[str, dict] = {}

    async def receive(self, timeout: float) -> dict:
        message = json.loads(
            await asyncio.wait_for(self.websocket.recv(), timeout=max(timeout, 0.01))
        )
        if message.get("method") == "turn/completed":
            turn = message["params"]["turn"]
            self.completed[turn["id"]] = turn
        # This report client does not execute tools or answer approvals.
        if "method" in message and "id" in message:
            await self.websocket.send(
                json.dumps(
                    {
                        "id": message["id"],
                        "error": {
                            "code": -32601,
                            "message": "The monitor client only delivers status reports",
                        },
                    }
                )
            )
        return message

    async def call(self, method: str, params: dict) -> dict:
        self.sequence += 1
        request_id = self.sequence
        await self.websocket.send(
            json.dumps({"id": request_id, "method": method, "params": params})
        )
        deadline = time.monotonic() + 30
        while True:
            message = await self.receive(deadline - time.monotonic())
            if message.get("id") == request_id:
                if "error" in message:
                    raise RpcError(str(message["error"]))
                return message["result"]

    async def wait_turn(self, turn_id: str) -> dict:
        deadline = time.monotonic() + 900
        while turn_id not in self.completed:
            await self.receive(deadline - time.monotonic())
        return self.completed[turn_id]


def write_receipt(path: Path, receipt: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


async def deliver(server, args) -> dict:
    thread_id = str(UUID(args.thread_id))
    receipt = json.loads(args.receipt.read_text()) if args.receipt.exists() else {}
    if receipt.get("phase") in {"dispatching", "accepted"}:
        if not receipt.get("turn_id"):
            raise RpcError("Ambiguous previous dispatch; inspect chat before retrying")
        response = await server.call(
            "thread/turns/list",
            {"threadId": thread_id, "limit": 20, "itemsView": "notLoaded"},
        )
        turn = next(
            (t for t in response["data"] if t["id"] == receipt["turn_id"]), None
        )
        if turn and turn["status"] == "completed":
            receipt["phase"] = "completed"
            write_receipt(args.receipt, receipt)
            return receipt
        if turn and turn["status"] == "inProgress":
            raise Deferred("Previously accepted report is still running")
        raise RpcError("Previous report did not complete; inspect chat before retrying")

    response = await server.call(
        "thread/read", {"threadId": thread_id, "includeTurns": False}
    )
    if response["thread"]["status"]["type"] in {"active", "systemError"}:
        raise Deferred("Chat is active; postponing the report")
    await server.call("thread/resume", {"threadId": thread_id, "excludeTurns": True})
    if args.refresh_unit:
        result = await asyncio.to_thread(
            subprocess.run,
            ["systemctl", "--user", "start", args.refresh_unit],
            capture_output=True,
            timeout=140,
            check=False,
        )
        if result.returncode:
            print(
                "Status refresh failed; reporting the saved timestamp", file=sys.stderr
            )
    # Refresh may take time; don't send into a turn started meanwhile.
    response = await server.call(
        "thread/read", {"threadId": thread_id, "includeTurns": False}
    )
    if response["thread"]["status"]["type"] != "idle":
        raise Deferred("Chat became active; postponing the report")
    snapshot = json.loads(args.snapshot.read_text()) if args.snapshot.exists() else None
    receipt = {
        "phase": "dispatching",
        "thread_id": thread_id,
        "client_message_id": str(uuid4()),
        "snapshot_checked_at": (snapshot or {}).get("checked_at_utc"),
        "attempted_at": datetime.now(timezone.utc).isoformat(),
    }
    write_receipt(args.receipt, receipt)
    result = await server.call(
        "turn/start",
        {
            "threadId": thread_id,
            "input": [
                {
                    "type": "text",
                    "text": (
                        args.prompt_file.read_text()
                        if getattr(args, "prompt_file", None)
                        else report_prompt(snapshot)
                    ),
                }
            ],
            "clientUserMessageId": receipt["client_message_id"],
            "turnTrigger": "berrutti-monitor",
        },
    )
    receipt.update(phase="accepted", turn_id=result["turn"]["id"])
    write_receipt(args.receipt, receipt)
    turn = await server.wait_turn(receipt["turn_id"])
    receipt.update(
        phase=turn["status"], finished_at=datetime.now(timezone.utc).isoformat()
    )
    write_receipt(args.receipt, receipt)
    if turn["status"] != "completed":
        raise RpcError(f"Report turn ended with status {turn['status']}")
    return receipt


async def run(args) -> None:
    # Optional host dependency; core benchmark and unit tests don't load it.
    from websockets.asyncio.client import unix_connect
    from websockets.exceptions import ConnectionClosed

    try:
        async with unix_connect(
            str(args.socket), uri="ws://localhost", open_timeout=10, max_size=2**22
        ) as websocket:
            server = AppServer(websocket)
            await server.call(
                "initialize",
                {
                    "clientInfo": {"name": "berrutti_monitor", "version": "2.0"},
                    "capabilities": {"experimentalApi": True},
                },
            )
            await websocket.send(json.dumps({"method": "initialized", "params": {}}))
            print(json.dumps(await deliver(server, args)))
    except ConnectionClosed as error:
        raise Deferred(
            "App server disconnected; retry after checking delivery receipt"
        ) from error


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread-id", required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--refresh-unit")
    parser.add_argument("--prompt-file", type=Path)
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except (Deferred, OSError, TimeoutError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(75) from error
    except RpcError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
