from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import secrets
import socket
import sqlite3
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import qrcode
import imageio_ffmpeg
from flask import Flask, abort, jsonify, request, send_file, send_from_directory, session
from werkzeug.utils import secure_filename


PROJECT_DIR = Path(__file__).resolve().parent
if os.name == "nt":
    DATA_ROOT = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
else:
    DATA_ROOT = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
DATA_DIR = DATA_ROOT / "SquatLab"
DATABASE_PATH = DATA_DIR / "app.sqlite3"
SECRET_PATH = DATA_DIR / "session_secret.txt"
WEB_DIR = PROJECT_DIR / "web"
ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm", "m4v"}
MAX_UPLOAD_BYTES = 250 * 1024 * 1024
PASSWORD_ROUNDS = 310_000

DATA_DIR.mkdir(parents=True, exist_ok=True)
if not SECRET_PATH.is_file():
    SECRET_PATH.write_text(secrets.token_hex(48), encoding="ascii")

app = Flask(__name__)
app.secret_key = SECRET_PATH.read_text(encoding="ascii").strip()
app.config.update(
    MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("YOLO_HTTPS", "0") == "1",
)

_analysis_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


def _init_database() -> None:
    with _connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                password_salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS analyses (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id),
                participant TEXT NOT NULL,
                source_name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                error TEXT,
                session_dir TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS analyses_by_user ON analyses(user_id, created_at DESC);
            """
        )


_init_database()


def _json_error(message: str, status: int):
    return jsonify({"error": message}), status


def _current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    with _connect() as connection:
        return connection.execute("SELECT id, username FROM users WHERE id = ?", (user_id,)).fetchone()


def _require_user():
    user = _current_user()
    if user is None:
        abort(401)
    return user


def _user_session_dir(user_id: str, analysis_id: str) -> Path:
    return DATA_DIR / "users" / user_id / "sessions" / analysis_id


def _owned_analysis(analysis_id: str, user_id: str):
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM analyses WHERE id = ? AND user_id = ?", (analysis_id, user_id)
        ).fetchone()
    return row


def _run_analysis(analysis_id: str, user_id: str, video_path: Path, session_dir: Path, participant: str):
    try:
        opencv_video = session_dir / "sentadilla_opencv.mp4"
        browser_video = session_dir / "sentadilla_anotada.mp4"
        with _analysis_lock:
            command = [
                sys.executable,
                str(PROJECT_DIR / "pose_tracker.py"),
                "--source", str(video_path),
                "--output", str(opencv_video),
                "--csv", str(session_dir / "sentadilla_metricas.csv"),
                "--summary", str(session_dir / "sentadilla_resumen.json"),
                "--participant", participant,
                "--headless",
            ]
            completed = subprocess.run(
                command,
                cwd=PROJECT_DIR,
                capture_output=True,
                text=True,
                timeout=60 * 60,
                check=False,
            )
        if completed.returncode != 0:
            error = (completed.stderr or completed.stdout or "El análisis terminó con error")[-4000:]
            status = "error"
        elif not opencv_video.is_file() or opencv_video.stat().st_size == 0:
            error = "El análisis no generó fotogramas de video. Revisa que el archivo se pueda reproducir y prueba MP4 H.264."
            status = "error"
        else:
            transcode = subprocess.run(
                [
                    imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(opencv_video),
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
                    str(browser_video),
                ],
                capture_output=True,
                text=True,
                timeout=10 * 60,
                check=False,
            )
            if transcode.returncode != 0 or not browser_video.is_file() or browser_video.stat().st_size == 0:
                error = (transcode.stderr or "No se pudo convertir el video a un formato compatible con el navegador.")[-4000:]
                status = "error"
            else:
                opencv_video.unlink(missing_ok=True)
                error = None
                status = "completed"
    except subprocess.TimeoutExpired:
        error = "El análisis superó el límite de 60 minutos."
        status = "error"
    except Exception as exception:  # Keep the user's session visible if a worker fails.
        error = str(exception)[-4000:]
        status = "error"

    with _connect() as connection:
        connection.execute(
            "UPDATE analyses SET status = ?, error = ? WHERE id = ? AND user_id = ?",
            (status, error, analysis_id, user_id),
        )


@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/manifest.webmanifest")
def manifest():
    return send_from_directory(WEB_DIR, "manifest.webmanifest")


@app.get("/sw.js")
def service_worker():
    response = send_from_directory(WEB_DIR, "sw.js", mimetype="application/javascript")
    response.headers["Service-Worker-Allowed"] = "/"
    return response


@app.get("/assets/<path:filename>")
def assets(filename: str):
    return send_from_directory(WEB_DIR / "assets", filename)


@app.post("/api/register")
def register():
    payload = request.get_json(silent=True) or {}
    username = str(payload.get("username", "")).strip()
    password = str(payload.get("password", ""))
    if len(username) < 3 or len(username) > 40:
        return _json_error("El usuario debe tener entre 3 y 40 caracteres.", 400)
    if len(password) < 8:
        return _json_error("La contraseña debe tener al menos 8 caracteres.", 400)
    salt = os.urandom(16)
    password_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_ROUNDS)
    user_id = uuid.uuid4().hex
    try:
        with _connect() as connection:
            connection.execute(
                "INSERT INTO users(id, username, password_salt, password_hash, created_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, username, salt.hex(), password_hash.hex(), datetime.now(timezone.utc).isoformat()),
            )
    except sqlite3.IntegrityError:
        return _json_error("Ese nombre de usuario ya está registrado en este equipo.", 409)
    session.clear()
    session["user_id"] = user_id
    return jsonify({"username": username})


@app.post("/api/login")
def login():
    payload = request.get_json(silent=True) or {}
    username = str(payload.get("username", "")).strip()
    password = str(payload.get("password", ""))
    with _connect() as connection:
        user = connection.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    if user is None:
        return _json_error("Usuario o contraseña incorrectos.", 401)
    salt = bytes.fromhex(user["password_salt"])
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_ROUNDS)
    if not secrets.compare_digest(actual.hex(), user["password_hash"]):
        return _json_error("Usuario o contraseña incorrectos.", 401)
    session.clear()
    session["user_id"] = user["id"]
    return jsonify({"username": user["username"]})


@app.post("/api/logout")
def logout():
    session.clear()
    return jsonify({"ok": True})


@app.get("/api/me")
def me():
    user = _current_user()
    return jsonify({"username": user["username"]}) if user else _json_error("Inicia sesión.", 401)


@app.get("/api/pairing")
def pairing():
    _require_user()
    host = _lan_ip()
    port = request.host.rsplit(":", 1)[-1] if ":" in request.host else "8000"
    url = f"{request.scheme}://{host}:{port}/"
    image = qrcode.make(url)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return jsonify({"url": url, "qr_png": base64.b64encode(buffer.getvalue()).decode("ascii")})


@app.get("/api/sessions")
def list_sessions():
    user = _require_user()
    with _connect() as connection:
        rows = connection.execute(
            "SELECT id, participant, source_name, created_at, status, error FROM analyses WHERE user_id = ? ORDER BY created_at DESC",
            (user["id"],),
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        if item["status"] == "completed":
            analysis = _owned_analysis(item["id"], user["id"])
            report_path = Path(analysis["session_dir"]) / "sentadilla_resumen.json"
            if report_path.is_file():
                try:
                    item["summary"] = json.loads(report_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    item["summary"] = None
        result.append(item)
    return jsonify(result)


@app.post("/api/sessions")
def create_session():
    user = _require_user()
    video = request.files.get("video")
    if video is None or not video.filename:
        return _json_error("Selecciona o graba un video.", 400)
    extension = Path(video.filename).suffix.lower().lstrip(".")
    if extension not in ALLOWED_EXTENSIONS:
        return _json_error("Formato de video no compatible. Usa MP4, MOV, AVI, MKV o WEBM.", 400)
    participant = str(request.form.get("participant", user["username"])).strip()[:80] or user["username"]
    analysis_id = uuid.uuid4().hex
    session_dir = _user_session_dir(user["id"], analysis_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    original_name = secure_filename(video.filename) or f"video.{extension}"
    video_path = session_dir / f"video_original.{extension}"
    video.save(video_path)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _connect() as connection:
        connection.execute(
            "INSERT INTO analyses(id, user_id, participant, source_name, created_at, status, session_dir) VALUES (?, ?, ?, ?, ?, 'processing', ?)",
            (analysis_id, user["id"], participant, original_name, now, str(session_dir)),
        )
    worker = threading.Thread(
        target=_run_analysis,
        args=(analysis_id, user["id"], video_path, session_dir, participant),
        daemon=True,
    )
    worker.start()
    return jsonify({"id": analysis_id, "status": "processing"}), 202


@app.get("/api/sessions/<analysis_id>")
def get_session(analysis_id: str):
    user = _require_user()
    record = _owned_analysis(analysis_id, user["id"])
    if record is None:
        abort(404)
    result = dict(record)
    directory = Path(record["session_dir"])
    report_path = directory / "sentadilla_resumen.json"
    csv_path = directory / "sentadilla_metricas.csv"
    result["report"] = None
    result["rows"] = []
    if record["status"] == "completed" and report_path.is_file():
        result["report"] = json.loads(report_path.read_text(encoding="utf-8"))
        if csv_path.is_file():
            with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
                result["rows"] = list(csv.DictReader(file))
    return jsonify(result)


@app.get("/api/sessions/<analysis_id>/video")
def session_video(analysis_id: str):
    user = _require_user()
    record = _owned_analysis(analysis_id, user["id"])
    if record is None:
        abort(404)
    path = Path(record["session_dir"]) / "sentadilla_anotada.mp4"
    if record["status"] != "completed" or not path.is_file():
        abort(404)
    return send_file(path, mimetype="video/mp4", conditional=True)


@app.get("/api/sessions/<analysis_id>/download/<kind>")
def download_session_file(analysis_id: str, kind: str):
    user = _require_user()
    record = _owned_analysis(analysis_id, user["id"])
    if record is None:
        abort(404)
    files = {
        "video": ("sentadilla_anotada.mp4", "video/mp4"),
        "csv": ("sentadilla_metricas.csv", "text/csv; charset=utf-8"),
        "json": ("sentadilla_resumen.json", "application/json"),
    }
    if record["status"] != "completed" or kind not in files:
        abort(404)
    filename, mimetype = files[kind]
    path = Path(record["session_dir"]) / filename
    if not path.is_file():
        abort(404)
    return send_file(path, mimetype=mimetype, as_attachment=True, download_name=filename, conditional=True)


@app.errorhandler(413)
def file_too_large(_error):
    return _json_error("El archivo excede el límite de 250 MB.", 413)


def _lan_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("192.0.2.1", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0", help="Interfaz de red; 0.0.0.0 permite conexiones en la red local")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--cert-file", type=Path, help="Certificado TLS para servir HTTPS (necesario para PWA/cámara en Android)")
    parser.add_argument("--key-file", type=Path, help="Clave privada TLS correspondiente al certificado")
    args = parser.parse_args()
    if bool(args.cert_file) != bool(args.key_file):
        parser.error("--cert-file y --key-file deben usarse juntos")
    ip = _lan_ip()
    scheme = "https" if args.cert_file else "http"
    ssl_context = (str(args.cert_file), str(args.key_file)) if args.cert_file else None
    print(f"App local: {scheme}://127.0.0.1:{args.port}")
    print(f"Acceso desde el teléfono en la misma Wi-Fi: {scheme}://{ip}:{args.port}")
    print(f"Cuentas y sesiones se guardan localmente en: {DATA_DIR}")
    if args.cert_file:
        os.environ["YOLO_HTTPS"] = "1"
        app.config["SESSION_COOKIE_SECURE"] = True
    app.run(host=args.host, port=args.port, debug=False, threaded=True, ssl_context=ssl_context)


if __name__ == "__main__":
    main()
