from collections import deque
from threading import Lock
from typing import Optional, Union

from datetime import timedelta
from fastapi import FastAPI, HTTPException, Header, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
import os
from uuid import uuid4
from datetime import datetime, timezone
from google.cloud import storage
import google.auth
from google.auth.transport.requests import Request as GoogleAuthRequest
import firebase_admin
from firebase_admin import auth as firebase_auth
from main import run_simulation

app = FastAPI()
_ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
_ALLOW_ALL_ORIGINS = "*" in _ALLOWED_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS if not _ALLOW_ALL_ORIGINS else ["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

_RATE_LIMIT_LOCK = Lock()
_RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
_RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "6"))
_RATE_LIMIT_BUCKETS: dict[str, deque] = {}

_REQUIRE_FIREBASE_AUTH = os.getenv("REQUIRE_FIREBASE_AUTH", "false").lower() == "true"
if not firebase_admin._apps:
    firebase_admin.initialize_app()


@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    origin = request.headers.get("origin", "")
    allow_origin = "*" if _ALLOW_ALL_ORIGINS else (origin if origin in _ALLOWED_ORIGINS else "")
    # Explicitly handle preflight for browser clients.
    if request.method == "OPTIONS":
        headers = {}
        if allow_origin:
            headers["Access-Control-Allow-Origin"] = allow_origin
            headers["Vary"] = "Origin"
        headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        headers["Access-Control-Allow-Headers"] = request.headers.get(
            "access-control-request-headers", "Authorization,Content-Type"
        )
        headers["Access-Control-Max-Age"] = "3600"
        return Response(status_code=204, headers=headers)

    response = await call_next(request)
    if allow_origin:
        response.headers["Access-Control-Allow-Origin"] = allow_origin
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Authorization,Content-Type"
    return response

class RunRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    map: str
    out: str = "evac.mp4"
    fire: Optional[Union[str, list[int], tuple[int, int]]] = None
    fire_r: Optional[int] = None
    fire_c: Optional[int] = None
    steps: int = Field(default=400, ge=20, le=2000)
    people: int = Field(default=200, ge=1, le=2000)
    grid: int = Field(default=140, ge=40, le=400)
    fps: int = Field(default=15, ge=1, le=60)
    render_every: int = Field(default=1, ge=1, le=20)
    alarm_at: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("alarm_at", "alarm-at"),
    )
    awareness_radius: int = Field(
        default=6,
        validation_alias=AliasChoices("awareness_radius", "awareness-radius"),
        ge=1,
        le=50,
    )
    bucket: Optional[str] = None

@app.get("/health")
def health():
    return {"ok": True, "auth_required": _REQUIRE_FIREBASE_AUTH}

@app.post("/run")
def run(req: RunRequest, request: Request, authorization: Optional[str] = Header(default=None)):
    try:
        enforce_rate_limit(request)
        user_uid = verify_auth_if_required(authorization)

        base_dir = os.path.dirname(__file__)
        map_path = resolve_map_path(base_dir, req.map)
        # In Cloud Run, write generated files to /tmp (writable ephemeral storage).
        out_name = os.path.basename(req.out)
        if not out_name.lower().endswith(".mp4"):
            out_name = f"{out_name}.mp4"
        out_path = os.path.join("/tmp", out_name)

        fire_arg = None
        if isinstance(req.fire, str):
            if req.fire.lower() == "random":
                fire_arg = "random"
        elif isinstance(req.fire, (list, tuple)) and len(req.fire) == 2:
            fire_arg = (int(req.fire[0]), int(req.fire[1]))
        elif req.fire_r is not None and req.fire_c is not None:
            fire_arg = (int(req.fire_r), int(req.fire_c))
        else:
            fire_arg = (70, 70)

        result = run_simulation(
            map_path,
            out_path,
            fire=fire_arg,
            steps=req.steps,
            people=req.people,
            grid=req.grid,
            fps=req.fps,
            render_every=req.render_every,
            alarm_at=req.alarm_at,
            awareness_radius=req.awareness_radius,
        )
        bucket_name = req.bucket or os.getenv("OUTPUT_BUCKET")
        upload = upload_to_gcs(result["out"], bucket_name) if bucket_name else None
        return {
            "ok": True,
            "uid": user_uid,
            "applied_alarm_at": req.alarm_at,
            "applied_awareness_radius": req.awareness_radius,
            "bucket_used": bucket_name,
            "uploaded": bool(upload),
            "storage": upload,
            **result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def upload_to_gcs(local_path: str, bucket_name: str):
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Output file not found for upload: {local_path}")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    object_name = f"runs/{timestamp}-{uuid4().hex[:8]}-{os.path.basename(local_path)}"
    blob = bucket.blob(object_name)
    blob.upload_from_filename(local_path, content_type="video/mp4")

    signed_url = None
    signing_error = None
    try:
        creds, _ = google.auth.default()
        auth_req = GoogleAuthRequest()
        creds.refresh(auth_req)
        sa_email = getattr(creds, "service_account_email", None)
        token = getattr(creds, "token", None)
        if sa_email and token:
            signed_url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(hours=6),
                method="GET",
                service_account_email=sa_email,
                access_token=token,
            )
    except Exception as e:
        signing_error = str(e)

    return {
        "bucket": bucket_name,
        "object": object_name,
        "gs_uri": f"gs://{bucket_name}/{object_name}",
        "https_url": signed_url or f"https://storage.googleapis.com/{bucket_name}/{object_name}",
        "signed_url": signed_url,
        "signed_url_expires_hours": 6 if signed_url else None,
        "signed_url_error": signing_error,
        "console_url": f"https://console.cloud.google.com/storage/browser/_details/{bucket_name}/{object_name}",
    }


def enforce_rate_limit(request: Request):
    # Hackathon-grade in-memory limiter per IP.
    client_ip = request.client.host if request.client else "unknown"
    now = datetime.now(timezone.utc).timestamp()
    cutoff = now - _RATE_LIMIT_WINDOW_SECONDS
    with _RATE_LIMIT_LOCK:
        q = _RATE_LIMIT_BUCKETS.setdefault(client_ip, deque())
        while q and q[0] < cutoff:
            q.popleft()
        if len(q) >= _RATE_LIMIT_MAX_REQUESTS:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: max {_RATE_LIMIT_MAX_REQUESTS} requests per {_RATE_LIMIT_WINDOW_SECONDS}s.",
            )
        q.append(now)


def verify_auth_if_required(authorization: Optional[str]) -> Optional[str]:
    if not _REQUIRE_FIREBASE_AUTH:
        return None
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token.")
    token = authorization.split(" ", 1)[1].strip()
    try:
        decoded = firebase_auth.verify_id_token(token)
        return decoded.get("uid")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid Firebase ID token.")


def resolve_map_path(base_dir: str, map_input: str) -> str:
    # Restrict to local map files in project root to avoid path traversal.
    map_name = os.path.basename(map_input)
    map_path = os.path.join(base_dir, map_name)
    if not os.path.exists(map_path):
        raise HTTPException(status_code=400, detail=f"Map file not found: {map_name}")
    return map_path
