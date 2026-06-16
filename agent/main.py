"""
main.py — Entry point for partner-decode-female-agent.

Provides:
  1. Click CLI: analyze / listen / decode / session / history / update-knowledge / cost-report / serve
  2. FastAPI REST server with all endpoints
  3. Prometheus metrics endpoint
"""
from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("partner_decode")

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config() -> dict:
    """Load agent_config.yaml if present, else return defaults."""
    try:
        import yaml
        config_path = Path(__file__).parent.parent / "config" / "agent_config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f) or {}
    except Exception as exc:
        logger.warning("Could not load config: %s — using defaults", exc)
    return {}


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class TextDecodeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text message to decode")
    session_id: Optional[str] = Field(None, description="Optional session ID for tracking")


class MicrophoneRequest(BaseModel):
    duration_seconds: int = Field(default=30, ge=3, le=300, description="Recording duration")
    session_id: Optional[str] = None


class SessionRequest(BaseModel):
    text: Optional[str] = Field(None, description="Text transcript (optional if audio provided)")
    session_id: Optional[str] = None


class KnowledgeUpdateResponse(BaseModel):
    status: str
    new_papers: int = 0
    message: str = ""


class HealthResponse(BaseModel):
    status: str
    version: str
    llm_providers: list
    visual_available: bool


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Partner Decode — Female Agent",
    description=(
        "Multimodal emotion and communication decoder for female partner communication. "
        "Powered by Gottman, Attachment Theory, and Five Love Languages frameworks."
    ),
    version="1.0.0",
)

_orchestrator = None


def get_orchestrator():
    """Lazy-initialize the orchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        from agent.orchestrator import PartnerDecodeOrchestrator
        config = load_config()
        _orchestrator = PartnerDecodeOrchestrator(config=config)
    return _orchestrator


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    from agent.modules.visual_analyzer import VisualAnalyzer
    visual_ok = VisualAnalyzer().is_available()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        llm_providers=["claude", "openai", "ollama"],
        visual_available=visual_ok,
    )


@app.post("/api/v1/analyze/audio")
async def analyze_audio(
    audio_file: UploadFile = File(..., description="Audio file (WAV/MP3/FLAC)"),
    session_id: Optional[str] = Form(None),
):
    """Upload an audio file and receive a full empathy report."""
    import tempfile
    session_id = session_id or str(uuid.uuid4())
    suffix = Path(audio_file.filename or "audio.wav").suffix or ".wav"
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await audio_file.read()
            tmp.write(content)
            tmp_path = tmp.name

        orchestrator = get_orchestrator()
        result = orchestrator.analyze_audio(audio_path=tmp_path, session_id=session_id)
        return JSONResponse(content=result)
    except Exception as exc:
        logger.error("analyze_audio endpoint error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@app.post("/api/v1/analyze/microphone")
async def analyze_microphone(request: MicrophoneRequest):
    """Record from microphone and return empathy report."""
    try:
        orchestrator = get_orchestrator()
        result = orchestrator.analyze_microphone(
            duration_seconds=request.duration_seconds
        )
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/v1/analyze/text")
async def analyze_text(request: TextDecodeRequest):
    """Decode a text message: emotion + Gottman + attachment + love language."""
    try:
        orchestrator = get_orchestrator()
        result = orchestrator.analyze_text(
            text=request.text,
            session_id=request.session_id,
        )
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/v1/analyze/session")
async def analyze_session(
    audio_file: Optional[UploadFile] = File(None, description="Audio file (optional)"),
    video_file: Optional[UploadFile] = File(None, description="Video file (optional)"),
    text: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
):
    """Full multimodal session: audio + text + optional video."""
    import tempfile
    session_id = session_id or str(uuid.uuid4())
    audio_tmp = None
    video_tmp = None

    try:
        if audio_file:
            suffix = Path(audio_file.filename or "audio.wav").suffix or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(await audio_file.read())
                audio_tmp = tmp.name

        if video_file:
            suffix = Path(video_file.filename or "video.mp4").suffix or ".mp4"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(await video_file.read())
                video_tmp = tmp.name

        orchestrator = get_orchestrator()
        result = orchestrator.analyze_session(
            audio_path=audio_tmp,
            text=text,
            video_path=video_tmp,
            session_id=session_id,
        )
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        for tmp_path in [audio_tmp, video_tmp]:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass


@app.get("/api/v1/sessions")
async def get_recent_sessions(n: int = 10):
    """Return the N most recent analysis sessions."""
    try:
        memory = get_orchestrator()._get_memory()
        sessions = memory.get_recent_sessions(n=n)
        return JSONResponse(content={"sessions": sessions, "count": len(sessions)})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/v1/knowledge/update")
async def update_knowledge():
    """Trigger a manual knowledge base update."""
    try:
        result = get_orchestrator().update_knowledge()
        return JSONResponse(content=result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/v1/cost")
async def get_cost_report():
    """Return LLM API cost breakdown."""
    try:
        report = get_orchestrator().get_cost_report()
        return JSONResponse(content=report)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/v1/stats")
async def get_stats():
    """Return overall agent statistics."""
    try:
        stats = get_orchestrator().get_stats()
        return JSONResponse(content=stats)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    try:
        metrics = get_orchestrator().get_prometheus_metrics()
        return PlainTextResponse(content=metrics, media_type="text/plain; version=0.0.4")
    except Exception as exc:
        return PlainTextResponse(content=f"# Error: {exc}\n", status_code=500)


# ---------------------------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(version="1.0.0")
def cli():
    """partner-decode-female-agent — Multimodal partner communication decoder."""
    pass


@cli.command()
@click.argument("audio_path", type=click.Path(exists=True))
@click.option("--session-id", default=None, help="Session ID for tracking")
@click.option("--json-out", is_flag=True, default=False, help="Output raw JSON")
def analyze(audio_path: str, session_id: Optional[str], json_out: bool):
    """Analyze an audio file and print the empathy report."""
    orchestrator = get_orchestrator()
    result = orchestrator.analyze_audio(audio_path=audio_path, session_id=session_id)

    if json_out:
        click.echo(json.dumps(result, indent=2))
    else:
        formatted = result.get("formatted_report", "")
        if formatted:
            click.echo(formatted)
        else:
            click.echo(json.dumps(result.get("report", result), indent=2))

        if result.get("classification", {}).get("counseling_recommended"):
            click.secho(
                "\n⚠ Professional counseling recommended — see report above.",
                fg="yellow",
                bold=True,
            )


@cli.command()
@click.option("--duration", default=30, show_default=True, help="Recording duration in seconds")
@click.option("--session-id", default=None)
@click.option("--json-out", is_flag=True)
def listen(duration: int, session_id: Optional[str], json_out: bool):
    """Record from microphone and analyze."""
    click.echo(f"Recording for {duration} seconds... (press Ctrl+C to cancel)")
    orchestrator = get_orchestrator()
    result = orchestrator.analyze_microphone(duration_seconds=duration)

    if json_out:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(result.get("formatted_report", json.dumps(result.get("report", {}), indent=2)))


@cli.command()
@click.argument("text")
@click.option("--session-id", default=None)
@click.option("--json-out", is_flag=True)
def decode(text: str, session_id: Optional[str], json_out: bool):
    """Decode a text message and print the empathy report."""
    orchestrator = get_orchestrator()
    result = orchestrator.analyze_text(text=text, session_id=session_id)

    if json_out:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(result.get("formatted_report", json.dumps(result.get("report", {}), indent=2)))


@cli.command()
@click.option("--audio", type=click.Path(), default=None, help="Audio file path")
@click.option("--video", type=click.Path(), default=None, help="Video file path")
@click.option("--text", default=None, help="Text transcript")
@click.option("--session-id", default=None)
@click.option("--json-out", is_flag=True)
def session(
    audio: Optional[str],
    video: Optional[str],
    text: Optional[str],
    session_id: Optional[str],
    json_out: bool,
):
    """Run a full multimodal session (audio + text + optional video)."""
    if not audio and not text:
        click.echo("Error: provide at least --audio or --text", err=True)
        raise SystemExit(1)

    orchestrator = get_orchestrator()
    result = orchestrator.analyze_session(
        audio_path=audio,
        text=text,
        video_path=video,
        session_id=session_id,
    )

    if json_out:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(result.get("formatted_report", json.dumps(result.get("report", {}), indent=2)))


@cli.command()
@click.option("--n", default=10, show_default=True, help="Number of recent sessions to show")
def history(n: int):
    """Display recent session history."""
    memory = get_orchestrator()._get_memory()
    sessions = memory.get_recent_sessions(n=n)
    if not sessions:
        click.echo("No sessions found.")
        return
    for s in sessions:
        click.echo(
            f"[{s.get('timestamp', '')[:19]}] "
            f"session={s.get('session_id', '')[:8]}... "
            f"distress={s.get('distress_score', 0.0):.2f} "
            f"attachment={s.get('attachment_pattern', '?')} "
            f"counseling={'YES' if s.get('counseling_flagged') else 'no'}"
        )


@cli.command("update-knowledge")
def update_knowledge_cmd():
    """Trigger a manual knowledge base update from ArXiv + Semantic Scholar."""
    click.echo("Updating knowledge base...")
    result = get_orchestrator().update_knowledge()
    if result.get("status") == "success":
        click.secho(f"Knowledge updated: {result.get('new_papers', 0)} new papers added.", fg="green")
    else:
        click.secho(f"Update failed: {result.get('error', 'unknown error')}", fg="red")


@cli.command("cost-report")
def cost_report_cmd():
    """Display LLM API cost breakdown."""
    report = get_orchestrator().get_cost_report()
    click.echo(f"\nTotal LLM cost: ${report.get('total_usd', 0.0):.6f}")
    click.echo("\nBreakdown:")
    for entry in report.get("by_provider_model", []):
        click.echo(
            f"  {entry['provider']}/{entry['model']}: "
            f"{entry['calls']} calls, "
            f"${entry['total_cost_usd']:.6f}"
        )


@cli.command()
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8011, show_default=True)
@click.option("--workers", default=2, show_default=True)
@click.option("--reload", is_flag=True, default=False)
def serve(host: str, port: int, workers: int, reload: bool):
    """Start the FastAPI REST server."""
    click.echo(f"Starting partner-decode-female-agent server on {host}:{port}")
    # Initialize scheduler on startup
    get_orchestrator().start_scheduler()
    uvicorn.run(
        "agent.main:app",
        host=host,
        port=port,
        workers=workers if not reload else 1,
        reload=reload,
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
    )


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
