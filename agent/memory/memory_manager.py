"""
memory_manager.py — SQLite WAL-mode persistent memory for partner-decode-female-agent.

5 tables: decode_sessions, communication_patterns, partner_profile,
llm_cost_log, knowledge_hashes. Thread-safe with threading.Lock.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, date
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "./data/partner_decode.db"


class MemoryManager:
    """
    Thread-safe SQLite-backed persistent memory for session history,
    communication pattern tracking, partner profile, LLM cost logging,
    and knowledge deduplication.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create all tables if they don't exist. Enable WAL mode."""
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.cursor()
                # Enable WAL mode for concurrent read/write
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA foreign_keys=ON")

                # Table 1: decode_sessions — one row per analysis session
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS decode_sessions (
                        session_id          TEXT PRIMARY KEY,
                        timestamp           TEXT NOT NULL,
                        audio_path          TEXT,
                        text_input          TEXT,
                        has_video           INTEGER DEFAULT 0,
                        gottman_criticism   REAL DEFAULT 0.0,
                        gottman_contempt    REAL DEFAULT 0.0,
                        gottman_defensiveness REAL DEFAULT 0.0,
                        gottman_stonewalling  REAL DEFAULT 0.0,
                        attachment_pattern  TEXT,
                        love_language_top   TEXT,
                        distress_score      REAL DEFAULT 0.0,
                        counseling_flagged  INTEGER DEFAULT 0,
                        urgency_level       TEXT,
                        llm_provider        TEXT,
                        latency_ms          REAL DEFAULT 0.0,
                        cost_usd            REAL DEFAULT 0.0,
                        fallback_used       INTEGER DEFAULT 0,
                        raw_result_json     TEXT
                    )
                """)

                # Table 2: communication_patterns — daily UPSERT with running averages
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS communication_patterns (
                        date                    TEXT PRIMARY KEY,
                        gottman_avg_score       REAL DEFAULT 0.0,
                        attachment_pattern_freq TEXT DEFAULT '{}',
                        love_language_top       TEXT,
                        counseling_flag_count   INTEGER DEFAULT 0,
                        session_count           INTEGER DEFAULT 0,
                        avg_distress_score      REAL DEFAULT 0.0
                    )
                """)

                # Table 3: partner_profile — persistent behavioral profile
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS partner_profile (
                        profile_id              TEXT PRIMARY KEY DEFAULT 'default',
                        name                    TEXT DEFAULT 'Partner',
                        created_at              TEXT NOT NULL,
                        updated_at              TEXT NOT NULL,
                        dominant_attachment_style TEXT DEFAULT 'unknown',
                        primary_love_language   TEXT DEFAULT 'unknown',
                        avg_distress_score      REAL DEFAULT 0.0,
                        total_sessions          INTEGER DEFAULT 0,
                        notes                   TEXT DEFAULT ''
                    )
                """)

                # Table 4: llm_cost_log — per-call cost tracking
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS llm_cost_log (
                        id          INTEGER PRIMARY KEY AUTOINCREMENT,
                        provider    TEXT NOT NULL,
                        model       TEXT NOT NULL,
                        tokens_in   INTEGER DEFAULT 0,
                        tokens_out  INTEGER DEFAULT 0,
                        cost_usd    REAL DEFAULT 0.0,
                        timestamp   TEXT NOT NULL,
                        session_id  TEXT
                    )
                """)

                # Table 5: knowledge_hashes — SHA256 dedup for crawled papers
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS knowledge_hashes (
                        url_hash    TEXT PRIMARY KEY,
                        source_url  TEXT,
                        added_at    TEXT NOT NULL
                    )
                """)

                conn.commit()
                logger.info("MemoryManager initialized: %s", self.db_path)
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Session Operations
    # ------------------------------------------------------------------

    def save_session(
        self,
        session_id: str,
        audio_path: Optional[str] = None,
        text_input: Optional[str] = None,
        has_video: bool = False,
        classification_result: Optional[Any] = None,
        interpretation_result: Optional[Any] = None,
        latency_ms: float = 0.0,
        cost_usd: float = 0.0,
    ) -> None:
        """Persist a complete analysis session to the database."""
        timestamp = datetime.utcnow().isoformat()

        horsemen = {}
        attachment_pattern = "unknown"
        love_language_top = "unknown"
        distress_score = 0.0
        counseling_flagged = False
        urgency_level = "low"
        llm_provider = "unknown"
        fallback_used = False
        raw_result = {}

        if classification_result is not None:
            horsemen = classification_result.gottman_horsemen
            attachment_pattern = classification_result.attachment_pattern
            love_lang_scores = classification_result.love_language_signals
            if love_lang_scores:
                love_language_top = max(love_lang_scores, key=love_lang_scores.get)
            distress_score = classification_result.overall_distress_score
            counseling_flagged = classification_result.needs_counseling_flag

        if interpretation_result is not None:
            urgency_level = interpretation_result.urgency_level
            llm_provider = interpretation_result.llm_provider_used
            fallback_used = interpretation_result.fallback_used
            raw_result = {
                "actual_feeling": interpretation_result.actual_feeling,
                "urgency_level": interpretation_result.urgency_level,
                "counseling_recommended": interpretation_result.counseling_recommended,
            }

        with self._lock:
            conn = self._connect()
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO decode_sessions
                    (session_id, timestamp, audio_path, text_input, has_video,
                     gottman_criticism, gottman_contempt, gottman_defensiveness, gottman_stonewalling,
                     attachment_pattern, love_language_top, distress_score, counseling_flagged,
                     urgency_level, llm_provider, latency_ms, cost_usd, fallback_used, raw_result_json)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    session_id,
                    timestamp,
                    audio_path,
                    text_input,
                    int(has_video),
                    horsemen.get("criticism", 0.0),
                    horsemen.get("contempt", 0.0),
                    horsemen.get("defensiveness", 0.0),
                    horsemen.get("stonewalling", 0.0),
                    attachment_pattern,
                    love_language_top,
                    distress_score,
                    int(counseling_flagged),
                    urgency_level,
                    llm_provider,
                    latency_ms,
                    cost_usd,
                    int(fallback_used),
                    json.dumps(raw_result),
                ))
                conn.commit()
            finally:
                conn.close()

        # Update daily communication patterns
        self._upsert_daily_patterns(
            attachment_pattern=attachment_pattern,
            love_language_top=love_language_top,
            counseling_flagged=counseling_flagged,
            distress_score=distress_score,
        )

    def get_recent_sessions(self, n: int = 10) -> List[Dict]:
        """Return the most recent N sessions."""
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    "SELECT * FROM decode_sessions ORDER BY timestamp DESC LIMIT ?", (n,)
                )
                cols = [desc[0] for desc in cursor.description]
                return [dict(zip(cols, row)) for row in cursor.fetchall()]
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Communication Patterns (Daily UPSERT)
    # ------------------------------------------------------------------

    def _upsert_daily_patterns(
        self,
        attachment_pattern: str,
        love_language_top: str,
        counseling_flagged: bool,
        distress_score: float,
    ) -> None:
        """Update or insert today's communication pattern summary."""
        today = date.today().isoformat()
        with self._lock:
            conn = self._connect()
            try:
                # Fetch existing row
                row = conn.execute(
                    "SELECT * FROM communication_patterns WHERE date = ?", (today,)
                ).fetchone()

                if row is None:
                    # Insert new row
                    freq = {attachment_pattern: 1}
                    conn.execute("""
                        INSERT INTO communication_patterns
                        (date, gottman_avg_score, attachment_pattern_freq,
                         love_language_top, counseling_flag_count, session_count, avg_distress_score)
                        VALUES (?,?,?,?,?,?,?)
                    """, (
                        today,
                        distress_score,
                        json.dumps(freq),
                        love_language_top,
                        int(counseling_flagged),
                        1,
                        distress_score,
                    ))
                else:
                    # Update running averages
                    cols = [desc[0] for desc in conn.execute(
                        "SELECT * FROM communication_patterns WHERE date = ?", (today,)
                    ).description]
                    row_dict = dict(zip(cols, conn.execute(
                        "SELECT * FROM communication_patterns WHERE date = ?", (today,)
                    ).fetchone()))

                    n = row_dict["session_count"] + 1
                    new_avg_distress = (
                        row_dict["avg_distress_score"] * row_dict["session_count"] + distress_score
                    ) / n

                    freq = json.loads(row_dict.get("attachment_pattern_freq", "{}"))
                    freq[attachment_pattern] = freq.get(attachment_pattern, 0) + 1

                    conn.execute("""
                        UPDATE communication_patterns
                        SET session_count = ?,
                            avg_distress_score = ?,
                            attachment_pattern_freq = ?,
                            love_language_top = ?,
                            counseling_flag_count = counseling_flag_count + ?
                        WHERE date = ?
                    """, (
                        n,
                        new_avg_distress,
                        json.dumps(freq),
                        love_language_top,
                        int(counseling_flagged),
                        today,
                    ))
                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Partner Profile
    # ------------------------------------------------------------------

    def get_partner_profile(self) -> Dict:
        """Return the current partner profile."""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM partner_profile WHERE profile_id = 'default'"
                ).fetchone()
                if row is None:
                    return {}
                cols = [desc[0] for desc in conn.execute(
                    "SELECT * FROM partner_profile WHERE profile_id = 'default'"
                ).description]
                return dict(zip(cols, row))
            finally:
                conn.close()

    def update_partner_profile(
        self,
        name: Optional[str] = None,
        attachment_style: Optional[str] = None,
        love_language: Optional[str] = None,
        distress_score: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> None:
        """Create or incrementally update the partner profile."""
        now = datetime.utcnow().isoformat()
        with self._lock:
            conn = self._connect()
            try:
                existing = conn.execute(
                    "SELECT * FROM partner_profile WHERE profile_id = 'default'"
                ).fetchone()

                if existing is None:
                    conn.execute("""
                        INSERT INTO partner_profile
                        (profile_id, name, created_at, updated_at,
                         dominant_attachment_style, primary_love_language,
                         avg_distress_score, total_sessions, notes)
                        VALUES ('default',?,?,?,?,?,?,1,?)
                    """, (
                        name or "Partner",
                        now,
                        now,
                        attachment_style or "unknown",
                        love_language or "unknown",
                        distress_score or 0.0,
                        notes or "",
                    ))
                else:
                    updates = ["updated_at = ?"]
                    values = [now]
                    if name is not None:
                        updates.append("name = ?")
                        values.append(name)
                    if attachment_style is not None:
                        updates.append("dominant_attachment_style = ?")
                        values.append(attachment_style)
                    if love_language is not None:
                        updates.append("primary_love_language = ?")
                        values.append(love_language)
                    if distress_score is not None:
                        updates.append("avg_distress_score = (avg_distress_score * total_sessions + ?) / (total_sessions + 1)")
                        values.append(distress_score)
                    if notes is not None:
                        updates.append("notes = ?")
                        values.append(notes)
                    updates.append("total_sessions = total_sessions + 1")
                    values.append("default")

                    conn.execute(
                        f"UPDATE partner_profile SET {', '.join(updates)} WHERE profile_id = ?",
                        values,
                    )
                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # LLM Cost Logging
    # ------------------------------------------------------------------

    def log_llm_cost(
        self,
        provider: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
        session_id: Optional[str] = None,
    ) -> None:
        """Log a single LLM API call cost."""
        timestamp = datetime.utcnow().isoformat()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("""
                    INSERT INTO llm_cost_log
                    (provider, model, tokens_in, tokens_out, cost_usd, timestamp, session_id)
                    VALUES (?,?,?,?,?,?,?)
                """, (provider, model, tokens_in, tokens_out, cost_usd, timestamp, session_id))
                conn.commit()
            finally:
                conn.close()

    def get_cost_report(self) -> Dict:
        """Return aggregated cost breakdown by provider and model."""
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute("""
                    SELECT provider, model,
                           COUNT(*) as calls,
                           SUM(tokens_in) as total_tokens_in,
                           SUM(tokens_out) as total_tokens_out,
                           SUM(cost_usd) as total_cost_usd
                    FROM llm_cost_log
                    GROUP BY provider, model
                    ORDER BY total_cost_usd DESC
                """).fetchall()
                return {
                    "by_provider_model": [
                        {
                            "provider": row[0],
                            "model": row[1],
                            "calls": row[2],
                            "total_tokens_in": row[3],
                            "total_tokens_out": row[4],
                            "total_cost_usd": round(row[5], 6),
                        }
                        for row in rows
                    ],
                    "total_usd": round(
                        sum(row[5] for row in rows), 6
                    ) if rows else 0.0,
                }
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Knowledge Hash Deduplication
    # ------------------------------------------------------------------

    def add_knowledge_hash(self, url: str) -> None:
        """Store SHA256 hash of a URL to prevent re-crawling."""
        url_hash = hashlib.sha256(url.encode()).hexdigest()
        added_at = datetime.utcnow().isoformat()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO knowledge_hashes (url_hash, source_url, added_at) VALUES (?,?,?)",
                    (url_hash, url, added_at),
                )
                conn.commit()
            finally:
                conn.close()

    def has_knowledge_hash(self, url: str) -> bool:
        """Return True if this URL has already been crawled."""
        url_hash = hashlib.sha256(url.encode()).hexdigest()
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT 1 FROM knowledge_hashes WHERE url_hash = ?", (url_hash,)
                ).fetchone()
                return row is not None
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        """Return overall statistics across all sessions."""
        with self._lock:
            conn = self._connect()
            try:
                total_sessions = conn.execute(
                    "SELECT COUNT(*) FROM decode_sessions"
                ).fetchone()[0]
                avg_distress = conn.execute(
                    "SELECT AVG(distress_score) FROM decode_sessions"
                ).fetchone()[0] or 0.0
                counseling_flags = conn.execute(
                    "SELECT COUNT(*) FROM decode_sessions WHERE counseling_flagged = 1"
                ).fetchone()[0]
                most_common_attachment = conn.execute(
                    "SELECT attachment_pattern, COUNT(*) as cnt FROM decode_sessions "
                    "GROUP BY attachment_pattern ORDER BY cnt DESC LIMIT 1"
                ).fetchone()
                return {
                    "total_sessions": total_sessions,
                    "avg_distress_score": round(avg_distress, 3),
                    "counseling_flag_count": counseling_flags,
                    "most_common_attachment": most_common_attachment[0] if most_common_attachment else "unknown",
                }
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Open a SQLite connection with WAL mode."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn
