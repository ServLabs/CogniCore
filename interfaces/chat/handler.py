"""
Chat WebSocket Handler

Manages WebSocket connection lifecycle and message processing.
Accepts connections, dispatches message types, and bridges
the response pipeline to the client via streaming events.
"""

import asyncio
import json
import time
import uuid
from typing import Any, Optional

from fastapi import WebSocket, WebSocketDisconnect

from logger import log
from observability import audit
from config import config
from interfaces.chat.streaming import EventStream
from interfaces.chat.websocket import ConnectionManager, Session
from response import StreamChannel
from memory import add_user_message
from control import get_salience_network, UserMessageEvent


# ══════════════════════════════════════════════════════════════════════════════
# WebSocket Entry Point
# ══════════════════════════════════════════════════════════════════════════════

async def handle_websocket(
    websocket: WebSocket,
    manager: ConnectionManager,
    user_id: Optional[str],
    conversation_id: Optional[str],
) -> None:
    """
    Handle a full WebSocket connection lifecycle.

    Rejects immediately if user_id is missing.
    Creates or reuses conversation_id.
    Enters receive loop until disconnect.
    """
    # Reject anonymous connections
    if not user_id:
        await websocket.close(code=1008, reason="user_id is required")
        audit.log_raw(
            "interface", "connection_rejected", "chat", "failed",
            error="user_id is required",
        )
        return

    session = await manager.connect(websocket, user_id)
    session.conversation_id = conversation_id or str(uuid.uuid4())

    try:
        await websocket.send_json({
            "type": "connected",
            "session_id": session.session_id,
            "conversation_id": session.conversation_id,
        })

        while True:
            data = await websocket.receive_text()
            # Reject oversized messages (default 64KB)
            max_size = getattr(config, "max_ws_message_bytes", 65536)
            if len(data) > max_size:
                await websocket.send_json({
                    "type": "error",
                    "error": f"Message too large (max {max_size} bytes)",
                })
                continue
            await _dispatch(session, data)

    except WebSocketDisconnect:
        manager.disconnect(session.session_id)
    except Exception as e:
        log.error(f"WebSocket error: {e}", exc_info=True)
        manager.disconnect(session.session_id)


# ══════════════════════════════════════════════════════════════════════════════
# Message Dispatch
# ══════════════════════════════════════════════════════════════════════════════

async def _dispatch(session: Session, raw: str) -> None:
    """Parse a raw WebSocket frame and route by message type."""
    try:
        message = json.loads(raw)
    except json.JSONDecodeError:
        await _send_error(session, "Invalid JSON")
        return

    session.touch()
    msg_type = message.get("type", "message")

    match msg_type:
        case "message":
            await _on_user_message(session, message)
        case "reply":
            _on_reply(session, message)
        case "cancel":
            await _on_cancel(session)
        case "ping":
            if session.websocket:
                await session.websocket.send_json({"type": "pong"})
        case "set_conversation":
            session.conversation_id = message.get("conversation_id") or str(uuid.uuid4())
            if session.websocket:
                await session.websocket.send_json({
                    "type": "conversation_set",
                    "conversation_id": session.conversation_id,
                })
        case _:
            await _send_error(session, f"Unknown message type: {msg_type}")


# ══════════════════════════════════════════════════════════════════════════════
# Message Processing
# ══════════════════════════════════════════════════════════════════════════════

async def _on_user_message(session: Session, message: dict[str, Any]) -> None:
    """Validate, stream, and process a user chat message."""
    text = message.get("text", "")
    if not text:
        await _send_error(session, "Empty message")
        return

    audit.log_raw(
        "interface", "message_received", "chat", "started",
        session_id=session.session_id, details={"length": len(text)},
    )

    stream = session.create_stream()
    stream_task = asyncio.create_task(_forward_stream(session, stream))

    try:
        await _process(session, text, stream)
    except Exception as e:
        log.error(f"Message processing error: {e}", exc_info=True)
        stream.emit_error(str(e), recoverable=False)
        stream.emit_done({"error": str(e)})
    finally:
        await stream_task


async def _process(session: Session, text: str, stream: EventStream) -> None:
    """Route a user message through CEN → Salience → Worker → Pipeline."""
    start_time = time.monotonic()

    try:
        await add_user_message(session.user_id, session.conversation_id, text)

        # Create a StreamChannel for two-way pipeline communication
        channel = StreamChannel(stream)
        session.channel = channel  # type: ignore[attr-defined]

        # Build typed event and submit to Salience → CEN worker pool
        event = UserMessageEvent(
            message=text,
            user_id=session.user_id,
            convo_id=session.conversation_id,
            channel=channel,
        )

        salience = get_salience_network()
        await salience.ingest(event)

        # The CEN worker pool picks up this event and runs pipeline.process().
        # Pipeline emits status/response/done events through the StreamChannel,
        # which are forwarded to the WebSocket by _forward_stream.
        # We just wait for the stream to complete.
        await stream.wait_done()

        session.channel = None  # type: ignore[attr-defined]

        elapsed_ms = (time.monotonic() - start_time) * 1000

        audit.log_raw(
            "interface", "message_processed", "chat", "completed",
            session_id=session.session_id, duration_ms=elapsed_ms,
        )

    except Exception as e:
        stream.emit_error(str(e), recoverable=False)
        stream.emit_done({"error": str(e)})
        raise


# ══════════════════════════════════════════════════════════════════════════════
# Clarification Reply
# ══════════════════════════════════════════════════════════════════════════════

def _on_reply(session: Session, message: dict[str, Any]) -> None:
    """Route a user reply to the waiting pipeline channel."""
    text = message.get("text", "")
    channel: Optional[StreamChannel] = getattr(session, "channel", None)

    if channel and channel.waiting_for_reply:
        channel.provide_reply(text)


# ══════════════════════════════════════════════════════════════════════════════
# Stream Forwarding
# ══════════════════════════════════════════════════════════════════════════════

async def _forward_stream(session: Session, stream: EventStream) -> None:
    """Forward EventStream events to the WebSocket client."""
    if not session.websocket:
        return

    async for event in stream:
        try:
            await session.websocket.send_text(event.to_json())
        except Exception:
            break


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

async def _on_cancel(session: Session) -> None:
    """Cancel the in-flight request for this session."""
    session.close_stream()

    if session.websocket:
        await session.websocket.send_json({"type": "cancelled", "text": "Request cancelled"})

    audit.log_raw(
        "interface", "request_cancel", "chat", "completed",
        session_id=session.session_id,
    )


async def _send_error(session: Session, error: str) -> None:
    """Send an error payload to the client."""
    if session.websocket:
        await session.websocket.send_json({"type": "error", "text": error})
