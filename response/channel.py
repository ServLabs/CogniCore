"""
Pipeline Channel

Two-way communication bridge between the response pipeline and the handler.
Allows the pipeline to ask clarifying questions and receive answers
without knowing about websockets or transport.
"""

import asyncio
from typing import Protocol


class PipelineChannel(Protocol):
    """
    Protocol for pipeline ↔ handler communication.

    The pipeline calls ask_user() when it needs clarification.
    The handler implementation wires this to the actual transport (websocket).
    """

    async def ask_user(self, questions: list[str]) -> str:
        """
        Send clarifying questions to the user and wait for a reply.

        Args:
            questions: List of questions to present to the user.

        Returns:
            The user's reply text.
        """
        ...

    async def send_status(self, stage: str, text: str) -> None:
        """
        Send a status update to the user (non-blocking, informational).

        Args:
            stage: Pipeline stage name.
            text: Status message.
        """
        ...


class StreamChannel:
    """
    Concrete PipelineChannel implementation backed by EventStream + asyncio.Future.

    Used by the handler to bridge the pipeline's ask_user() calls
    to the websocket client.
    """

    def __init__(self, stream):
        """
        Args:
            stream: EventStream instance for emitting events.
        """
        self.stream = stream
        self._reply_future: asyncio.Future | None = None

    async def ask_user(self, questions: list[str]) -> str:
        """Emit a clarification event and suspend until reply arrives."""
        loop = asyncio.get_running_loop()
        self._reply_future = loop.create_future()

        self.stream.emit(
            "clarification",
            "\n".join(questions),
            stage="decision",
            questions=questions,
        )

        reply = await self._reply_future
        self._reply_future = None
        return reply

    async def send_status(self, stage: str, text: str) -> None:
        """Emit a status/thinking event."""
        self.stream.emit_thinking(text, stage=stage)

    def provide_reply(self, text: str) -> None:
        """
        Called by the handler when the user replies to a clarification.

        Args:
            text: The user's reply text.
        """
        if self._reply_future and not self._reply_future.done():
            self._reply_future.set_result(text)

    @property
    def waiting_for_reply(self) -> bool:
        """True if the pipeline is currently suspended waiting for user input."""
        return self._reply_future is not None and not self._reply_future.done()


class NullChannel:
    """
    No-op channel for non-interactive callers (e.g., scheduled tasks, CEN).

    ask_user() returns empty string (pipeline will proceed with best effort).
    send_status() is silently ignored.
    """

    async def ask_user(self, questions: list[str]) -> str:
        return ""

    async def send_status(self, stage: str, text: str) -> None:
        pass
