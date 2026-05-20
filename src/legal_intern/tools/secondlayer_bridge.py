"""Bridge to SecondLayer MCP backend tools.

Allows agents to call SecondLayer API tools (search_court_decisions,
get_legislation, etc.) via the HTTP API at legal.org.ua/api/tools/:toolName.
"""

from __future__ import annotations

from typing import Any

import httpx


class SecondLayerBridge:
    """HTTP client for SecondLayer MCP tool execution."""

    def __init__(self, api_url: str, api_key: str) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.api_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=60.0,
            )
        return self._client

    async def call_tool(self, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        """Execute a SecondLayer MCP tool via HTTP API."""
        client = await self._get_client()
        response = await client.post(
            f"/tools/{tool_name}",
            json=params,
        )
        response.raise_for_status()
        return response.json()

    async def search_court_decisions(
        self,
        query: str,
        jurisdiction: str = "civil",
        limit: int = 10,
    ) -> list[dict]:
        result = await self.call_tool(
            "search_court_decisions",
            {"query": query, "jurisdiction": jurisdiction, "limit": limit},
        )
        return result.get("decisions", result.get("results", []))

    async def get_legislation(self, query: str, article: str = "") -> dict:
        params: dict[str, Any] = {"query": query}
        if article:
            params["article"] = article
        return await self.call_tool("get_legislation", params)

    async def search_supreme_court(self, query: str, category: str = "") -> list[dict]:
        params: dict[str, Any] = {"query": query}
        if category:
            params["category"] = category
        result = await self.call_tool("search_supreme_court_positions", params)
        return result.get("positions", result.get("results", []))

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
