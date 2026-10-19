"""Tools that call the internet. Off unless `agent.options.allow_network:
true` in config (see registry.py), because OpenPersona is offline-first.

Both use services that need no API key:
- `web_search`: DuckDuckGo via the `ddgs` package (`tools-online` extra).
- `weather`: Open-Meteo's geocoding + forecast APIs via httpx.

Failures (no network, service down, nothing found) come back to the model
as text, so it can tell the user instead of the turn failing.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Annotated

from companion.tools.base import Tool, ToolContext, ToolInputError

if TYPE_CHECKING:
    import httpx

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_TIMEOUT_S = 10.0

# WMO weather interpretation codes used by Open-Meteo, grouped coarsely.
_WMO = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog", 51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "freezing drizzle", 57: "freezing drizzle", 61: "light rain", 63: "rain",
    65: "heavy rain", 66: "freezing rain", 67: "freezing rain", 71: "light snow",
    73: "snow", 75: "heavy snow", 77: "snow grains", 80: "light showers", 81: "showers",
    82: "violent showers", 85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with hail",
}


def build_web_search_tools(ctx: ToolContext) -> list[Tool]:
    async def web_search(query: Annotated[str, "What to search the web for"]) -> str:
        if not query.strip():
            raise ToolInputError("Search query is empty.")

        def _search() -> list[dict]:
            from ddgs import DDGS

            return DDGS().text(query, max_results=5) or []

        try:
            results = await asyncio.to_thread(_search)
        except ImportError:
            return "Web search isn't installed (pip install -e \".[tools-online]\")."
        except Exception as exc:  # noqa: BLE001 — ddgs raises several types; all mean "no results"
            return f"Web search failed: {type(exc).__name__}: {exc}"
        if not results:
            return f"No web results for {query!r}."
        return "\n".join(
            f"- {r.get('title', '').strip()}: {r.get('body', '').strip()[:300]} ({r.get('href', '')})"
            for r in results
        )

    return [Tool(
        "web_search",
        "Search the web for current information (news, opening hours, facts that "
        "may have changed). Cite the source links you use.",
        web_search,
    )]


async def fetch_weather(client: httpx.AsyncClient, place: str, days: int) -> str:
    geo = await client.get(GEOCODE_URL, params={"name": place, "count": 1, "format": "json"})
    geo.raise_for_status()
    matches = geo.json().get("results") or []
    if not matches:
        return f"Couldn't find a place called {place!r}."
    loc = matches[0]
    forecast = await client.get(FORECAST_URL, params={
        "latitude": loc["latitude"], "longitude": loc["longitude"], "timezone": "auto",
        "forecast_days": days, "current": "temperature_2m,weather_code",
        "daily": "temperature_2m_min,temperature_2m_max,precipitation_probability_max,weather_code",
    })
    forecast.raise_for_status()
    data = forecast.json()
    name = ", ".join(p for p in (loc.get("name"), loc.get("admin1"), loc.get("country")) if p)
    lines = [f"Weather for {name}:"]
    current = data.get("current") or {}
    if current:
        lines.append(f"Now: {current.get('temperature_2m')}°C, "
                     f"{_WMO.get(current.get('weather_code'), 'unknown')}")
    daily = data.get("daily") or {}
    for i, day in enumerate(daily.get("time", [])):
        rain = daily["precipitation_probability_max"][i]
        lines.append(
            f"{day}: {daily['temperature_2m_min'][i]}–{daily['temperature_2m_max'][i]}°C, "
            f"{_WMO.get(daily['weather_code'][i], 'unknown')}"
            + (f", {rain}% chance of rain" if rain is not None else "")
        )
    return "\n".join(lines)


def build_weather_tools(ctx: ToolContext) -> list[Tool]:
    async def weather(
        place: Annotated[str, "City or place name, e.g. 'Lisbon' or 'Pune, India'"],
        days: Annotated[int, "Days of forecast including today, 1 to 7"] = 3,
    ) -> str:
        if not place.strip():
            raise ToolInputError("Place is empty.")
        days = max(1, min(int(days), 7))
        import httpx

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
                return await fetch_weather(client, place.strip(), days)
        except httpx.HTTPError as exc:
            return f"Weather service unavailable: {type(exc).__name__}."

    return [Tool(
        "weather",
        "Current weather and a daily forecast (up to 7 days) for a place.",
        weather,
    )]
