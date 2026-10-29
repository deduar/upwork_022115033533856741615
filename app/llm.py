import anthropic

from app import config

_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)


def complete(system: str, user: str, max_tokens: int = 16000) -> str:
    """Single Claude call. Returns the text reply ('' if the model declined)."""
    response = _client.beta.messages.create(
        model=config.CHAT_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        # If a safety classifier declines, the API retries on a fallback model.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        return ""
    return "".join(b.text for b in response.content if b.type == "text").strip()
