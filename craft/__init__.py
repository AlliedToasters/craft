"""craft — LLM-powered Minecraft agent."""

# Load .env eagerly so any `from craft.X import Y` (config, tools,
# testkit, llm, …) sees the same env vars regardless of import order.
# Defaults in craft.config kick in only for keys absent from both the
# real environment and .env.
from dotenv import load_dotenv  # noqa: E402

load_dotenv()
