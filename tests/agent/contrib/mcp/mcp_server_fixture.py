"""A tiny FastMCP server used by the live stdio tests. Run as a subprocess."""

from mcp.server.fastmcp import FastMCP

srv = FastMCP("luca-test")


@srv.tool()
def echo(text: str) -> str:
    """Echo the text back."""
    return f"echo: {text}"


@srv.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


if __name__ == "__main__":
    srv.run()
