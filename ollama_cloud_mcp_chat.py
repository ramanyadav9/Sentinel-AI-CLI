import asyncio
import json
import os
import httpx
from typing import Optional, Any, Literal
from dataclasses import dataclass
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.table import Table
from rich.style import Style
from rich import box
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style as PromptStyle
from dotenv import load_dotenv
import ollama

load_dotenv()

# Universal Console for Rich output
console = Console()

# ------------------------------
# Configuration & Dataclasses
# ------------------------------
@dataclass
class MCPServer:
    name: str
    type: Literal["local", "remote"]
    command: Optional[str] = None
    args: Optional[list[str]] = None
    url: Optional[str] = None
    env: Optional[dict] = None
    headers: Optional[dict] = None

from contextlib import AsyncExitStack

# ... (rest of imports remain same)

class SentinelAIAgent:
    def __init__(self):
        # 1. Load Ollama Cloud Configuration
        self.ollama_api_key = os.getenv("OLLAMA_API_KEY")
        # FORCE use of official cloud endpoint regardless of local env vars (which might be 0.0.0.0)
        self.ollama_host = "https://ollama.com" 
        self.model = os.getenv("OLLAMA_CLOUD_MODEL", "gpt-oss:120b-cloud")

        if not self.ollama_api_key:
            raise ValueError("OLLAMA_API_KEY missing! Check your .env file.")

        # 2. Initialize Ollama Client
        self.client = ollama.Client(
            host=self.ollama_host,
            headers={"Authorization": f"Bearer {self.ollama_api_key}"},
            timeout=30.0
        )

        # 3. Agent State
        self.exit_stack = AsyncExitStack()
        self.mcp_servers: dict[str, ClientSession] = {}
        self.available_tools: list[dict] = []
        self.conversation_history: list[dict] = []

    # ------------------------------
    # Connection Methods
    # ------------------------------
    async def connect_mcp_server(self, server: MCPServer):
        """Connect to an MCP server (Local or Remote)"""
        try:
            if server.type == "local":
                params = StdioServerParameters(
                    command=server.command,
                    args=server.args,
                    env=server.env
                )
                transport = await self.exit_stack.enter_async_context(stdio_client(params))
            elif server.type == "remote":
                transport = await self.exit_stack.enter_async_context(sse_client(server.url, headers=server.headers or {}))
            else:
                return False

            read, write = transport
            session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

            # Tool Discovery
            tools = await session.list_tools()
            self._register_tools(server.name, tools.tools)
            
            self.mcp_servers[server.name] = session
            return True

        except Exception as e:
            console.print(f"[bold red]✗ Connection Failed:[/bold red] {server.name} ({str(e)})")
            return False

    def _register_tools(self, server_name: str, tools: list):
        for tool in tools:
            self.available_tools.append({
                "name": f"{server_name}_{tool.name}",
                "description": tool.description,
                "input_schema": tool.inputSchema,
                "server": server_name,
                "original_name": tool.name
            })

    async def chat(self, user_message: str):
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # Prepare tools for Ollama
        ollama_tools = [{
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"]
            }
        } for t in self.available_tools]

        max_turns = 5
        for _ in range(max_turns):
            try:
                # API Call
                response = self.client.chat(
                    model=self.model,
                    messages=self.conversation_history,
                    tools=ollama_tools if ollama_tools else None,
                    stream=False
                )
                
                msg = response["message"]
                self.conversation_history.append(msg)

                # Check for tool calls
                if not msg.get("tool_calls"):
                    return msg.get("content", "")

                # Execute Tools
                console.print(Panel(f"[bold yellow]⚡ executing {len(msg['tool_calls'])} tool(s)...[/bold yellow]", border_style="yellow"))
                
                for tool_call in msg["tool_calls"]:
                    fname = tool_call["function"]["name"]
                    fargs = tool_call["function"]["arguments"]
                    
                    # Find tool
                    tool_def = next((t for t in self.available_tools if t["name"] == fname), None)
                    if not tool_def:
                        continue

                    console.print(f"  [cyan]› {fname}[/cyan]")
                    
                    # Execute
                    session = self.mcp_servers[tool_def["server"]]
                    result = await session.call_tool(tool_def["original_name"], fargs)
                    
                    # Feed back to history
                    self.conversation_history.append({
                        "role": "tool",
                        "content": str(result.content)
                    })
                    
            except Exception as e:
                console.print(f"[bold red]API Error:[/bold red] {str(e)}")
                if hasattr(e, 'response'):
                    try:
                        console.print(f"[dim]Response Body: {e.response.text}[/dim]")
                    except:
                        pass
                return f"Error: {str(e)}"
        
        return "Max tool iterations reached."

    async def close(self):
        await self.exit_stack.aclose()

# ------------------------------
# UI Components
# ------------------------------
def print_banner():
    grid = Table.grid(expand=True)
    grid.add_column(justify="center", ratio=1)
    grid.add_column(justify="right")
    
    title = Text("SENTINEL-AI CLI", style="bold green", justify="center")
    subtitle = Text("Powered by Ollama Cloud & MCP", style="dim white", justify="center")
    
    panel = Panel(
        title,
        subtitle=subtitle,
        style="green",
        border_style="green",
        padding=(1, 2)
    )
    console.print(panel)

# ------------------------------
# Main
# ------------------------------
async def main():
    print_banner()

    agent = SentinelAIAgent()
    
    # --- MCP CONFIGURATION ---
    mcp_config = [
        MCPServer(
            name="sentinel-ai",
            type="local",
            command=r"E:\CyberSentinal All\Sentinel-AI\mcp-server\venv\Scripts\python.exe",
            args=["-m", "sentinel_mcp.server", "--config", r"E:\CyberSentinal All\Sentinel-AI\mcp-server\config.yaml"]
        ),
        # Add other servers here...
    ]

    # Connection Phase
    with console.status("[bold green]Initializing Secure Connection...[/bold green]", spinner="dots"):
        # 1. Connect to Ollama
        try:
            agent.client.list() # Test auth
            console.print(f"[bold green]✓[/bold green] Authorized with Ollama Cloud ({agent.model})")
        except Exception as e:
            console.print(f"[bold red]✗ Auth Failed![/bold red] {str(e)}")
            console.print_exception()
            return

        # 2. Connect to MCP
        for server in mcp_config:
            if await agent.connect_mcp_server(server):
                console.print(f"[bold green]✓[/bold green] Linked MCP: [cyan]{server.name}[/cyan]")

    # Chat Loop
    console.print("\n[dim]System Ready. Waiting for input...[/dim]\n")
    
    # Custom prompt style
    p_style = PromptStyle.from_dict({
        'prompt': '#00ff00 bold',
    })
    session = PromptSession(history=FileHistory(".sentinel_history"), style=p_style)

    while True:
        try:
            user_input = await asyncio.to_thread(session.prompt, "Sentinel-AI > ")
            
            if user_input.lower() in ("exit", "quit"):
                console.print("[green]Session Terminated.[/green]")
                break
                
            if not user_input.strip(): continue

            with console.status("[bold green]Processing...[/bold green]", spinner="line"):
                response = await agent.chat(user_input)

            console.print(Panel(Markdown(response), title="[bold green]Response[/bold green]", border_style="dim white"))
            console.print()

        except (KeyboardInterrupt, EOFError):
            break
    
    await agent.close()

if __name__ == "__main__":
    asyncio.run(main())
