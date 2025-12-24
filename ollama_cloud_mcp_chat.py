import asyncio
import json
import os
import httpx
import time
import math
from typing import Optional, Any, Literal
from dataclasses import dataclass
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
try:
    from mcp.client.streamable_http import streamablehttp_client as http_client
except ImportError:
    http_client = None  # Handle older mcp versions gracefully or error later
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
    transport: Literal["sse", "http"] = "sse"

from contextlib import AsyncExitStack

# ... (rest of imports remain same)

class SentinelAIAgent:
    def __init__(self):
        # 1. Load Configurations
        self.primary_host = os.getenv("PRIMARY_OLLAMA_HOST")
        self.primary_model = os.getenv("PRIMARY_OLLAMA_MODEL")
        
        self.fallback_host = os.getenv("FALLBACK_OLLAMA_HOST", "https://ollama.com")
        self.fallback_model = os.getenv("FALLBACK_OLLAMA_MODEL")
        self.ollama_api_key = os.getenv("OLLAMA_API_KEY")

        if not self.primary_host or not self.primary_model:
            console.print("[bold yellow]⚠ PRIMARY_OLLAMA_HOST or MODEL not set in .env![/bold yellow]")

        # 2. Initialize Clients
        # Primary Client (Usually local/remote server without auth)
        self.primary_client = ollama.Client(
            host=self.primary_host,
            timeout=30.0
        )
        
        # Fallback Client (Usually Ollama Cloud with auth)
        fallback_headers = {}
        if self.ollama_api_key:
            fallback_headers["Authorization"] = f"Bearer {self.ollama_api_key}"
            
        self.fallback_client = ollama.Client(
            host=self.fallback_host,
            headers=fallback_headers,
            timeout=30.0
        )

        # 3. Agent State
        self.exit_stack = AsyncExitStack()
        self.mcp_servers: dict[str, ClientSession] = {}
        self.available_tools: list[dict] = []
        self.conversation_history: list[dict] = []
        self.using_fallback = False

    def safe_json(obj):
        try:
            return json.dumps(obj)
        except TypeError:
            return json.dumps(str(obj))

    def estimate_tokens(text: str) -> int:
        """Rough estimate of tokens (1 token ~= 4 chars)"""
        if not text: return 0
        return math.ceil(len(text) / 4)

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
                if server.transport == "sse":
                    transport = await self.exit_stack.enter_async_context(sse_client(server.url, headers=server.headers or {}))
                elif server.transport == "http":
                    if http_client is None:
                         raise ImportError("mcp.client.http module not found. Please upgrade 'mcp' package.")
                    transport = await self.exit_stack.enter_async_context(http_client(server.url, headers=server.headers or {}))
                else:
                    raise ValueError(f"Unknown transport: {server.transport}")
            else:
                return False

            # Extract read/write streams safely (handle potential 3-item tuples)
            if isinstance(transport, tuple) and len(transport) >= 2:
                read, write = transport[0], transport[1]
            else:
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

        max_turns = 10
        for _ in range(max_turns):
            current_client = self.primary_client
            current_model = self.primary_model
            
            if self.using_fallback:
                current_client = self.fallback_client
                current_model = self.fallback_model

            try:
                # API Call
                if _ == 0:
                   console.print(f"[dim]Calling {current_model} with {len(ollama_tools)} tools...[/dim]")
                
                response = current_client.chat(
                    model=current_model,
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
                    
                    # Execute with timing and stats
                    start_time = time.time()
                    
                    session = self.mcp_servers[tool_def["server"]]
                    result = await session.call_tool(tool_def["original_name"], fargs)
                    
                    duration = time.time() - start_time
                    
                    # --- TRUNCATION LOGIC ---
                    # Logs can be massive. We truncate to keep context manageable while using model capacity.
                    content_str = str(result.content)
                    max_chars = 500000 # Roughly 125k tokens (fits within most model windows)
                    if len(content_str) > max_chars:
                        console.print(f"  [yellow]⚠ Truncating response from {len(content_str)} to {max_chars} chars[/yellow]")
                        content_str = content_str[:max_chars] + "\n... [TRUNCATED DUE TO SIZE] ..."

                    # Calculate stats
                    input_str = json.dumps(fargs)
                    input_tokens = SentinelAIAgent.estimate_tokens(input_str)
                    output_tokens = SentinelAIAgent.estimate_tokens(content_str)
                    total_tokens = input_tokens + output_tokens
                    
                    # Print Stats Panel
                    stats_grid = Table.grid(expand=True, padding=(0,2))
                    stats_grid.add_column(style="dim")
                    stats_grid.add_column(style="bold white")
                    
                    stats_grid.add_row("Duration:", f"{duration:.2f}s")
                    stats_grid.add_row("Input Tokens:", f"{input_tokens}")
                    stats_grid.add_row("Output Tokens:", f"{output_tokens}")
                    stats_grid.add_row("Total Tokens:", f"{total_tokens}")
                    
                    console.print(Panel(
                        stats_grid, 
                        title=f"[bold green]Tool Stats: {fname}[/bold green]",
                        border_style="dim green",
                        width=60
                    ))
                    
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id"),
                        "content": content_str # Use truncated version
                    })
            except Exception as e:
                if not self.using_fallback:
                    console.print(f"[bold yellow]⚠ Primary Server Failed:[/bold yellow] {str(e)}")
                    console.print(f"[dim]Attempting Fallback...[/dim]")
                    self.using_fallback = True
                    # Re-attempt loop iteration with fallback client
                    continue
                
                console.print(f"[bold red]API Error (Fallback):[/bold red] {str(e)}")
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
        # --- Local Sentinel-AI Server (Default) ---
        MCPServer(
            name="sentinel-ai",
            type="local",
            command=r"E:\CyberSentinal All\Sentinel-AI\mcp-server\run_mcp_server.bat",
            args=[]
        ),

        # --- EXAMPLES: Uncomment to enable ---

        # 1. Filesystem (Read/Write local files)
        # MCPServer(
        #     name="filesystem",
        #     type="local",
        #     command="npx",
        #     args=["-y", "@modelcontextprotocol/server-filesystem", "C:\\Users\\91916\\Documents"] 
        # ),

        # 2. GitHub (Search repos, read files, manage issues)
        # Requires GITHUB_PERSONAL_ACCESS_TOKEN in env
        # MCPServer(
        #     name="github",
        #     type="local",
        #     command="npx",
        #     args=["-y", "@modelcontextprotocol/server-github"],
        #     env={"GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")}
        # ),

        # 3. Brave Search (Internet search)
        # Requires BRAVE_API_KEY in env
        # MCPServer(
        #     name="brave-search",
        #     type="local",
        #     command="npx",
        #     args=["-y", "@modelcontextprotocol/server-brave-search"],
        #     env={"BRAVE_API_KEY": os.getenv("BRAVE_API_KEY")}
        # ),

        # 4. Google Maps (Location search)
        # Requires GOOGLE_MAPS_API_KEY in env
        # MCPServer(
        #     name="google-maps",
        #     type="local",
        #     command="npx",
        #     args=["-y", "@modelcontextprotocol/server-google-maps"],
    ]


    # Connection Phase
    with console.status("[bold green]Initializing Secure Connection...[/bold green]", spinner="dots"):
        # 1. Connect to Ollama Primary
        try:
            agent.primary_client.list() # Test connection
            console.print(f"[bold green]✓[/bold green] Linked Primary: [cyan]{agent.primary_host}[/cyan] ({agent.primary_model})")
        except Exception as e:
            console.print(f"[bold yellow]⚠ Primary Offline:[/bold yellow] {agent.primary_host}")
            agent.using_fallback = True

        if agent.using_fallback:
            try:
                agent.fallback_client.list() # Test auth
                console.print(f"[bold cyan]ℹ Fallback Active:[/bold cyan] {agent.fallback_host} ({agent.fallback_model})")
            except Exception as e:
                console.print(f"[bold red]✗ Fallback Auth Failed![/bold red] {str(e)}")
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
