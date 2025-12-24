# Ollama Cloud MCP Chat Agent

This agent connects directly to Ollama's cloud API to use hosted models like `qwen-cloud` or `llama3.2-cloud`, while retaining full support for Model Context Protocol (MCP) tools.

## Setup

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configuration:**
    - **Ollama Cloud Key**:
        - You need an account on [Ollama Cloud](https://ollama.com).
        - Go to [Settings](https://ollama.com/account/settings) to generate your API Key.
        - Create a file named `.env` in this directory (copy from `.env.example`).
        - Add your key: `OLLAMA_API_KEY="your-key-here"`.
    - **MCP Servers**:
        - Configure any local or remote MCP servers you want to use in `ollama_cloud_mcp_chat.py`.

3.  **Run the Agent:**
    ```bash
    python ollama_cloud_mcp_chat.py
    ```

## Features

- **Ollama Cloud**: Access hosted models without local GPU requirements.
- **MCP Support**: Use both local (filesystem, SQLite, etc.) and remote (Tavily, Browserbase, etc.) MCP tools.
- **Rich CLI**: A beautiful command-line interface with formatted output.

## Adding New MCP Servers

To add a new MCP server (e.g., a filesystem server or a new security tool), you need to register it in the `ollama_cloud_mcp_chat.py` file.

1.  Open `ollama_cloud_mcp_chat.py`.
2.  Locate the `main()` function and the `mcp_config` list.
3.  Add a new `MCPServer` entry to the list.

**Example: Adding a Read-Only Filesystem MCP**

```python
MCPServer(
    name="filesystem",
    type="local",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "C:\\Users\\Username\\Documents"],
)
```

**Example: Adding a Remote MCP**

```python
MCPServer(
    name="remote-tool",
    type="remote",
    url="https://api.example.com/mcp/sse",
    headers={"Authorization": "Bearer YOUR_TOKEN"}
)
```

**Note**: Ensure any commands (like `npx` or python scripts) are available in your system PATH or provide the full absolute path.

## 📚 Available MCP Servers Directory

Here are some popular official MCP servers you can add. Most are run using `npx`.

### 📂 Filesystem
Give the agent access to read/write files in a specific directory.
- **Command**: `npx -y @modelcontextprotocol/server-filesystem <path-to-folder>`
- **Env Vars**: None

### 🐙 GitHub
Search repositories, read code, open issues/PRs.
- **Command**: `npx -y @modelcontextprotocol/server-github`
- **Env Vars**: `GITHUB_PERSONAL_ACCESS_TOKEN`

### 🦁 Brave Search
Perform internet searches.
- **Command**: `npx -y @modelcontextprotocol/server-brave-search`
- **Env Vars**: `BRAVE_API_KEY` (Get from [Brave API](https://api.search.brave.com/app/dashboard))

### 🗺️ Google Maps
Search for places, get directions and elevation.
- **Command**: `npx -y @modelcontextprotocol/server-google-maps`
- **Env Vars**: `GOOGLE_MAPS_API_KEY`

### 📹 LiveKit Docs
Access LiveKit documentation directly.
- **Type**: Remote (SSE)
- **URL**: `https://docs.livekit.io/mcp`
- **Env Vars**: None

### 🐘 PostgreSQL
Read-only database access to inspect schemas and run queries.
- **Command**: `npx -y @modelcontextprotocol/server-postgres <postgres-connection-string>`
- **Env Vars**: None
