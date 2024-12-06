# Setting Up MCP Servers on Windows

## Prerequisites
1. Node.js (v18+)
   - Download: https://nodejs.org/
2. Python 3.10+
   - Download: https://www.python.org/downloads/
   - Check "Add Python to PATH"

## Installation Steps

### 1. Install Package Managers
```powershell
npm install -g uv
```

### 2. Install MCP Servers
```powershell
npm install -g @modelcontextprotocol/server-memory
npm install -g @modelcontextprotocol/server-everything
npm install -g @modelcontextprotocol/server-brave-search
uvx mcp-server-sqlite
```

### 3. Configure Claude Desktop
Path: `%AppData%\Claude Desktop\claude_desktop_config.json`

```json
{
  "globalShortcut": "Ctrl+Space",
  "mcpServers": {
    "sqlite": {
      "command": "uvx",
      "args": ["mcp-server-sqlite", "--db-path", "C:\\Users\\YourUsername\\test.db"]
    },
    "memory": {
      "command": "node",
      "args": ["C:\\Users\\YourUsername\\AppData\\Roaming\\npm\\node_modules\\@modelcontextprotocol\\server-memory\\dist\\index.js"],
      "env": { "DEBUG": "*" }
    },
    "brave-search": {
      "command": "node",
      "args": ["C:\\Users\\YourUsername\\AppData\\Roaming\\npm\\node_modules\\@modelcontextprotocol\\server-brave-search\\dist\\index.js"],
      "env": {
        "BRAVE_API_KEY": "YOUR_API_KEY_HERE",
        "DEBUG": "*"
      }
    }
  }
}
```

## Verification
```powershell
npm list -g --depth=0
uvx mcp-server-sqlite --version
```

## Troubleshooting
- Restart Claude Desktop after config changes
- Check file paths in config
- Verify all packages installed globally