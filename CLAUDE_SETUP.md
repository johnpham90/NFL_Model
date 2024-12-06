# Setting Up MCP Servers on Windows

## Prerequisites

### 1. Node.js v18+
- Download from: https://nodejs.org/
- Verify in PowerShell:
```powershell
node --version
npm --version
```

### 2. Python 3.10+
- Download from: https://www.python.org/downloads/
- Check "Add Python to PATH"

## Installation

### 1. Package Managers
```powershell
# Open PowerShell as admin
npm install -g uv
```

### 2. MCP Servers
Node.js servers:
```powershell
npm install -g @modelcontextprotocol/server-memory
npm install -g @modelcontextprotocol/server-everything
npm install -g @modelcontextprotocol/server-brave-search
```

Python servers:
```powershell
uvx mcp-server-sqlite
```

### 3. Claude Desktop Configuration

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
      "env": {
        "DEBUG": "*"
      }
    },
    "everything": {
      "command": "node",
      "args": ["C:\\Users\\YourUsername\\AppData\\Roaming\\npm\\node_modules\\@modelcontextprotocol\\server-everything\\dist\\index.js"],
      "env": {
        "DEBUG": "*"
      }
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