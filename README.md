# Setting Up MCP Servers on Windows

## Prerequisites

### Download Git
https://git-scm.com/downloads/win

### 1. Download Node.js v18+
- Download from: https://nodejs.org/
- Verify in Git Bash:
```Git Bash
node --version
npm --version
```

### 2. Download Python 3.10+
- Download from: https://www.python.org/downloads/
- Check "Add Python to PATH"

## Installation

### 1. Package Managers
```Git Bash
# Open Git Bash and run
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
```

**Configuration Notes:**
- Replace `YourUsername` with your Windows username
- Replace `YOUR_API_KEY_HERE` with actual API keys
- Use double backslashes in Windows paths
- Point to `dist/index.js` in npm modules directory

## Server Setup

### Memory Server
- No additional setup
- Debug logging enabled

### Everything Server
- No additional setup
- Debug logging enabled

### Brave Search Server
1. Get API key: https://brave.com/search/api/
2. Add to config's env section

## Verification

```powershell
# List packages
npm list -g --depth=0

# Test servers
npx @modelcontextprotocol/server-memory
npx @modelcontextprotocol/server-brave-search
uvx mcp-server-sqlite
```

## Troubleshooting

### "Could not attach to MCP server"
- Verify config paths
- Check global package installation
- Confirm dist/index.js exists

### Server not visible
- Restart Claude Desktop
- Check JSON syntax
- Verify file paths

## Best Practices
- Use global installations
- Use full paths to dist/index.js
- Keep DEBUG env variable
- Restart after config changes
