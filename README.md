# Setting Up MCP Servers on Windows

## Prerequisites

### 1. Download Git
- https://git-scm.com/downloads/win

### 2. Download Node.js v18+
- Download from: https://nodejs.org/
- Verify in Git Bash:
```Git Bash
node --version
npm --version
```

### 3. Download Python 3.10+
- Download from: https://www.python.org/downloads/
- Check "Add Python to PATH"
- If already installed, ensure that Python Path is add to Enviroment Variables
- https://phoenixnap.com/kb/windows-set-environment-variable#:~:text=want%20to%20check.-,Set%20Environment%20Variable%20in%20Windows%20via%20GUI,Variable%20prompt%20and%20click%20OK.

### 4. Dowload Claude Desktop
- https://claude.ai/download
- Cloud Desktop will require a Pro Plan Subscription $20/month
  
## Installation


### 1. Package Managers
```Git Bash
# Open Git Bash and run
npm install -g uv

```

### 2. MCP Servers
Install Node.js servers using Git Bash:
```Git Bash
npm install -g @modelcontextprotocol/server-memory
npm install -g @modelcontextprotocol/server-everything
npm install -g @modelcontextprotocol/server-brave-search
```

### 3. Claude Desktop Configuration

Navigate to your MCP Configuration Path. 
- It is typically here: `%AppData%\Claude Desktop\claude_desktop_config.json`
- But you will want to verify
- To verify tour path:
  - Navigate to Claude Desktop
  - On the top Left go to Files > Settings (CTRL + Comma)
  - Click on `Developer` Tab > `Edit Config`
  - the json config file name should be named `claude_desktop_cofig`
  - Use a text editor or IDE to update the config file ie Notepad ++, VS Code

```json
{
  "mcpServers": {
    "memory": {
      "command": "node",
      "args": [
        "C:\\Users\\YourUserName\\AppData\\Roaming\\npm\\node_modules\\@modelcontextprotocol\\server-memory\\dist\\index.js"
      ],
      "env": {
        "DEBUG": "*"
      }
    },
    "everything": {
      "command": "node",
      "args": [
        "C:\\Users\\YourUserName\\AppData\\Roaming\\npm\\node_modules\\@modelcontextprotocol\\server-everything\\dist\\index.js"
      ],
      "env": {
        "DEBUG": "*"
      }
    },
    "brave-search": {
      "command": "node",
      "args": [
        "C:\\Users\\YourUserName\\AppData\\Roaming\\npm\\node_modules\\@modelcontextprotocol\\server-brave-search\\dist\\index.js"
      ],
      "env": {
        "BRAVE_API_KEY": "YOUR_BRAVE_TOKEN",
        "DEBUG": "*"
      }
    },
    "github": {
      "command": "node",
      "args": [
        "C:\\Users\\YourUserName\\AppData\\Roaming\\npm\\node_modules\\@modelcontextprotocol\\server-github\\dist\\index.js"
      ],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "GIT_HUB_ACCES_TOEKN",
        "DEBUG": "*"
      }
    },
    "mcp-installer": {
      "command": "npx",
      "args": [
        "@anaisbetts/mcp-installer"
      ]
    },
    "server-puppeteer": {
      "command": "npx",
      "args": [
        "@modelcontextprotocol/server-puppeteer"
      ]
    },
    "mcp-shell": {
      "command": "uvx",
      "args": [
        "@anthropic-ai/mcp-shell"
      ]
    }
  }
}
```

**Configuration Notes:**
- Replace `YourUsername` with your Windows username
- Replace `YOUR_BRAVE_TOKEN` with actual API keys
- Replace  `GIT_HUB_ACCES_TOEKN` with your git hub token
- Use double backslashes in Windows paths
- Point to `dist/index.js` in npm modules directory

**API Keys:
-Brave Search Server:
  - Get API key: https://brave.com/search/api/
  - This Subscription is free

-GitHub Tokenn:
  - https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens
  - Repo Admin will have to generate a Token, and assign security privleges for that token


## Verification

```# Use Git Bash to get a list servers that were installed
# List packages
npm list -g --depth=0

```

```# Use git bash to test servers that were installed
npx @modelcontextprotocol/server-memory
npx @modelcontextprotocol/server-brave-search
npx @modelcontextprotocol/server-github
@modelcontextprotocol/server-filesystem

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
