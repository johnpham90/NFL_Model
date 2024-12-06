# Claude Desktop & MCP Setup Guide

## Prerequisites
- Git Bash
- Node.js v16+
- Python & uv
- Anthropic API key
- GitHub token

## 1. Install Claude Desktop
```bash
# Clone and install
git clone https://github.com/anthropics/claude-desktop.git
cd claude-desktop
npm install
npm start
```

## 2. Configure API Keys
```bash
# Edit bash profile
nano ~/.bashrc

# Add keys
export ANTHROPIC_API_KEY='your-key'
export GITHUB_TOKEN='your-token'

# Reload
source ~/.bashrc
```

## 3. Install Python uv
```bash
# Install
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify
uv --version
```

## 4. Install MCP Servers
```bash
# Core server
npm install -g @anthropic-ai/mcp-server

# Specific servers
npx @anthropic-ai/mcp-shell  # For shell commands
npx @anthropic-ai/mcp-nfl-model  # For NFL data
```

## 5. Configure Claude Desktop
1. Open Settings
2. Go to Connections
3. Add MCP connections:
   ```
   Name: shell
   URL: http://localhost:3000
   
   Name: nfl
   URL: http://localhost:3001
   ```

## 6. Verify Setup
Test in Claude Desktop:
```
User: Test MCP connection
Claude: Connection active
```

## Common Issues

### API Key Problems
```bash
echo $ANTHROPIC_API_KEY  # Check key
```

### Port Conflicts
```bash
lsof -i :3000  # Check ports
pkill -f mcp-server  # Kill existing
```

### Connection Failed
1. Restart MCP servers
2. Restart Claude Desktop
3. Check server logs

## Updates
```bash
# Update Claude
cd claude-desktop
git pull
npm install

# Update MCP
npm update -g @anthropic-ai/mcp-server
```

## Security
- Store API keys securely
- Use HTTPS for connections
- Keep dependencies updated
- Monitor server logs

## Documentation
- [Claude API](https://docs.anthropic.com/claude/)
- [MCP Server](https://docs.anthropic.com/mcp/)
- [GitHub Integration](https://docs.github.com/en/rest)
