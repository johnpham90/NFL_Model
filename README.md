# Claude Desktop & MCP Server Setup Guide

## Prerequisites
- Git
- Node.js (v16 or higher)
- npm or yarn
- Anthropic API key
- GitHub account

## Installation Steps

### 1. Install Claude Desktop

```bash
# Clone the Claude Desktop repository
git clone https://github.com/anthropics/claude-desktop.git
cd claude-desktop

# Install dependencies
npm install

# Start the application
npm start
```

### 2. Configure Environment Variables

Add your Anthropic API key to your bash profile:

```bash
# Open .bashrc in text editor
nano ~/.bashrc

# Add these lines to .bashrc
export ANTHROPIC_API_KEY='your-api-key-here'
export GITHUB_TOKEN='your-github-token-here'  # For GitHub integration

# Save and reload bash profile
source ~/.bashrc
```

### 3. Install MCP Server

```bash
# Install global dependencies
npm install -g @anthropic-ai/mcp-server

# Install specific MCP servers
npx @anthropic-ai/mcp-nfl-model  # For NFL model specific functions
```

### 4. Connect Claude Desktop to MCP

1. Launch Claude Desktop
2. Open Settings (gear icon)
3. Navigate to "Connections"
4. Add new MCP connection:
   - Name: NFL_Model
   - URL: http://localhost:3000
   - Click "Connect"

### 5. Verify Installation

Test the setup with a simple query in Claude Desktop:
```
User: Test connection to NFL_Model MCP
Claude: Connection successful - ready to process NFL data
```

## Available MCP Functions

- `analyze_nfl_data`: Process NFL statistics
- `generate_predictions`: Create game predictions
- `update_models`: Refresh prediction models
- Additional functions documented in MCP server

## Troubleshooting

Common issues and solutions:

1. **API Key Issues**
   ```bash
   echo $ANTHROPIC_API_KEY  # Verify key is set
   ```

2. **MCP Connection Errors**
   ```bash
   # Restart MCP server
   pkill -f mcp-server
   npx @anthropic-ai/mcp-nfl-model
   ```

3. **Port Conflicts**
   ```bash
   # Check ports in use
   lsof -i :3000
   ```

## Updates and Maintenance

Keep your installation up to date:

```bash
# Update Claude Desktop
cd claude-desktop
git pull
npm install

# Update MCP server
npm update -g @anthropic-ai/mcp-server
```

## Support

For issues or questions:
- Claude Desktop: [Anthropic Support](https://support.anthropic.com)
- MCP Server: Check GitHub Issues or create new issue
- NFL Model: Refer to repository documentation

## Security Notes

- Never commit API keys to version control
- Regularly rotate API keys
- Keep all packages updated
- Use secure connections for data transfer