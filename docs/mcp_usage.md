# Using the MCP Server with Claude

The Model Context Protocol (MCP) lets AI tools interact with your RAG server. Claude's desktop app can run an MCP server over stdio, allowing direct access to your local data.

## 1. Install the server in Claude

Run the `mcp install` command from the project root. This adds your RAG server to Claude's configuration:

```bash
mcp install src/rag/mcp_server.py:mcp --name RAG
```

You can include environment variables if needed:

```bash
mcp install src/rag/mcp_server.py:mcp --name RAG \
  --env-var RAG_MCP_API_KEY=my-secret-key
```

## 2. Start the server from Claude


Open Claude Desktop and choose your newly added **RAG** server from the MCP servers list. When launched, Claude connects to the server over stdio and can call the available tools.

## 3. Other integrations

The `mcp dev` command runs the server with the MCP Inspector for testing. You can also start the server manually with `mcp run src/rag/mcp_server.py:mcp` if you prefer.

## 4. Stop the server

Stop the server from within Claude or terminate the `mcp run` process when running manually.
