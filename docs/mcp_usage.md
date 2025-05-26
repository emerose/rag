# Using the MCP Server with ChatGPT and Cursor

The Model Context Protocol (MCP) lets AI tools connect to your local RAG server. These steps show how to run the server and make it available to ChatGPT and the Cursor editor.

## 1. Start the MCP server

1. Launch the HTTP server:
   ```bash
   rag mcp-http --host 127.0.0.1 --port 8000
   ```
2. (Optional) Set an API key so only you can access the server:
   ```bash
   export RAG_MCP_API_KEY="my-secret-key"
   rag mcp-http --host 127.0.0.1 --port 8000
   ```

Keep the terminal open while the server is running.

## 2. Expose the server with Tailscale (optional)

Sometimes ChatGPT cannot reach `localhost` directly. Tailscale lets you share
your local server over a secure tunnel so ChatGPT can connect.

1. [Install Tailscale](https://tailscale.com/download) and run `tailscale up`.
2. Start the MCP server, listening on all interfaces:
   ```bash
   rag mcp-http --host 0.0.0.0 --port 8000
   ```
3. Find your Tailscale IP address:
   ```bash
   tailscale ip -4
   ```
4. Use `http://<tailscale-ip>:8000` when adding the server in ChatGPT or Cursor.

## 3. Add the server to ChatGPT

1. Open ChatGPT in your browser.
2. Choose **Settings & beta** → **Model context**.
3. Click **Add server** and enter `http://localhost:8000`.
4. If you set `RAG_MCP_API_KEY`, also enter the same key when prompted.
5. Save the configuration and enable the new context.

ChatGPT can now call your RAG tools whenever you ask a question.

## 4. Add the server to Cursor

1. In Cursor, open **Preferences**.
2. Go to **AI** → **Model context servers**.
3. Click **Add server** and type `http://localhost:8000`.
4. Provide the API key if required and confirm.

Once added, Cursor will use your RAG server for relevant features.

## 5. Stop the server

Press `Ctrl+C` in the terminal window running `mcp-http` when you want to stop the server.

