# Using the MCP Server with ChatGPT and Cursor

The Model Context Protocol (MCP) lets AI tools connect to your local RAG server. These steps show how to run the server and make it available to ChatGPT and the Cursor editor.

## 1. Start the MCP server

1. Launch the server:
   ```bash
   rag serve-mcp --host 127.0.0.1 --port 8000
   ```
2. (Optional) Set an API key so only you can access the server:
   ```bash
   export RAG_MCP_API_KEY="my-secret-key"
   rag serve-mcp --host 127.0.0.1 --port 8000
   ```

Keep the terminal open while the server is running.

## 2. Add the server to ChatGPT

1. Open ChatGPT in your browser.
2. Choose **Settings & beta** → **Model context**.
3. Click **Add server** and enter `http://localhost:8000`.
4. If you set `RAG_MCP_API_KEY`, also enter the same key when prompted.
5. Save the configuration and enable the new context.

ChatGPT can now call your RAG tools whenever you ask a question.

## 3. Add the server to Cursor

1. In Cursor, open **Preferences**.
2. Go to **AI** → **Model context servers**.
3. Click **Add server** and type `http://localhost:8000`.
4. Provide the API key if required and confirm.

Once added, Cursor will use your RAG server for relevant features.

## 4. Stop the server

Press `Ctrl+C` in the terminal window running `serve-mcp` when you want to stop the server.

