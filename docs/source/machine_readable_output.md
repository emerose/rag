# Machine-Readable Output

All commands support machine-readable JSON output through the `--json` flag. JSON output is also automatically enabled when the output is not a terminal (e.g., when piping to another command).

```bash
# Enable JSON output explicitly
rag query "What are the main findings?" --json

# JSON output is automatic when piping
rag query "What are the main findings?" | jq .answer

# List documents in JSON format
rag list --json | jq '.table.rows[] | select(.[0] | contains(".pdf"))'

# Get indexing results in JSON
rag index docs/ --json | jq '.summary.total_files'
```

## JSON Output Format

Each command produces structured JSON output:

1. Simple messages:
   ```json
   {
     "message": "Successfully processed file.txt"
   }
   ```

2. Error messages:
   ```json
   {
     "error": "File not found: missing.txt"
   }
   ```

3. Tables (e.g., from `rag list`):
   ```json
   {
     "table": {
       "title": "Indexed Documents",
       "columns": ["File", "Type", "Modified", "Size"],
       "rows": [
         ["doc1.pdf", "PDF", "2024-03-20", "1.2MB"],
         ["doc2.txt", "Text", "2024-03-19", "50KB"]
       ]
     }
   }
   ```

4. Multiple tables:
   ```json
   {
     "tables": [
       {
         "title": "Summary",
         "columns": ["Metric", "Value"],
         "rows": [["Total Files", "10"], ["Processed", "8"]]
       },
       {
         "title": "Details",
         "columns": ["File", "Status"],
         "rows": [["file1.txt", "Success"], ["file2.pdf", "Error"]]
       }
     ]
   }
   ```

5. Command-specific data (e.g., from `rag query`):
   ```json
   {
     "answer": "The main findings are...",
     "sources": [
       {
         "file": "paper.pdf",
         "page": 12,
         "text": "..."
       }
     ],
     "metadata": {
       "tokens_used": 150,
       "model": "gpt-4"
     }
 }
  ```
