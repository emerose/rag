# Customizing Prompts

The RAG CLI supports different prompting strategies via the `--prompt` flag:

```bash
# Use chain-of-thought reasoning
rag query "What is the key feature of RAG?" --prompt cot

# Use a more conversational style
rag repl --prompt creative
```

Available prompt templates:

- `default`: Standard RAG prompt with citation guidance
- `cot`: Chain-of-thought prompt encouraging step-by-step reasoning
- `creative`: Engaging, conversational style while maintaining accuracy

You can view all available prompts at any time:

```bash
rag prompt list
```
