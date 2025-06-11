"""Type stubs for LangChain package."""

from langchain_core.documents import Document as CoreDocument

class Schema:
    Document = CoreDocument

# Make it available as schema (lowercase)
schema = Schema()
