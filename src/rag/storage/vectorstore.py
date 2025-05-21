"""Vectorstore management module for the RAG system.

This module provides functionality for managing vector stores using FAISS,
including creation, loading, saving, and querying operations.
"""

import logging
import os
import pickle
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import faiss
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from ..utils.logging_utils import log_message

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages vector stores for the RAG system.
    
    This class provides methods for creating, loading, saving, and querying
    vector stores using FAISS.
    """
    
    def __init__(self, 
                 cache_dir: Union[Path, str],
                 embeddings: Embeddings,
                 log_callback: Optional[Any] = None) -> None:
        """Initialize the vector store manager.
        
        Args:
            cache_dir: Directory for storing vector store cache files
            embeddings: Embedding provider
            log_callback: Optional callback for logging
        """
        self.cache_dir = Path(cache_dir)
        self.embeddings = embeddings
        self.log_callback = log_callback
        # Get the embedding dimension once at initialization
        self._embedding_dimension = None
        
    def _log(self, level: str, message: str) -> None:
        """Log a message.
        
        Args:
            level: Log level (INFO, WARNING, ERROR, etc.)
            message: The log message
        """
        log_message(level, message, "VectorStore", self.log_callback)
        
    def _get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings.
        
        Returns:
            Dimension of embeddings
        """
        if self._embedding_dimension is None:
            # Generate a sample embedding to get the dimension
            self._log("DEBUG", "Getting embedding dimension from provider")
            sample_embedding = self.embeddings.embed_query("sample text")
            self._embedding_dimension = len(sample_embedding)
            self._log("DEBUG", f"Embedding dimension: {self._embedding_dimension}")
        
        return self._embedding_dimension
        
    def get_cache_path(self, file_path: str) -> Path:
        """Get the path to the cached vector store for a file.
        
        Args:
            file_path: Path to the source file
            
        Returns:
            Path to the cached vector store
        """
        # Generate a filename-safe hash of the file path
        import hashlib
        file_hash = hashlib.md5(file_path.encode()).hexdigest()
        return self.cache_dir / f"{file_hash}.faiss"
        
    def load_vectorstore(self, file_path: str) -> Optional[FAISS]:
        """Load a vector store from cache.
        
        Args:
            file_path: Path to the source file
            
        Returns:
            Loaded FAISS vector store or None if not found
        """
        cache_path = self.get_cache_path(file_path)
        index_path = cache_path.with_suffix(".pkl")
        
        if not cache_path.exists() or not index_path.exists():
            self._log("DEBUG", f"Vector store not found for {file_path}")
            return None
            
        try:
            self._log("DEBUG", f"Loading vector store for {file_path}")
            vectorstore = FAISS.load_local(
                str(cache_path.parent),
                self.embeddings,
                str(cache_path.name),
                str(index_path.name)
            )
            
            # Verify the vector store has documents
            if not vectorstore.docstore._dict:
                self._log("WARNING", f"Loaded empty vector store for {file_path}, discarding")
                return None
                
            return vectorstore
            
        except Exception as e:
            self._log("ERROR", f"Failed to load vector store for {file_path}: {e}")
            return None
            
    def save_vectorstore(self, file_path: str, vectorstore: FAISS) -> bool:
        """Save a vector store to cache.
        
        Args:
            file_path: Path to the source file
            vectorstore: FAISS vector store to save
            
        Returns:
            True if successful, False otherwise
        """
        cache_path = self.get_cache_path(file_path)
        
        try:
            self._log("DEBUG", f"Saving vector store for {file_path}")
            
            # Create directory if it doesn't exist
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Save the vectorstore
            index_name = cache_path.name
            vectorstore.save_local(str(self.cache_dir), index_name)
            
            self._log("DEBUG", f"Saved vector store for {file_path}")
            return True
            
        except Exception as e:
            self._log("ERROR", f"Failed to save vector store for {file_path}: {e}")
            self._log("ERROR", f"Traceback: {traceback.format_exc()}")
            return False
            
    def create_vectorstore(self, documents: List[Document]) -> FAISS:
        """Create a new vector store from documents.
        
        Args:
            documents: List of documents to add to the vector store
            
        Returns:
            FAISS vector store containing the documents
        """
        self._log("DEBUG", f"Creating vector store with {len(documents)} documents")
        try:
            if not documents:
                return self.create_empty_vectorstore()
            return FAISS.from_documents(documents, self.embeddings)
        except Exception as e:
            self._log("ERROR", f"Failed to create vector store: {e}")
            self._log("ERROR", f"Traceback: {traceback.format_exc()}")
            raise
        
    def create_empty_vectorstore(self) -> FAISS:
        """Create an empty vector store.
        
        Returns:
            Empty FAISS vector store
        """
        self._log("DEBUG", "Creating empty vector store")
        try:
            # Get embedding dimension
            dim = self._get_embedding_dimension()
            
            # Create empty index
            index = faiss.IndexFlatL2(dim)
            
            # Create empty docstore
            docstore = InMemoryDocstore({})
            
            # Create FAISS instance with the empty index
            return FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=docstore,
                index_to_docstore_id={}
            )
        except Exception as e:
            self._log("ERROR", f"Failed to create empty vector store: {e}")
            self._log("ERROR", f"Traceback: {traceback.format_exc()}")
            raise
        
    def add_documents_to_vectorstore(self, 
                                    vectorstore: Optional[FAISS], 
                                    documents: List[Document],
                                    embeddings: List[List[float]]) -> FAISS:
        """Add documents to a vector store.
        
        Args:
            vectorstore: Existing vector store or None to create a new one
            documents: Documents to add
            embeddings: Pre-computed embeddings for the documents
            
        Returns:
            Updated FAISS vector store
        """
        if not documents:
            self._log("DEBUG", "No documents to add to vector store")
            return vectorstore if vectorstore else self.create_empty_vectorstore()
            
        if not vectorstore:
            # Create a new vector store if none exists
            self._log("DEBUG", f"Creating new vector store with {len(documents)} documents")
            vectorstore = self.create_empty_vectorstore()
            
        try:
            self._log("DEBUG", f"Adding {len(documents)} documents to vector store")
            self._log("DEBUG", f"Documents length: {len(documents)}, Embeddings length: {len(embeddings)}")
            
            # Log dimensions of embeddings for debugging
            for i, emb in enumerate(embeddings[:5]):  # Only log first 5 for brevity
                if emb:
                    self._log("DEBUG", f"Embedding {i} dimension: {len(emb)}")
                else:
                    self._log("DEBUG", f"Embedding {i} is empty")
            
            # Validate that we have embeddings for all documents
            if len(documents) != len(embeddings):
                self._log("ERROR", f"Number of documents ({len(documents)}) doesn't match number of embeddings ({len(embeddings)})")
                return vectorstore
                
            # Make sure no embeddings are empty
            valid_entries = []
            for i, (doc, emb) in enumerate(zip(documents, embeddings)):
                if not emb or len(emb) == 0:
                    self._log("WARNING", f"Empty embedding for document {i}, skipping")
                    continue
                valid_entries.append((doc, emb))
                
            if not valid_entries:
                self._log("WARNING", "No valid embeddings found, returning original vectorstore")
                return vectorstore
                
            self._log("DEBUG", f"Valid entries: {len(valid_entries)}")
            valid_docs, valid_embs = zip(*valid_entries)
            
            # Convert embeddings to numpy array
            self._log("DEBUG", "Converting embeddings to numpy array")
            embedding_matrix = np.array(valid_embs, dtype=np.float32)
            self._log("DEBUG", f"Embedding matrix shape: {embedding_matrix.shape}")
            
            # Add documents one by one with their embeddings
            for doc, embedding in zip(valid_docs, embedding_matrix):
                doc_id = str(len(vectorstore.docstore._dict))
                vectorstore.docstore._dict[doc_id] = doc
                vectorstore.index.add(np.array([embedding]))
                vectorstore.index_to_docstore_id[len(vectorstore.index_to_docstore_id)] = doc_id
            
            return vectorstore
            
        except Exception as e:
            self._log("ERROR", f"Failed to add documents to vector store: {e}")
            self._log("ERROR", f"Traceback: {traceback.format_exc()}")
            if vectorstore:
                return vectorstore
            return self.create_empty_vectorstore()
            
    def merge_vectorstores(self, vectorstores: List[FAISS]) -> FAISS:
        """Merge multiple vector stores into one.
        
        Args:
            vectorstores: List of FAISS vector stores to merge
            
        Returns:
            Merged FAISS vector store
        """
        if not vectorstores:
            return self.create_empty_vectorstore()
            
        if len(vectorstores) == 1:
            return vectorstores[0]
            
        # Start with the first vector store
        merged = vectorstores[0]
        
        # Merge in the rest
        for vs in vectorstores[1:]:
            try:
                # Get documents and their embeddings from the current store
                docs = []
                embeddings = []
                for i, (doc_id, doc) in enumerate(vs.docstore._dict.items()):
                    docs.append(doc)
                    try:
                        embeddings.append(vs.index.reconstruct(i))
                    except Exception as e:
                        self._log("ERROR", f"Failed to reconstruct embedding at index {i}: {e}")
                        continue
                    
                # Add to the merged store
                self.add_documents_to_vectorstore(merged, docs, embeddings)
                
            except Exception as e:
                self._log("ERROR", f"Failed to merge vector store: {e}")
                self._log("ERROR", f"Traceback: {traceback.format_exc()}")
                
        return merged
        
    def similarity_search(self, 
                          vectorstore: FAISS, 
                          query: str, 
                          k: int = 4) -> List[Document]:
        """Perform a similarity search on a vector store.
        
        Args:
            vectorstore: FAISS vector store to search
            query: Query string
            k: Number of results to return
            
        Returns:
            List of documents matching the query
        """
        self._log("DEBUG", f"Performing similarity search with k={k}")
        try:
            return vectorstore.similarity_search(query, k=k)
        except Exception as e:
            self._log("ERROR", f"Failed to perform similarity search: {e}")
            self._log("ERROR", f"Traceback: {traceback.format_exc()}")
            return [] 
