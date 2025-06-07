"""Tests for fake embedding service implementations."""

import pytest

from rag.embeddings.fakes import DeterministicEmbeddingService, FakeEmbeddingService


class TestFakeEmbeddingService:
    """Tests for FakeEmbeddingService."""

    def test_initialization_with_defaults(self) -> None:
        """Test default initialization."""
        service = FakeEmbeddingService()
        
        assert service.embedding_dimension == 384
        
        model_info = service.get_model_info()
        assert model_info["model_name"] == "fake-embedding-model"
        assert model_info["model_version"] == "1.0.0"
        assert model_info["dimension"] == "384"
        assert model_info["type"] == "fake"

    def test_initialization_with_custom_values(self) -> None:
        """Test initialization with custom values."""
        service = FakeEmbeddingService(
            embedding_dimension=768,
            model_name="custom-model",
            model_version="2.1.0",
        )
        
        assert service.embedding_dimension == 768
        
        model_info = service.get_model_info()
        assert model_info["model_name"] == "custom-model"
        assert model_info["model_version"] == "2.1.0"
        assert model_info["dimension"] == "768"

    def test_embed_query_deterministic(self) -> None:
        """Test that embedding generation is deterministic."""
        service = FakeEmbeddingService()
        
        query = "test query for embeddings"
        embedding1 = service.embed_query(query)
        embedding2 = service.embed_query(query)
        
        assert embedding1 == embedding2
        assert len(embedding1) == 384

    def test_embed_query_different_inputs(self) -> None:
        """Test that different inputs produce different embeddings."""
        service = FakeEmbeddingService()
        
        embedding1 = service.embed_query("query one")
        embedding2 = service.embed_query("query two")
        
        assert embedding1 != embedding2
        assert len(embedding1) == len(embedding2) == 384

    def test_embed_query_normalization(self) -> None:
        """Test that embeddings are normalized to unit vectors."""
        service = FakeEmbeddingService()
        
        embedding = service.embed_query("test normalization")
        
        # Calculate magnitude (should be close to 1.0 for unit vector)
        magnitude = sum(x * x for x in embedding) ** 0.5
        assert abs(magnitude - 1.0) < 1e-6

    def test_embed_texts_multiple(self) -> None:
        """Test embedding multiple texts at once."""
        service = FakeEmbeddingService()
        
        texts = ["first text", "second text", "third text"]
        embeddings = service.embed_texts(texts)
        
        assert len(embeddings) == 3
        assert all(len(emb) == 384 for emb in embeddings)
        
        # Should be deterministic
        embeddings2 = service.embed_texts(texts)
        assert embeddings == embeddings2
        
        # Different texts should produce different embeddings
        assert embeddings[0] != embeddings[1]
        assert embeddings[1] != embeddings[2]

    def test_embed_texts_consistency_with_embed_query(self) -> None:
        """Test that embed_texts produces same results as embed_query."""
        service = FakeEmbeddingService()
        
        text = "consistency test"
        
        single_embedding = service.embed_query(text)
        batch_embeddings = service.embed_texts([text])
        
        assert len(batch_embeddings) == 1
        assert single_embedding == batch_embeddings[0]

    def test_embed_query_error_cases(self) -> None:
        """Test error handling for invalid inputs."""
        service = FakeEmbeddingService()
        
        # Non-string input
        with pytest.raises(ValueError, match="Query must be a string"):
            service.embed_query(123)  # type: ignore
        
        # Empty query
        with pytest.raises(ValueError, match="Query cannot be empty"):
            service.embed_query("")
        
        # Whitespace-only query
        with pytest.raises(ValueError, match="Query cannot be empty"):
            service.embed_query("   \n\t   ")

    def test_embed_texts_error_cases(self) -> None:
        """Test error handling for invalid text lists."""
        service = FakeEmbeddingService()
        
        # Empty list
        with pytest.raises(ValueError, match="Cannot embed empty text list"):
            service.embed_texts([])
        
        # Non-string in list
        with pytest.raises(ValueError, match="Text must be a string"):
            service.embed_texts(["valid text", 123])  # type: ignore

    def test_custom_embedding_dimension(self) -> None:
        """Test that custom embedding dimensions work correctly."""
        for dimension in [128, 256, 512, 1024]:
            service = FakeEmbeddingService(embedding_dimension=dimension)
            
            embedding = service.embed_query("test dimension")
            assert len(embedding) == dimension
            assert service.embedding_dimension == dimension


class TestDeterministicEmbeddingService:
    """Tests for DeterministicEmbeddingService."""

    def test_initialization_with_defaults(self) -> None:
        """Test default initialization."""
        service = DeterministicEmbeddingService()
        
        assert service.embedding_dimension == 384
        
        model_info = service.get_model_info()
        assert model_info["model_name"] == "deterministic-model"
        assert model_info["type"] == "deterministic"
        assert model_info["predefined_count"] == "0"

    def test_predefined_embeddings(self) -> None:
        """Test using predefined embeddings."""
        predefined = {
            "hello": [0.1, 0.2, 0.3],
            "world": [0.4, 0.5, 0.6],
        }
        
        service = DeterministicEmbeddingService(
            embedding_dimension=3,
            predefined_embeddings=predefined,
        )
        
        # Should return predefined embeddings
        assert service.embed_query("hello") == [0.1, 0.2, 0.3]
        assert service.embed_query("world") == [0.4, 0.5, 0.6]
        
        # Should fall back to generated embedding for unknown text
        unknown_embedding = service.embed_query("unknown")
        assert len(unknown_embedding) == 3
        assert unknown_embedding != [0.1, 0.2, 0.3]

    def test_add_predefined_embedding(self) -> None:
        """Test adding predefined embeddings dynamically."""
        service = DeterministicEmbeddingService(embedding_dimension=3)
        
        # Add a predefined embedding
        service.add_predefined_embedding("test", [0.7, 0.8, 0.9])
        
        # Should return the predefined embedding
        assert service.embed_query("test") == [0.7, 0.8, 0.9]
        
        # Model info should reflect the addition
        model_info = service.get_model_info()
        assert model_info["predefined_count"] == "1"

    def test_predefined_embedding_validation(self) -> None:
        """Test validation of predefined embeddings."""
        # Wrong dimension in constructor
        with pytest.raises(ValueError, match="expected 3"):
            DeterministicEmbeddingService(
                embedding_dimension=3,
                predefined_embeddings={"test": [0.1, 0.2, 0.3, 0.4]},  # 4 dims
            )
        
        # Wrong dimension when adding
        service = DeterministicEmbeddingService(embedding_dimension=3)
        with pytest.raises(ValueError, match="doesn't match expected 3"):
            service.add_predefined_embedding("test", [0.1, 0.2])  # 2 dims

    def test_embed_texts_with_predefined(self) -> None:
        """Test batch embedding with predefined embeddings."""
        service = DeterministicEmbeddingService(
            embedding_dimension=3,
            predefined_embeddings={"hello": [0.1, 0.2, 0.3]},
        )
        
        texts = ["hello", "world"]
        embeddings = service.embed_texts(texts)
        
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]  # Predefined
        assert len(embeddings[1]) == 3  # Generated

    def test_deterministic_generation(self) -> None:
        """Test that fallback generation is deterministic."""
        service = DeterministicEmbeddingService()
        
        text = "deterministic test"
        embedding1 = service.embed_query(text)
        embedding2 = service.embed_query(text)
        
        assert embedding1 == embedding2

    def test_generated_embeddings_are_normalized(self) -> None:
        """Test that generated embeddings are normalized."""
        service = DeterministicEmbeddingService()
        
        embedding = service.embed_query("normalization test")
        
        # Calculate magnitude
        magnitude = sum(x * x for x in embedding) ** 0.5
        assert abs(magnitude - 1.0) < 1e-6

    def test_error_handling(self) -> None:
        """Test error handling."""
        service = DeterministicEmbeddingService()
        
        # Same error cases as FakeEmbeddingService
        with pytest.raises(ValueError, match="Query must be a string"):
            service.embed_query(123)  # type: ignore
        
        with pytest.raises(ValueError, match="Cannot embed empty text list"):
            service.embed_texts([])


class TestEmbeddingServiceCompatibility:
    """Test that both fake services work as drop-in replacements."""

    @pytest.mark.parametrize(
        "service_class",
        [FakeEmbeddingService, DeterministicEmbeddingService],
    )
    def test_protocol_compliance(self, service_class) -> None:
        """Test that both services implement the protocol correctly."""
        service = service_class()
        
        # Test all protocol methods
        assert isinstance(service.embedding_dimension, int)
        assert service.embedding_dimension > 0
        
        embedding = service.embed_query("test")
        assert isinstance(embedding, list)
        assert len(embedding) == service.embedding_dimension
        assert all(isinstance(x, float) for x in embedding)
        
        embeddings = service.embed_texts(["test1", "test2"])
        assert isinstance(embeddings, list)
        assert len(embeddings) == 2
        assert all(len(emb) == service.embedding_dimension for emb in embeddings)
        
        model_info = service.get_model_info()
        assert isinstance(model_info, dict)
        assert "model_name" in model_info