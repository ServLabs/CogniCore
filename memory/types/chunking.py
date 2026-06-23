"""
Token-Based Chunking (Section 18)

Standard token-based chunking for LFM documents.
Uses tiktoken for accurate token counting.
"""

from typing import Optional


class TokenChunker:
    """
    Token-based document chunking with overlap.
    
    Uses tiktoken for accurate token counting (same as OpenAI models).
    """
    
    def __init__(
        self,
        chunk_size: int = 512,      # tokens per chunk
        overlap: int = 50,           # overlap tokens
        model: str = "cl100k_base",  # tiktoken encoding
    ):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Tokens per chunk.
            overlap: Overlap tokens between chunks.
            model: Tiktoken encoding model.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.model = model
        self._encoder = None
    
    @property
    def encoder(self):
        """Lazy-load tiktoken encoder."""
        if self._encoder is None:
            try:
                import tiktoken
                self._encoder = tiktoken.get_encoding(self.model)
            except ImportError:
                # Fallback to word-based if tiktoken not available
                self._encoder = None
        return self._encoder
    
    def chunk(self, text: str) -> list[dict]:
        """
        Split text into overlapping token-based chunks.
        
        Args:
            text: Input text.
            
        Returns:
            List of chunk dicts with text and token info.
        """
        if self.encoder is None:
            return self._chunk_by_words(text)
        
        tokens = self.encoder.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoder.decode(chunk_tokens)
            
            chunks.append({
                "text": chunk_text,
                "start_token": start,
                "end_token": end,
                "token_count": len(chunk_tokens),
            })
            
            # Move start with overlap
            start = end - self.overlap
            if start >= len(tokens):
                break
        
        return chunks
    
    def _chunk_by_words(self, text: str) -> list[dict]:
        """
        Fallback word-based chunking.
        
        Args:
            text: Input text.
            
        Returns:
            List of chunk dicts.
        """
        words = text.split()
        chunks = []
        
        # Approximate: 1 word ≈ 1.3 tokens
        word_chunk_size = int(self.chunk_size / 1.3)
        word_overlap = int(self.overlap / 1.3)
        
        start = 0
        while start < len(words):
            end = min(start + word_chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            
            chunks.append({
                "text": chunk_text,
                "start_token": start,  # Actually word index
                "end_token": end,
                "token_count": len(chunk_words),  # Actually word count
            })
            
            start = end - word_overlap
            if start >= len(words):
                break
        
        return chunks
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Input text.
            
        Returns:
            Token count.
        """
        if self.encoder is None:
            # Approximate: 1 word ≈ 1.3 tokens
            return int(len(text.split()) * 1.3)
        
        return len(self.encoder.encode(text))
    
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to max tokens.
        
        Args:
            text: Input text.
            max_tokens: Maximum tokens.
            
        Returns:
            Truncated text.
        """
        if self.encoder is None:
            words = text.split()
            max_words = int(max_tokens / 1.3)
            return " ".join(words[:max_words])
        
        tokens = self.encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        return self.encoder.decode(tokens[:max_tokens])


# ══════════════════════════════════════════════════════════════════════════════
# Singleton
# ══════════════════════════════════════════════════════════════════════════════

_chunker: Optional[TokenChunker] = None


def get_chunker(
    chunk_size: int = 512,
    overlap: int = 50,
) -> TokenChunker:
    """Get a TokenChunker instance."""
    global _chunker
    if _chunker is None:
        _chunker = TokenChunker(chunk_size=chunk_size, overlap=overlap)
    return _chunker
