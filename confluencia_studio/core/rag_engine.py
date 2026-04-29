"""RAG (Retrieval-Augmented Generation) Engine for ConfluenciaStudio.

Indexes documentation, API references, and knowledge bases to provide
context-aware responses for the LLM assistant.
"""

from __future__ import annotations

import os
import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class DocumentChunk:
    """A chunk of indexed document."""
    id: str
    content: str
    source: str  # file path or URL
    title: str
    chunk_index: int
    embedding: Optional[List[float]] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'content': self.content,
            'source': self.source,
            'title': self.title,
            'chunk_index': self.chunk_index,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'DocumentChunk':
        return cls(
            id=d['id'],
            content=d['content'],
            source=d['source'],
            title=d['title'],
            chunk_index=d['chunk_index'],
            embedding=d.get('embedding'),
            metadata=d.get('metadata', {}),
        )


@dataclass
class SearchResult:
    """Result from RAG search."""
    chunk: DocumentChunk
    score: float  # similarity score (higher = more relevant)

    def to_dict(self) -> Dict:
        return {
            'chunk': self.chunk.to_dict(),
            'score': self.score,
        }


class SimpleEmbedding:
    """Simple TF-IDF based embedding for MVP (no external dependencies).

    For production, replace with sentence-transformers or OpenAI embeddings.
    """

    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_count = 0

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        text = text.lower()
        tokens = re.findall(r'\b[a-z0-9_]{2,}\b', text)
        return tokens

    def fit(self, documents: List[str]):
        """Build vocabulary and IDF from documents."""
        self.doc_count = len(documents)
        doc_freq: Dict[str, int] = {}

        for doc in documents:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                doc_freq[token] = doc_freq.get(token, 0) + 1

        # Build vocab and IDF
        for token, freq in doc_freq.items():
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
            self.idf[token] = max(1.0, self.doc_count / (freq + 1))

    def encode(self, text: str) -> List[float]:
        """Encode text to TF-IDF vector."""
        tokens = self._tokenize(text)
        tf: Dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1

        # Create sparse vector (but return as dense for simplicity)
        vec = [0.0] * len(self.vocab)
        for token, count in tf.items():
            if token in self.vocab:
                idx = self.vocab[token]
                vec[idx] = count * self.idf.get(token, 1.0)

        # Normalize
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]

        return vec

    def similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity."""
        if len(vec1) != len(vec2):
            return 0.0
        dot = sum(a * b for a, b in zip(vec1, vec2))
        return dot  # Already normalized


class RAGEngine:
    """Retrieval-Augmented Generation engine for LLM context injection.

    Features:
    - Document chunking and indexing
    - Simple TF-IDF embeddings (upgradeable to transformer embeddings)
    - Semantic search
    - Context prompt building

    Usage:
        rag = RAGEngine()
        rag.index_directory("docs/")
        rag.index_confluencia_api()
        results = rag.search("how to train a drug model")
        context = rag.build_context_prompt("how to train a drug model")
    """

    def __init__(self, persist_dir: str = "~/.confluencia/rag"):
        self.persist_dir = Path(persist_dir).expanduser()
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.chunks: List[DocumentChunk] = []
        self.embeddings: Dict[str, List[float]] = {}  # chunk_id -> embedding
        self.embedding_model = SimpleEmbedding()

        # Confluencia API documentation (built-in)
        self._api_docs: Dict[str, str] = {}

        # Load persisted state
        self._load_state()

    def index_directory(self, docs_dir: str, recursive: bool = True) -> int:
        """Index all documentation files in a directory.

        Supported formats: .md, .rst, .txt, .py (docstrings)

        Args:
            docs_dir: Directory path
            recursive: Whether to search recursively

        Returns:
            Number of chunks indexed
        """
        docs_path = Path(docs_dir)
        if not docs_path.exists():
            return 0

        patterns = ['*.md', '*.rst', '*.txt']
        if recursive:
            patterns = [f'**/{p}' for p in patterns]

        files = []
        for pattern in patterns:
            files.extend(docs_path.glob(pattern))

        chunks_added = 0
        for file_path in files:
            chunks_added += self._index_file(file_path)

        # Re-train embeddings
        self._rebuild_embeddings()
        self._save_state()

        return chunks_added

    def _index_file(self, file_path: Path) -> int:
        """Index a single file."""
        try:
            content = file_path.read_text(encoding='utf-8')
        except Exception:
            return 0

        # Split into chunks
        chunks = self._chunk_text(content, str(file_path), file_path.stem)
        for chunk in chunks:
            self.chunks.append(chunk)

        return len(chunks)

    def _chunk_text(self, text: str, source: str, title: str, chunk_size: int = 500, overlap: int = 50) -> List[DocumentChunk]:
        """Split text into overlapping chunks."""
        # Split by paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)

        chunks = []
        current_chunk = ""
        chunk_index = 0

        for para in paragraphs:
            if len(current_chunk) + len(para) > chunk_size and current_chunk:
                # Save current chunk
                chunk_id = hashlib.md5(f"{source}:{chunk_index}".encode()).hexdigest()[:12]
                chunks.append(DocumentChunk(
                    id=chunk_id,
                    content=current_chunk.strip(),
                    source=source,
                    title=title,
                    chunk_index=chunk_index,
                ))
                chunk_index += 1
                # Keep overlap
                current_chunk = current_chunk[-overlap:] + "\n\n" + para
            else:
                current_chunk += "\n\n" + para

        # Final chunk
        if current_chunk.strip():
            chunk_id = hashlib.md5(f"{source}:{chunk_index}".encode()).hexdigest()[:12]
            chunks.append(DocumentChunk(
                id=chunk_id,
                content=current_chunk.strip(),
                source=source,
                title=title,
                chunk_index=chunk_index,
            ))

        return chunks

    def index_confluencia_api(self) -> int:
        """Index Confluencia API documentation (built-in)."""
        # Built-in API documentation for Confluencia modules
        api_docs = {
            "drug_train": """
Train a drug efficacy prediction model.

Usage: drug train --data <path> [options]

Parameters:
  --data PATH           Path to training data CSV (required)
  --model-type STR      Model type: rf, xgb, lgbm, mlp, moe (default: moe)
  --n-estimators INT    Number of trees for RF/XGB (default: 100)
  --test-size FLOAT     Test set proportion (default: 0.2)
  --output PATH         Output model path

Features used: Morgan fingerprints + physicochemical descriptors + dose-response features

Example:
  drug train --data data/drug_train.csv --model-type moe --n-estimators 200
""",
            "drug_predict": """
Predict drug efficacy for new molecules.

Usage: drug predict --smiles <SMILES> --model <path>
       drug predict --data <path> --model <path>

Parameters:
  --smiles STR      Single SMILES string to predict
  --data PATH       CSV file with SMILES column
  --model PATH      Path to trained model (required)

Output: Prediction results with efficacy score and uncertainty

Example:
  drug predict --smiles "CC(=O)Oc1ccccc1C(=O)O" --model models/drug_moe.pkl
""",
            "drug_screen": """
Screen molecules for drug efficacy.

Usage: drug screen --data <path> --model <path> [options]

Parameters:
  --data PATH       Molecules to screen (CSV with SMILES column)
  --model PATH      Trained model path
  --top-k INT       Return top K candidates (default: 10)
  --threshold FLOAT Efficacy threshold filter

Output: Ranked list of molecules with predicted efficacy
""",
            "drug_pk": """
Simulate pharmacokinetics using 3-compartment model.

Usage: drug pk --ka <float> --kd <float> [options]

Parameters:
  --ka FLOAT    Absorption rate constant (1/h)
  --kd FLOAT    Distribution rate constant (1/h)
  --ke FLOAT    Elimination rate constant (1/h)
  --dose FLOAT  Dose in mg
  --time FLOAT  Simulation duration in hours

Output: Concentration-time curve data and plot

Example:
  drug pk --ka 0.5 --kd 0.3 --ke 0.1 --dose 200 --time 72
""",
            "epitope_predict": """
Predict epitope-MHC binding affinity.

Usage: epitope predict --epitope <sequence> --allele <HLA> [options]
       epitope predict --data <path> [options]

Parameters:
  --epitope STR   Peptide sequence (8-11 amino acids)
  --allele STR    MHC allele (e.g., HLA-A*02:01)
  --data PATH     CSV with epitope sequences and optional alleles
  --model PATH    Trained model path

Output: Predicted binding affinity (IC50 nM) and binder classification

Example:
  epitope predict --epitope SLYNTVATL --allele HLA-A*02:01
""",
            "epitope_train": """
Train epitope-MHC binding prediction model.

Usage: epitope train --data <path> [options]

Parameters:
  --data PATH        IEDB-format training data
  --model-type STR   Model: rf, xgb, mlp, moe, mamba (default: moe)
  --use-mhc BOOL     Include MHC pseudo-sequence features (recommended)
  --use-esm2 BOOL    Use ESM-2 embeddings (requires download)

The MHC pseudo-sequence encoding captures allele-specific binding patterns.
""",
            "circrna_immune": """
Analyze circRNA immunogenicity.

Usage: circrna immune --data <path> [options]

Parameters:
  --data PATH     circRNA expression data
  --output PATH   Output directory

Output: Immunogenicity scores and cell type predictions
""",
            "joint_evaluate": """
Joint drug-epitope efficacy evaluation.

Usage: joint evaluate --smiles <SMILES> --epitope <seq> --allele <HLA> [options]

Parameters:
  --smiles STR     Drug SMILES
  --epitope STR    Epitope sequence
  --allele STR     MHC allele
  --dose FLOAT     Drug dose in mg

Output: Composite efficacy score with clinical/binding/kinetics breakdown
""",
            "chart_pk": """
Generate PK concentration-time curve.

Usage: chart pk --data <path> [options]

Parameters:
  --data PATH    PK simulation data
  --output PATH  Output plot path
""",
        }

        chunks_added = 0
        for name, doc in api_docs.items():
            chunk_id = hashlib.md5(f"api:{name}".encode()).hexdigest()[:12]
            self.chunks.append(DocumentChunk(
                id=chunk_id,
                content=doc.strip(),
                source="confluencia_api",
                title=f"API: {name}",
                chunk_index=0,
                metadata={'type': 'api', 'name': name},
            ))
            chunks_added += 1

        self._api_docs = api_docs
        self._rebuild_embeddings()
        self._save_state()

        return chunks_added

    def _rebuild_embeddings(self):
        """Rebuild all embeddings."""
        if not self.chunks:
            return

        # Train embedding model on all chunk contents
        documents = [c.content for c in self.chunks]
        self.embedding_model.fit(documents)

        # Encode all chunks
        self.embeddings = {}
        for chunk in self.chunks:
            self.embeddings[chunk.id] = self.embedding_model.encode(chunk.content)

    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        """Search for relevant document chunks.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of SearchResult sorted by relevance
        """
        if not self.chunks or not self.embeddings:
            return []

        # Encode query
        query_vec = self.embedding_model.encode(query)

        # Compute similarities
        results = []
        for chunk in self.chunks:
            if chunk.id not in self.embeddings:
                continue
            chunk_vec = self.embeddings[chunk.id]
            score = self.embedding_model.similarity(query_vec, chunk_vec)
            results.append(SearchResult(chunk=chunk, score=score))

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        return results[:k]

    def build_context_prompt(self, query: str, k: int = 5, max_tokens: int = 2000) -> str:
        """Build context prompt for LLM from search results.

        Args:
            query: User query
            k: Number of chunks to include
            max_tokens: Approximate token limit

        Returns:
            Formatted context string for LLM system prompt
        """
        results = self.search(query, k=k)

        if not results:
            return ""

        context_parts = ["## Relevant Documentation\n"]
        total_chars = 0
        max_chars = max_tokens * 4  # Rough char estimate

        for result in results:
            chunk = result.chunk
            entry = f"\n### {chunk.title} (source: {chunk.source})\n{chunk.content}\n"

            if total_chars + len(entry) > max_chars:
                break

            context_parts.append(entry)
            total_chars += len(entry)

        return "".join(context_parts)

    def _save_state(self):
        """Persist RAG state to disk."""
        state_file = self.persist_dir / "rag_state.json"

        state = {
            'chunks': [c.to_dict() for c in self.chunks],
            'vocab': self.embedding_model.vocab,
            'idf': self.embedding_model.idf,
            'doc_count': self.embedding_model.doc_count,
        }

        # Write atomically
        temp_file = state_file.with_suffix('.tmp')
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
        temp_file.rename(state_file)

    def _load_state(self):
        """Load persisted RAG state."""
        state_file = self.persist_dir / "rag_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)

            self.chunks = [DocumentChunk.from_dict(c) for c in state.get('chunks', [])]
            self.embedding_model.vocab = state.get('vocab', {})
            self.embedding_model.idf = state.get('idf', {})
            self.embedding_model.doc_count = state.get('doc_count', 0)

            # Rebuild embeddings
            self._rebuild_embeddings()

        except Exception:
            pass

    def clear(self):
        """Clear all indexed data."""
        self.chunks = []
        self.embeddings = {}
        self.embedding_model = SimpleEmbedding()

        state_file = self.persist_dir / "rag_state.json"
        if state_file.exists():
            state_file.unlink()


# Global singleton
_rag_engine: Optional[RAGEngine] = None

def get_rag_engine() -> RAGEngine:
    """Get the global RAG engine instance."""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
        # Auto-index API docs
        _rag_engine.index_confluencia_api()
    return _rag_engine
