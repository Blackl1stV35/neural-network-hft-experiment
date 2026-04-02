"""Sentiment pipeline: news fetching, FinBERT embedding extraction."""

from __future__ import annotations

import json
import os
import time
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from loguru import logger


class NewsAPIFetcher:
    """Fetch gold/forex related news from NewsAPI (free tier: 100 req/day)."""

    BASE_URL = "https://newsapi.org/v2/everything"
    GOLD_KEYWORDS = [
        "gold price", "XAUUSD", "gold futures", "precious metals",
        "Federal Reserve", "FOMC", "inflation", "treasury yields",
        "dollar index", "DXY", "safe haven", "gold market",
    ]

    def __init__(self, api_key: Optional[str] = None, cache_dir: str = "data/sentiment_cache"):
        self.api_key = api_key or os.getenv("NEWSAPI_KEY")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not self.api_key:
            logger.warning("NEWSAPI_KEY not set. Sentiment pipeline will use cached data only.")

    def fetch(
        self,
        hours_back: int = 24,
        max_articles: int = 20,
    ) -> list[dict]:
        """Fetch recent news articles about gold/forex."""
        if not self.api_key:
            return self._load_cache()

        query = " OR ".join(f'"{kw}"' for kw in self.GOLD_KEYWORDS[:5])
        from_date = (datetime.utcnow() - timedelta(hours=hours_back)).strftime("%Y-%m-%dT%H:%M:%S")

        try:
            resp = requests.get(
                self.BASE_URL,
                params={
                    "q": query,
                    "from": from_date,
                    "sortBy": "publishedAt",
                    "language": "en",
                    "pageSize": max_articles,
                    "apiKey": self.api_key,
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            articles = data.get("articles", [])

            # Extract relevant fields
            processed = []
            for a in articles:
                text = f"{a.get('title', '')}. {a.get('description', '')}"
                processed.append({
                    "text": text.strip(),
                    "source": a.get("source", {}).get("name", "unknown"),
                    "published_at": a.get("publishedAt", ""),
                    "url": a.get("url", ""),
                })

            self._save_cache(processed)
            logger.info(f"Fetched {len(processed)} news articles")
            return processed

        except Exception as e:
            logger.warning(f"News fetch failed: {e}. Using cache.")
            return self._load_cache()

    def _cache_path(self) -> Path:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        return self.cache_dir / f"news_{today}.json"

    def _save_cache(self, articles: list[dict]) -> None:
        with open(self._cache_path(), "w") as f:
            json.dump(articles, f, indent=2)

    def _load_cache(self) -> list[dict]:
        path = self._cache_path()
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return []


class FinBERTSentiment:
    """Extract rich sentiment embeddings from FinBERT.

    Instead of just positive/negative/neutral labels, extracts intermediate
    hidden state representations for richer sentiment features.
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: str = "cpu",
        max_length: int = 512,
        cache_dir: str = "data/sentiment_cache",
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._tokenizer = None

    def _load_model(self) -> None:
        """Lazy-load model and tokenizer."""
        if self._model is not None:
            return

        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        logger.info(f"Loading FinBERT model: {self.model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, output_hidden_states=True
        )
        self._model.to(self.device)
        self._model.eval()
        logger.info("FinBERT loaded successfully")

    def get_embeddings(
        self,
        texts: list[str],
        use_hidden_states: bool = True,
        layer: int = -2,
    ) -> np.ndarray:
        """Extract embeddings from FinBERT.

        Args:
            texts: List of text strings.
            use_hidden_states: If True, use intermediate hidden states (richer).
                             If False, use classification logits (simpler).
            layer: Which hidden layer to use (-1 = last, -2 = second to last).

        Returns:
            Embeddings array. Shape: (n_texts, 768) if hidden states, (n_texts, 3) if logits.
        """
        self._load_model()
        import torch

        cache_key = self._cache_key(texts, use_hidden_states, layer)
        cached = self._load_embedding_cache(cache_key)
        if cached is not None:
            return cached

        embeddings = []
        batch_size = 8

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                inputs = self._tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                ).to(self.device)

                outputs = self._model(**inputs)

                if use_hidden_states and outputs.hidden_states:
                    # Mean-pool the chosen hidden layer
                    hidden = outputs.hidden_states[layer]
                    attention_mask = inputs["attention_mask"].unsqueeze(-1)
                    pooled = (hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
                    embeddings.append(pooled.cpu().numpy())
                else:
                    # Use softmax logits
                    logits = torch.softmax(outputs.logits, dim=-1)
                    embeddings.append(logits.cpu().numpy())

        result = np.concatenate(embeddings, axis=0).astype(np.float32)
        self._save_embedding_cache(cache_key, result)

        logger.info(f"Generated embeddings: shape={result.shape}")
        return result

    def get_consensus_embedding(
        self,
        texts: list[str],
        use_hidden_states: bool = True,
    ) -> np.ndarray:
        """Get a single consensus embedding from multiple news articles.

        Averages all article embeddings into one vector, weighted by recency.
        """
        if not texts:
            dim = 768 if use_hidden_states else 3
            return np.zeros(dim, dtype=np.float32)

        embeddings = self.get_embeddings(texts, use_hidden_states)
        # Simple average (could add recency weighting later)
        consensus = embeddings.mean(axis=0)
        return consensus

    def _cache_key(self, texts: list[str], use_hidden: bool, layer: int) -> str:
        content = json.dumps(texts, sort_keys=True) + f"_{use_hidden}_{layer}"
        return hashlib.md5(content.encode()).hexdigest()

    def _load_embedding_cache(self, key: str) -> Optional[np.ndarray]:
        path = self.cache_dir / f"emb_{key}.npy"
        if path.exists():
            return np.load(str(path))
        return None

    def _save_embedding_cache(self, key: str, arr: np.ndarray) -> None:
        path = self.cache_dir / f"emb_{key}.npy"
        np.save(str(path), arr)


class SentimentService:
    """Orchestrates news fetching + embedding generation on a schedule."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_device: str = "cpu",
        update_interval: int = 900,
    ):
        self.news_fetcher = NewsAPIFetcher(api_key=api_key)
        self.sentiment_model = FinBERTSentiment(device=model_device)
        self.update_interval = update_interval
        self._last_update = 0.0
        self._cached_embedding: Optional[np.ndarray] = None

    def get_current_sentiment(self, force: bool = False) -> np.ndarray:
        """Get current consensus sentiment embedding, refreshing if stale."""
        now = time.time()
        if not force and self._cached_embedding is not None:
            if now - self._last_update < self.update_interval:
                return self._cached_embedding

        articles = self.news_fetcher.fetch()
        texts = [a["text"] for a in articles if a["text"]]

        if not texts:
            logger.warning("No news articles available, using zero embedding")
            self._cached_embedding = np.zeros(768, dtype=np.float32)
        else:
            self._cached_embedding = self.sentiment_model.get_consensus_embedding(texts)

        self._last_update = now
        return self._cached_embedding
