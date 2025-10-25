import os
import json
from collections import Counter
from typing import List, Dict, Optional, Any
import matplotlib.pyplot as plt

from .review import Review
from .corpus_document import CorpusDocument
from .game_corpus import GameCorpus


class Corpus:
    """Corpus as a collection of GameCorpus objects."""
    def __init__(self, games: Optional[List[GameCorpus]] = None):
        self.games: List[GameCorpus] = games if games else []

    # vista plana de documentos (compute-on-demand)
    @property
    def documents(self) -> List[CorpusDocument]:
        return [doc for g in self.games for doc in g.documents]

    # -------------------------
    # Constructors / I/O
    # -------------------------
    @classmethod
    def from_json(cls, path: str) -> "Corpus":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        games: List[GameCorpus] = []

        if isinstance(data, list):
            # Cada elemento de la lista puede tener game_id, metadata y reviews
            for game_obj in data:
                gid = game_obj.get("game_id")
                gmeta = game_obj.get("metadata", {})
                reviews = game_obj.get("reviews", [])

                game_corpus = GameCorpus(game_id=gid, metadata=gmeta, documents=[])

                for r in reviews:
                    rev = Review.from_dict({**r, "game_id": gid})
                    processed = {}
                    if "processed" in r or "patterns" in r or "clean_text" in r:
                        processed["clean_text"] = r.get("clean_text", "")
                        processed["language"] = r.get("language", "unknow")
                        processed.update(r.get("processed", {}))
                        processed["patterns"] = r.get("patterns", {})

                    doc = CorpusDocument(rev, processed if processed else None)
                    game_corpus.add_document(doc)

                games.append(game_corpus)

            return cls(games=games)

        raise ValueError("Formato JSON no reconocido para Corpus.")


    def to_list_of_games(self) -> List[Dict[str, Any]]:
        return [g.to_dict() for g in self.games]

    def to_json(self, path: str, list_format: bool = True):
        out = self.to_list_of_games() if list_format else {
            "metadata": {"games": {g.game_id: g.metadata for g in self.games}},
            "reviews": [d.to_dict() for d in self.documents]
            }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    # -------------------------
    # Helpers: selección (trabaja sobre la vista plana)
    # -------------------------
    def _select(self, game_ids: Optional[List[Any]] = None, categories: Optional[List[str]] = None,
                labels: Optional[List[str]] = None) -> List[CorpusDocument]:
        docs = self.documents
        if game_ids is not None:
            docs = [d for d in docs if d.game_id in game_ids]
        if categories is not None:
            docs = [d for d in docs if d.category in categories]
        return docs

    # -------------------------
    # NLTK-like / exploración (la mayoría de los métodos se mantienen)
    # -------------------------
    def game_ids(self, categories: Optional[List[str]] = None, labels: Optional[List[str]] = None, game_ids: Optional[List[Any]] = None):
        docs = self._select(categories=categories, labels=labels, game_ids=game_ids)
        return [d.game_id for d in docs]

    def categories(self, game_ids: Optional[List[Any]] = None, labels: Optional[List[str]] = None):
        docs = self._select(game_ids=game_ids, labels=labels)
        return list(sorted(set(d.category for d in docs)))

    def raw(self, i: Optional[int] = None):
        if i is None:
            return [doc.raw_text for doc in self.documents]
        return self.documents[i].raw_text

    def raw_join(self, game_ids: Optional[List[Any]] = None, categories: Optional[List[str]] = None,
                 labels: Optional[List[str]] = None) -> str:
        docs = self._select(game_ids=game_ids, categories=categories, labels=labels)
        return "\n".join(d.raw_text for d in docs)

    def words(self, game_ids: Optional[List[Any]] = None, categories: Optional[List[str]] = None,
              labels: Optional[List[str]] = None) -> List[str]:
        docs = self._select(game_ids=game_ids, categories=categories, labels=labels)
        return [tok for d in docs for tok in d.processed.get("tokens", [])]

    def sents(self, i: Optional[int] = None):
        if i is None:
            return [doc.processed.get("sentences", []) for doc in self.documents]
        return self.documents[i].processed.get("sentences", [])

    def label_distribution(self, by_game_id: Optional[Any] = None):
        if by_game_id is not None:
            docs = [d for d in self.documents if getattr(d.review, "game_id", None) == by_game_id]
        else:
            docs = self.documents
        dist = Counter(d.review.label for d in docs if d.review.label is not None)
        return dict(dist)

    def documents_by_label(self, label: str) -> List[CorpusDocument]:
        return [d for d in self.documents if d.review.label == label]

    def games_list(self) -> List[Any]:
        return sorted([g.game_id for g in self.games if g.game_id is not None])

    def documents_by_game(self, game_id: Any) -> List[CorpusDocument]:
        g = next((gg for gg in self.games if gg.game_id == game_id), None)
        return g.documents if g else []

    # -------------------------
    # File-like utilities
    # -------------------------
    def abspath(self, game_id):
        return os.path.abspath(str(game_id))

    def encoding(self, game_id):
        return "utf-8"

    def open(self, game_id):
        for doc in self.documents:
            if doc.game_id == game_id:
                from io import StringIO
                return StringIO(doc.raw_text)
        raise FileNotFoundError(f"No existe el archivo con id: {game_id}")

    def readme(self, game_id: Optional[Any] = None):
        if game_id:
            g = next((gg for gg in self.games if gg.game_id == game_id), None)
            return g.metadata.get("readme", "No README available") if g else "No README available"
        if len(self.games) == 1:
            return self.games[0].metadata.get("readme", "No README available")
        return "No README available"

    # -------------------------
    # Stats/utils (operan sobre la vista plana o sobre metadata por juego)
    # -------------------------
    def ratings(self) -> List[Optional[float]]:
        return [doc.review.rating for doc in self.documents]

    def users(self) -> List[str]:
        return [doc.review.username for doc in self.documents]

    def filter_by_language(self, lang: str) -> List[CorpusDocument]:
        return [doc for doc in self.documents if doc.language == lang]

    def filter_by_game(self, game_ids: List[Any]) -> List[CorpusDocument]:
        return [doc for doc in self.documents if doc.review.game_id in game_ids]

    def filter_by_label(self, labels: List[str]) -> List[CorpusDocument]:
        return [doc for doc in self.documents if doc.review.label in labels]

    # -------------------------
    # NLTK-like helpers (copiados para compatibilidad)
    # -------------------------
    def contexts(self, word: str, window: int = 5, game_ids: Optional[List[Any]] = None, categories: Optional[List[str]] = None):
        docs = self._select(game_ids=game_ids, categories=categories)
        contexts = []
        for doc in docs:
            tokens = doc.processed.get("tokens", [])
            for i, tok in enumerate(tokens):
                if tok.lower() == word.lower():
                    left = tokens[max(0, i - window):i]
                    right = tokens[i+1:i+1+window]
                    contexts.append((" ".join(left), tok, " ".join(right)))
        return contexts

    def common_contexts(self, words: List[str], window: int = 1, game_ids: Optional[List[Any]] = None, categories: Optional[List[str]] = None):
        if len(words) < 2:
            raise ValueError("Se requieren al menos dos palabras")
        docs = self._select(game_ids=game_ids, categories=categories)
        contexts = Counter()
        lower_words = [w.lower() for w in words]
        for doc in docs:
            tokens = doc.processed.get("tokens", [])
            for i, tok in enumerate(tokens):
                if tok.lower() in lower_words:
                    left = tokens[max(0, i-window):i]
                    right = tokens[i+1:i+1+window]
                    context = " ".join(left + ["_"] + right)
                    contexts[context] += 1
        return contexts.most_common()

    def lexical_dispersion_plot(self, words: List[str], game_ids: Optional[List[Any]] = None, categories: Optional[List[str]] = None):
        docs = self._select(game_ids=game_ids, categories=categories)
        all_tokens = []
        for doc in docs:
            all_tokens.extend(doc.processed.get("tokens", []))
        points = []
        for word in words:
            for i, tok in enumerate(all_tokens):
                if tok.lower() == word.lower():
                    points.append((i, word))
        if not points:
            print("No se encontraron ocurrencias de las palabras indicadas.")
            return
        x, y = zip(*[(i, words.index(w)) for i, w in points])
        plt.figure(figsize=(12, 6))
        plt.plot(x, y, "b|", scalex=0.1)
        plt.yticks(range(len(words)), words)
        plt.xlabel("Posición en el corpus")
        plt.title("Dispersión léxica")
        plt.show()

    def frequency_distribution(self, game_ids: Optional[List[Any]] = None, categories: Optional[List[str]] = None) -> Counter:
        docs = self._select(game_ids=game_ids, categories=categories)
        all_tokens = [tok.lower() for doc in docs for tok in doc.processed.get("tokens", [])]
        return Counter(all_tokens)

    def most_common(self, n: int = 20, game_ids: Optional[List[Any]] = None, categories: Optional[List[str]] = None):
        return self.frequency_distribution(game_ids, categories).most_common(n)

    def hapaxes(self, game_ids: Optional[List[Any]] = None, categories: Optional[List[str]] = None):
        fdist = self.frequency_distribution(game_ids, categories)
        return [word for word, freq in fdist.items() if freq == 1]

    def word_length_distribution(self, game_ids: Optional[List[Any]] = None, categories: Optional[List[str]] = None):
        docs = self._select(game_ids=game_ids, categories=categories)
        lengths = [len(tok) for doc in docs for tok in doc.processed.get("tokens", [])]
        return Counter(lengths)

    def plot_word_length_distribution(self, game_ids: Optional[List[Any]] = None, categories: Optional[List[str]] = None):
        dist = self.word_length_distribution(game_ids, categories)
        plt.figure(figsize=(10, 5))
        plt.bar(list(dist.keys()), list(dist.values()))
        plt.xlabel("Longitud de palabra")
        plt.ylabel("Frecuencia")
        plt.title("Distribución de longitudes de palabra")
        plt.show()

    def ngrams(self, n: int = 2, game_ids: Optional[List[Any]] = None, categories: Optional[List[str]] = None):
        docs = self._select(game_ids=game_ids, categories=categories)
        tokens = [tok.lower() for doc in docs for tok in doc.processed.get("tokens", [])]
        if not tokens or n <= 0:
            return []
        return list(zip(*[tokens[i:] for i in range(n)]))

    def bigrams(self, game_ids: Optional[List[Any]] = None, categories: Optional[List[str]] = None):
        return self.ngrams(2, game_ids, categories)

    def trigrams(self, game_ids: Optional[List[Any]] = None, categories: Optional[List[str]] = None):
        return self.ngrams(3, game_ids, categories)

    def collocations(self, top_n: int = 20, game_ids: Optional[List[Any]] = None, categories: Optional[List[str]] = None):
        bigrams = self.bigrams(game_ids, categories)
        counts = Counter(bigrams)
        return counts.most_common(top_n)

    def lexical_diversity(self, tokens: List[str]) -> float:
        return len(set(tokens)) / len(tokens) if tokens else 0.0

    def category_stats(self):
        stats = []
        for cat in self.categories():
            tokens = self.words(categories=[cat])
            types = set(tokens)
            div = len(types) / len(tokens) if tokens else 0
            stats.append((cat, len(tokens), len(types), div))
        return stats

    def print_category_stats(self):
        print(f"{'Category':<15}{'Tokens':<10}{'Types':<10}{'LexDiv':<10}")
        for cat, tokens, types, div in self.category_stats():
            print(f"{cat:<15}{tokens:<10}{types:<10}{div:.3f}")

    def review_counts_by_category(self):
        counts = Counter()
        for game in self.games:
            for doc in game.documents:
                cat = getattr(doc.review, 'category', None)
                if cat:
                    counts[cat] += 1
        return counts

    def print_review_counts(self):
        counts = self.review_counts_by_category()
        print(f"{'Category':<15}{'Num Reviews':<10}")
        for cat, n in counts.items():
            print(f"{cat:<15}{n:<10}")

    def plot_frequency_distribution(self, n: int = 30, game_ids: Optional[List[Any]] = None, categories: Optional[List[str]] = None,
                                    cumulative: bool = False, title: Optional[str] = None):
        fdist = self.frequency_distribution(game_ids, categories)
        most_common = fdist.most_common(n)
        if not most_common:
            print("No hay palabras para plotear.")
            return
        words, freqs = zip(*most_common)
        plt.figure(figsize=(12, 6))
        if cumulative:
            plt.plot(range(len(freqs)), [sum(freqs[:i+1]) for i in range(len(freqs))], marker="o")
        else:
            plt.bar(words, freqs)
        plt.title(title or "Frecuencia de palabras")
        plt.xlabel("Palabras")
        plt.ylabel("Frecuencia")
        plt.xticks(rotation=45)
        plt.show()

    # -------------------------
    # Estadísticas agregadas / por juego
    # -------------------------
    def _stats_field(self, field: str, game_id: Optional[Any] = None) -> int:
        alt_field = field.replace("reviews", "review") if "reviews" in field else field

        def _get_from_meta(meta: dict):
            if not isinstance(meta, dict):
                return 0
            stats = meta.get("stats", {}) or {}
            val = stats.get(field, stats.get(alt_field, 0))
            try:
                return int(val)
            except Exception:
                try:
                    return int(float(val))
                except Exception:
                    return 0

        if game_id is not None:
            g = next((gg for gg in self.games if gg.game_id == game_id), None)
            return _get_from_meta(g.metadata if g else {})

        total = 0
        for g in self.games:
            total += _get_from_meta(g.metadata if g else {})
        return total

    def num_reviews(self, game_id: Optional[Any] = None) -> int:
        return self._stats_field("num_reviews", game_id)

    def num_reviews_commented(self, game_id: Optional[Any] = None) -> int:
        return self._stats_field("num_reviews_commented", game_id)

    def num_reviews_rated(self, game_id: Optional[Any] = None) -> int:
        return self._stats_field("num_reviews_rated", game_id)

    def num_reviews_rated_and_commented(self, game_id: Optional[Any] = None) -> int:
        return self._stats_field("num_reviews_rated_and_commented", game_id)

    def rating_distribution(self, game_id: Optional[Any] = None):
        docs = self.documents_by_game(game_id) if game_id is not None else self.documents
        ratings = [d.review.rating for d in docs if d.review.rating not in (None, "N/A")]
        return dict(Counter(ratings))

    def unique_users(self) -> List[str]:
        return list(set(d.review.username for d in self.documents if d.review))

    def num_unique_users(self) -> int:
        return len(self.unique_users())

    def all_users(self) -> List[str]:
        return [d.review.username for d in self.documents if d.review]

    def no_unique_users(self) -> List[str]:
        counts = Counter(self.all_users())
        return [user for user, c in counts.items() if c > 1]

    def num_no_unique_users(self) -> int:
        return len(self.no_unique_users())
    
    def all_classifications(self, field: str = "mechanics") -> List[str]:
        """
        Devuelve una lista plana con todos los valores de classifications de los juegos.
        field: 'mechanics', 'categories' o 'families'
        """
        if field not in ("mechanics", "categories", "families"):
            raise ValueError("field debe ser 'mechanics', 'categories' o 'families'")
        all_vals = []
        for g in self.games:
            classif = g.metadata.get("classifications", {}) or {}
            values = classif.get(field, [])
            if isinstance(values, list):
                all_vals.extend(values)
            elif isinstance(values, str):
                all_vals.append(values)
        return sorted(set(all_vals))
    
    def classifications_by_game(self, game_id: Any) -> Dict[str, List[str]]:
        """
        Devuelve las classifications de un juego concreto.
        """
        g = next((gg for gg in self.games if gg.game_id == game_id), None)
        if not g:
            return {"mechanics": [], "categories": [], "families": []}
        classif = g.metadata.get("classifications", {}) or {}
        return {
            "mechanics": classif.get("mechanics", []),
            "categories": classif.get("categories", []),
            "families": classif.get("families", [])
        }

    # -------------------------
    # Métodos por metadata de juego
    # -------------------------
    def metadata_for(self, game_id: Any) -> Dict[str, Any]:
        g = next((gg for gg in self.games if gg.game_id == game_id), None)
        return g.metadata if g else {}

    def overall_rank(self, game_id: Optional[Any] = None):
        meta = self.metadata_for(game_id) if game_id is not None else (self.games[0].metadata if len(self.games) == 1 else {})
        return meta.get("rankings", {}).get("overall_rank") if meta else None

    def strategy_rank(self, game_id: Optional[Any] = None):
        meta = self.metadata_for(game_id) if game_id is not None else (self.games[0].metadata if len(self.games) == 1 else {})
        return meta.get("rankings", {}).get("strategygames_rank") if meta else None

    def complexity_poll_avg(self, game_id: Optional[Any] = None):
        meta = self.metadata_for(game_id) if game_id is not None else (self.games[0].metadata if len(self.games) == 1 else {})
        return meta.get("polls", {}).get("complexity_poll", {}).get("poll_avg") if meta else None

    def avg_rating(self, game_id: Optional[Any] = None):
        meta = self.metadata_for(game_id) if game_id is not None else (self.games[0].metadata if len(self.games) == 1 else {})
        return meta.get("stats", {}).get("avg_rating") if meta else None

    def bayes_average(self, game_id: Optional[Any] = None):
        meta = self.metadata_for(game_id) if game_id is not None else (self.games[0].metadata if len(self.games) == 1 else {})
        return meta.get("stats", {}).get("bayes_average") if meta else None
    
    # Interfaz con MongoDB
    from ..storage import MongoCorpusStorage
    def save_to_mongo(self, storage: MongoCorpusStorage, verbose: bool = True):
        """Guarda el corpus en MongoDB."""
        if not storage:
            raise ValueError("Se requiere una instancia de MongoCorpusStorage")
        storage.save_corpus(self, verbose=verbose)
    
    @classmethod
    def load_from_mongo(cls, storage: MongoCorpusStorage,
                        game_ids: Optional[List[int]] = None,
                        categories: Optional[List[str]] = None,
                        limit: Optional[int] = None,
                        verbose: bool = True) -> "Corpus":
        """Carga un corpus desde MongoDB."""
        if not storage:
            raise ValueError("Se requiere una instancia de MongoCorpusStorage")
        return storage.load_corpus(
            game_ids=game_ids,
            categories=categories,
            limit=limit,
            verbose=verbose
        )
        
    def iter_reviews_from_mongo(self, storage, query=None, batch_size=500):
        cursor = storage.reviews_collection.find(query or {}).batch_size(batch_size)
        for doc in cursor:
            yield storage._mongo_to_corpus_document(doc)