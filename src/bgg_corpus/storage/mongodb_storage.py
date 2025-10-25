from typing import List, Dict, Optional
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import BulkWriteError
from tqdm import tqdm


class MongoCorpusStorage:
    """
    Storage class for saving and loading Corpus data to/from MongoDB.
    """
    
    def __init__(
        self,
        db_name: str = "bgg_corpus",
        host: str = "localhost",
        port: int = 27017,
        username: Optional[str] = None,
        password: Optional[str] = None
    ):
        if username and password:
            self.client = MongoClient(
                host=host,
                port=port,
                username=username,
                password=password
            )
        else:
            self.client = MongoClient(host=host, port=port)
        
        self.db = self.client[db_name]
        self.metadata_collection = self.db['game_metadata']
        self.reviews_collection = self.db['reviews']
        self.username = username
        self.host = host
        self.port = port
        self.db_name = db_name
        self._create_indexes()

    def __str__(self):
        user_info = f"{self.username}@" if self.username else ""
        return f"MongoCorpusStorage({user_info}{self.host}:{self.port}/{self.db_name})"

    def __repr__(self):
        return self.__str__()
    
    def _create_indexes(self):
        """Crea índices para optimizar consultas."""
        self.metadata_collection.create_index([("game_id", ASCENDING)], unique=True)
        self.metadata_collection.create_index([("game_info.name", ASCENDING)])
        self.metadata_collection.create_index([("stats.avg_rating", DESCENDING)])
        self.metadata_collection.create_index([("rankings.overall_rank", ASCENDING)])
        
        self.reviews_collection.create_index([("game_id", ASCENDING)])
        self.reviews_collection.create_index([("username", ASCENDING)])
        self.reviews_collection.create_index([("category", ASCENDING)])
        self.reviews_collection.create_index([("rating", DESCENDING)])
        self.reviews_collection.create_index([("timestamp", DESCENDING)])
        self.reviews_collection.create_index([
            ("game_id", ASCENDING),
            ("category", ASCENDING)
        ])
    
    def save_corpus(self, corpus, verbose: bool = True):
        """Guarda un Corpus completo en MongoDB."""
        if verbose:
            print(f"Guardando corpus en MongoDB...")

        metadata_docs = []
        for game in corpus.games:
            meta_doc = {
                "game_id": game.game_id,
                "metadata": game.metadata,
                "total_reviews": len(game.documents),
                "categories_count": game.count_by_category(),
            }
            metadata_docs.append(meta_doc)
        
        if metadata_docs:
            if verbose:
                print(f"  Guardando {len(metadata_docs)} juegos...")
            
            for doc in tqdm(metadata_docs, disable=not verbose, desc="Metadata"):
                self.metadata_collection.update_one(
                    {"game_id": doc["game_id"]},
                    {"$set": doc},
                    upsert=True
                )
        
        # Guardar reviews
        review_docs = []
        for game in corpus.games:
            for doc in game.documents:
                review_doc = self._corpus_document_to_mongo(doc)
                review_docs.append(review_doc)
        
        if review_docs:
            if verbose:
                print(f"  Guardando {len(review_docs)} reviews...")
            
            batch_size = 1000
            for i in tqdm(range(0, len(review_docs), batch_size), 
                         disable=not verbose, desc="Reviews"):
                batch = review_docs[i:i+batch_size]
                try:
                    self.reviews_collection.insert_many(batch, ordered=False)
                except BulkWriteError as e:
                    if verbose:
                        print(f"    Advertencia: {len(e.details['writeErrors'])} duplicados ignorados")
        
        if verbose:
            print(f"✓ Corpus guardado: {len(metadata_docs)} juegos, {len(review_docs)} reviews")
    
    from ..models import CorpusDocument
    def _corpus_document_to_mongo(self, doc: CorpusDocument) -> Dict:
        """Convierte CorpusDocument a documento MongoDB."""
        review_dict = {
            "game_id": doc.review.game_id,
            "username": doc.review.username,
            "rating": doc.review.rating,
            "comment": doc.review.comment,
            "timestamp": doc.review.timestamp,
            "category": doc.review.category,
        }
        
        # Agregar datos procesados si existen
        if doc.text or doc.clean_text or doc.language:
            review_dict["processed"] = {
                "clean_text": doc.clean_text,
                "language": doc.language,
                "text_stats": doc.text,
                "patterns": doc.patterns
            }
        
        return review_dict
    
    def load_corpus(
        self,
        game_ids: Optional[List[int]] = None,
        categories: Optional[List[str]] = None,
        limit: Optional[int] = None,
        verbose: bool = True
    ):
        """Carga un Corpus desde MongoDB."""
        if verbose:
            print(f"Cargando corpus desde MongoDB...")
        
        meta_query = {}
        if game_ids:
            meta_query["game_id"] = {"$in": game_ids}
        
        metadata_docs = list(self.metadata_collection.find(meta_query))
        
        if not metadata_docs:
            print(f"⚠️ No se encontró ningun corpus")
            from ..models import Corpus
            return Corpus(games=[])
        
        if verbose:
            print(f"  Encontrados {len(metadata_docs)} juegos")
        
        games = []
        for meta_doc in tqdm(metadata_docs, disable=not verbose, desc="Cargando juegos"):
            game_id = meta_doc["game_id"]
            
            review_query = {
                "game_id": game_id
            }
            if categories:
                review_query["category"] = {"$in": categories}
            
            review_cursor = self.reviews_collection.find(review_query)
            if limit:
                review_cursor = review_cursor.limit(limit)
                
            from ..models import GameCorpus
            game_corpus = GameCorpus(
                game_id=game_id,
                metadata=meta_doc.get("metadata", {}),
                documents=[]
            )
            
            for review_doc in review_cursor:
                corpus_doc = self._mongo_to_corpus_document(review_doc)
                game_corpus.add_document(corpus_doc)
            
            games.append(game_corpus)
        
        corpus = Corpus(games=games)
        
        if verbose:
            total_reviews = len(corpus.documents)
            print(f"✓ Corpus cargado: {len(games)} juegos, {total_reviews} reviews")
        
        return corpus
    
    def _mongo_to_corpus_document(self, doc: Dict) -> CorpusDocument:
        """Convierte documento MongoDB a CorpusDocument."""
        from ..models import Review
        
        review = Review(
            username=doc.get("username", "unknown"),
            rating=doc.get("rating"),
            comment=doc.get("comment", ""),
            timestamp=doc.get("timestamp"),
            game_id=doc.get("game_id")
        )
        
        if "category" in doc:
            review.category = doc["category"]
        if "label" in doc:
            review.label = doc["label"]
        
        processed = None
        if "processed" in doc:
            proc = doc["processed"]
            processed = {
                "clean_text": proc.get("clean_text"),
                "language": proc.get("language"),
                "text": proc.get("text_stats", {}),
                "patterns": proc.get("patterns", {})
            }
        from ..models import CorpusDocument
        return CorpusDocument(review, processed)
    
    def get_corpus_stats(self) -> Dict:
        """Obtiene estadísticas del corpus."""
        pipeline = [
            {"$group": {
                "_id": "$category",
                "count": {"$sum": 1},
                "avg_rating": {"$avg": "$rating"}
            }}
        ]
        
        category_stats = list(self.reviews_collection.aggregate(pipeline))
        
        total_games = self.metadata_collection.count_documents({})
        
        return {
            "total_games": total_games,
            "total_reviews": sum(s["count"] for s in category_stats),
            "by_category": {s["_id"]: s for s in category_stats}
        }
    
    def get_game_metadata(self, game_id: int) -> Optional[Dict]:
        """Obtiene metadata de un juego específico."""
        return self.metadata_collection.find_one({
            "game_id": game_id,
        })
    
    def get_reviews_by_game(
        self,
        game_id: int,
        category: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """Obtiene reviews de un juego específico."""
        query = {"game_id": game_id}
        if category:
            query["category"] = category
        
        cursor = self.reviews_collection.find(query)
        if limit:
            cursor = cursor.limit(limit)
        
        return list(cursor)
    
    def delete_corpus(self, verbose: bool = True):
        """Elimina un corpus completo."""
        if verbose:
            print(f"Eliminando corpus ...")
        
        meta_result = self.metadata_collection.delete_many({})
        review_result = self.reviews_collection.delete_many({})
        
        if verbose:
            print(f"  Eliminados: {meta_result.deleted_count} juegos, "
                  f"{review_result.deleted_count} reviews")
    
    def export_to_json(
        self,
        output_path: str,
        list_format: bool = True,
        verbose: bool = True
    ):
        """Exporta corpus de MongoDB a JSON."""
        corpus = self.load_corpus(verbose=verbose)
        corpus.to_json(output_path, list_format=list_format)
        if verbose:
            print(f"✓ Exportado a {output_path}")
    
    def import_from_json(
        self,
        json_path: str,
        verbose: bool = True
    ):
        """Importa corpus desde JSON a MongoDB."""
        if verbose:
            print(f"Importando desde {json_path}...")
        from ..models import Corpus
        corpus = Corpus.from_json(json_path)
        self.save_corpus(corpus, verbose=verbose)
    
    def close(self):
        """Cierra la conexión a MongoDB."""
        self.client.close()