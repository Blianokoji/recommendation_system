from sentence_transformers import SentenceTransformer

class EmbeddingModelSingleton:
    _instance = None
    _model = None
    
    @classmethod
    def get_model(cls, model_name="all-MiniLM-L6-v2"):
        if cls._instance is None:
            cls._instance = cls()
            print(f"[INFO] Initializing global SentenceTransformer model: {model_name}")
            cls._model = SentenceTransformer(model_name)
        return cls._model
