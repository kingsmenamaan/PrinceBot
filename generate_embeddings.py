import json
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import os
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BERTEmbedding:
    def __init__(self, model_name="multi-qa-mpnet-base-dot-v1"):
        """Initialize SentenceTransformer model for embedding generation."""
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.encoder = SentenceTransformer(model_name, device=self.device)
            logger.info(f"✅ Initialized SentenceTransformer model: {model_name} on {self.device}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize SentenceTransformer: {e}")
            raise

    def encode(self, texts):
        """Encode texts into embeddings with normalization."""
        try:
            embeddings = self.encoder.encode(texts, convert_to_numpy=True)
            # Normalize embeddings
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            logger.info(f"✅ Encoded {len(texts)} texts into embeddings with shape {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"❌ Failed to encode texts: {e}")
            raise

def generate_embeddings(dataset_path, embeddings_path, checkpoint_path, batch_size=128):
    """Generate and save embeddings for questions in the dataset."""
    # Load dataset
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"✅ Loaded dataset with {len(data)} entries from {dataset_path}")
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        raise

    # Extract questions from 'input' field and remove 'answer: ' prefix
    questions = [item["input"].replace("answer: ", "").strip() for item in data if "input" in item]
    if not questions:
        logger.error("❌ No valid questions found in the dataset")
        raise ValueError("No valid questions found in the dataset")
    logger.info(f"✅ Extracted {len(questions)} valid questions")

    # Skip if embeddings already exist
    if os.path.exists(embeddings_path):
        logger.info(f"⚠️ Embeddings already exist at {embeddings_path}, skipping generation")
        with open(embeddings_path, "rb") as f:
            query_embeddings = pickle.load(f)
        return query_embeddings

    # Load or initialize embeddings
    query_embeddings = []
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "rb") as f:
                query_embeddings = pickle.load(f)
            logger.info(f"✅ Loaded checkpoint with {len(query_embeddings)} embeddings")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load checkpoint: {e}. Starting from scratch.")

    encoder = BERTEmbedding()
    failed_queries = []
    start_idx = len(query_embeddings)

    for i in tqdm(range(start_idx, len(questions), batch_size), desc="Generating embeddings"):
        batch_questions = questions[i:i + batch_size]
        try:
            batch_embeddings = encoder.encode(batch_questions)
            query_embeddings.extend(batch_embeddings)
            with open(checkpoint_path, "wb") as f:
                pickle.dump(query_embeddings, f)
            logger.info(f"✅ Saved checkpoint at index {i + batch_size} with {len(query_embeddings)} embeddings")
        except Exception as e:
            failed_queries.extend(batch_questions)
            logger.warning(f"⚠️ Failed to process batch starting at index {i}: {e}")

    # Save final embeddings
    try:
        embedding_dict = {q: e for q, e in zip(questions, query_embeddings)}
        with open(embeddings_path, "wb") as f:
            pickle.dump(embedding_dict, f)
        logger.info(f"✅ Saved {len(embedding_dict)} embeddings to {embeddings_path}")
    except Exception as e:
        logger.error(f"❌ Failed to save embeddings: {e}")
        raise

    # Log failed queries
    if failed_queries:
        fail_log = embeddings_path.replace(".pkl", "_failed_queries.txt")
        with open(fail_log, "w", encoding="utf-8") as f:
            f.write("\n".join(failed_queries))
        logger.warning(f"⚠️ Saved {len(failed_queries)} failed queries to {fail_log}")
    else:
        logger.info("✅ All embeddings generated successfully")

    return embedding_dict

if __name__ == "__main__":
    DATASET_PATH = "C:/Users/HP/Desktop/princebot/princebot/model/full_dataset_fixed.json"  # Updated to match your dataset
    EMBEDDINGS_PATH = "C:/Users/HP/Desktop/princebot/princebot/model/query_embeddings.pkl"
    CHECKPOINT_PATH = "C:/Users/HP/Desktop/princebot/princebot/model/query_embeddings_checkpoint.pkl"

    try:
        generate_embeddings(DATASET_PATH, EMBEDDINGS_PATH, CHECKPOINT_PATH, batch_size=128)
    except Exception as e:
        logger.error(f"❌ Error generating embeddings: {e}")