import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import os
import pickle
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
QUERIES_PATH = "C:/Users/HP/Desktop/princebot/princebot/model/queries.json"
PROPERTIES_PATH = "C:/Users/HP/Desktop/princebot/princebot/model/properties.json"
MERGED_PATH = "C:/Users/HP/Desktop/princebot/princebot/model/full_dataset_fixed.json"
INDEX_PATH = "C:/Users/HP/Desktop/princebot/princebot/model/faiss_index/index.faiss"
EMBEDDINGS_PATH = "C:/Users/HP/Desktop/princebot/princebot/model/query_embeddings.pkl"

def merge_datasets():
    """Merge queries.json and properties.json into a single dataset."""
    try:
        with open(QUERIES_PATH, "r", encoding="utf-8") as f1:
            queries = json.load(f1)
        logger.info(f"✅ Loaded {len(queries)} entries from {QUERIES_PATH}")
    except Exception as e:
        logger.error(f"❌ Failed to load {QUERIES_PATH}: {e}")
        raise

    try:
        with open(PROPERTIES_PATH, "r", encoding="utf-8") as f2:
            properties = json.load(f2)
        logger.info(f"✅ Loaded {len(properties)} entries from {PROPERTIES_PATH}")
    except Exception as e:
        logger.error(f"❌ Failed to load {PROPERTIES_PATH}: {e}")
        raise

    merged = []
    for item in tqdm(queries + properties, desc="Merging datasets"):
        # Check for expected keys; adjust based on actual dataset structure
        if "question" in item and "answer" in item:
            merged.append({
                "input": f"answer: {item['question']}",
                "output": item["answer"]
            })
        elif "input" in item and "output" in item:
            # Handle case where dataset already uses "input" and "output"
            merged.append({
                "input": item["input"],
                "output": item["output"]
            })
        else:
            logger.warning(f"⚠️ Skipping item with missing keys: {item}")

    if not merged:
        logger.error("❌ No valid entries found after merging")
        raise ValueError("No valid entries found after merging")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(MERGED_PATH), exist_ok=True)
    with open(MERGED_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✅ Merged {len(merged)} entries into {MERGED_PATH}")
    return merged

def generate_faiss_and_embeddings(data):
    """Generate embeddings and FAISS index for the merged dataset."""
    questions = [item["input"].replace("answer: ", "").strip() for item in data]
    if not questions:
        logger.error("❌ No valid questions extracted from dataset")
        raise ValueError("No valid questions extracted from dataset")
    logger.info(f"✅ Extracted {len(questions)} questions")

    # Load SentenceTransformer with device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer("multi-qa-mpnet-base-dot-v1", device=device)
    logger.info(f"✅ SentenceTransformer loaded on {device}")

    # Generate embeddings with normalization
    embeddings = encoder.encode(questions, convert_to_numpy=True, show_progress_bar=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    logger.info(f"✅ Generated {len(embeddings)} normalized embeddings with shape {embeddings.shape}")

    # Create FAISS IVF index
    dim = embeddings.shape[1]
    nlist = 100  # Number of clusters; adjust based on dataset size
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)

    # Train the index
    embeddings = embeddings.astype("float32")
    index.train(embeddings)
    index.add(embeddings)
    logger.info(f"✅ Added {index.ntotal} vectors to FAISS index")

    # Save FAISS index
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    logger.info(f"✅ Saved FAISS index to {INDEX_PATH}")

    # Save embeddings
    emb_dict = {q: e for q, e in zip(questions, embeddings)}
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(emb_dict, f)
    logger.info(f"✅ Saved {len(emb_dict)} embeddings to {EMBEDDINGS_PATH}")

if __name__ == "__main__":
    try:
        merged_data = merge_datasets()
        generate_faiss_and_embeddings(merged_data)
    except Exception as e:
        logger.error(f"❌ Error in main execution: {e}")