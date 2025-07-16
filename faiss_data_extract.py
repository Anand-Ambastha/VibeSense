import faiss
import pandas as pd
import numpy as np
from PIL import Image
import clip
import torch
import json
import os
from pprint import pprint
from sklearn.preprocessing import normalize
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load FAISS index
index = faiss.read_index("catalog_clip_cosine.index")

# Load catalog metadata and product data
catalog_meta = pd.read_csv("catalog_metadata.csv")
product_data = pd.read_excel("product_data.xlsx")

print("product_data.xlsx columns:", product_data.columns.tolist())

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def embed_image(path):
    img = Image.open(path).convert("RGB")
    with torch.no_grad():
        tensor = preprocess(img).unsqueeze(0).to(device)
        emb = model.encode_image(tensor).cpu().numpy()
        emb /= np.linalg.norm(emb)
    return emb


def search_similar_products(query_path, threshold=0.75, top_k=1):
    print(f"\n Processing: {os.path.basename(query_path)}")
    
    query_vec = embed_image(query_path)

    # FAISS search
    D, I = index.search(query_vec.astype("float32"), k=top_k)
    similarities = D[0]
    print("Top distances:", D[0])
    print("Top indices:", I[0])

    filtered = [(idx, sim) for idx, sim in zip(I[0], similarities) if sim >= threshold]

    results = {
        "query_image": os.path.basename(query_path),
        "matches": []
    }

    if not filtered:
        print("No matches found above similarity threshold.")
    else:
        print(" Filtered Matches:")
        for idx, sim in filtered:
            product_id = str(catalog_meta.iloc[idx]["id"])
            row = product_data[product_data["id"].astype(str) == str(product_id)]
            if not row.empty:
                tags = row.iloc[0].get("product_tags", "")
                product_type = row.iloc[0].get("product_type", "unknown")
                color = next((tag.split(":")[1].strip()
                              for tag in tags.split(",")
                              if tag.strip().lower().startswith("colour:")), "unknown")
                details = row.iloc[0].to_dict()
            else:
                product_type = "unknown"
                color = "unknown"
                details = {}
            match_type = "Exact Match" if sim >= 0.9 else "Similar Match"
            matched_details_df = product_data[product_data["id"] == product_id]
            records = matched_details_df.to_dict(orient="records")
            details = records[0] if records else {}

            print(f" {match_type} | ID: {product_id} | Sim: {round(float(sim), 4)}")

            results["matches"].append({
                "matched_product_id": product_id,
                "match_type": match_type,
                "product_type" : product_type,
                "Colour": color,
                "similarity": round(float(sim), 4),
                "details": details
            })
    return results

#search_similar_products(r"C:\Users\Dell\flickd-hkt\videos\2025-05-27_13-46-16_UTC.jpg")

print("Vectors in FAISS index:", index.ntotal)
