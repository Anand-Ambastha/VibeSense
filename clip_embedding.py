import clip
import torch
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import faiss
import numpy as np
import pickle
import time
from sklearn.preprocessing import normalize


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

df = pd.read_csv("images.csv")

embeddings = []
valid_entries = []
bad_urls = [] 

def fetch_image(url, retries=5, delay=10):
    for attempt in range(1, retries + 1):
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                    " AppleWebKit/537.36 (KHTML, like Gecko)"
                    " Chrome/138.0.0.0 Safari/537.36"
                ),
                "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.shopify.com/"
            }
            response = requests.get(url, headers=headers, timeout=60)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if not content_type.startswith("image/"):
                print(f"Skipped (not an image): {url} - Content-Type: {content_type}")
                return None

            img = Image.open(BytesIO(response.content))
            img.verify()
            img = Image.open(BytesIO(response.content)).convert("RGB")
            return img

        except requests.exceptions.Timeout:
            print(f"[Timeout] Attempt {attempt}/{retries} for {url}")
            if attempt < retries:
                time.sleep(delay)
            else:
                print(f"[FAILED] Giving up on {url}")
                bad_urls.append(url)
        except requests.exceptions.RequestException as e:
            print(f"[Request Error] {url}: {e}")
            bad_urls.append(url)
            return None
        except Image.UnidentifiedImageError:
            print(f"[Image Error] Cannot identify image: {url}")
            bad_urls.append(url)
            return None
        except Exception as e:
            print(f"[Other Error] {url}: {e}")
            bad_urls.append(url)
            return None

print("Embedding catalog images...")

for i, row in tqdm(df.iterrows(), total=len(df)):
    url = row["image_url"]
    image = fetch_image(url)
    if image is None:
        continue

    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(image_tensor).cpu().numpy()
    embeddings.append(emb)
    valid_entries.append(row)

    time.sleep(0.2)

valid_df = pd.DataFrame(valid_entries)

if embeddings:
    np_embeddings = np.vstack(embeddings).astype("float32")
    
    # Normalize the embeddings row-wise
    np_embeddings = normalize(np_embeddings, axis=1)

    dim = np_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product = cosine similarity for normalized vectors
    index.add(np_embeddings)

    faiss.write_index(index, "catalog_clip_cosine.index")
    print(" Saved FAISS index: catalog_clip_cosine.index")

    valid_df.to_csv("catalog_metadata.csv", index=False)
    print(" Saved metadata: catalog_metadata.csv")
else:
    print(" No embeddings generated. Check your URLs.")

if bad_urls:
    with open("failed_urls.txt", "w") as f:
        for url in bad_urls:
            f.write(url + "\n")
    print(f"Saved {len(bad_urls)} failed URLs to failed_urls.txt")
