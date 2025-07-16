from sentence_transformers import SentenceTransformer, util
import torch


model = SentenceTransformer('all-MiniLM-L6-v2')

VIBE_LABELS = [
  "Coquette",
  "Clean Girl",
  "Cottagecore",
  "Streetcore",
  "Y2K",
  "Boho",
  "Party Glam"
]

def extract_vibes(caption_txt_path, top_k=3):

    with open(caption_txt_path, "r", encoding="utf-8") as f:
        caption = f.read().strip()

    if not caption:
        print("Caption file is empty.")
        return ["unknown"]

    caption_emb = model.encode(caption, convert_to_tensor=True)
    vibe_embs = model.encode(VIBE_LABELS, convert_to_tensor=True)
    scores = util.cos_sim(caption_emb, vibe_embs)[0]
    top = torch.topk(scores, k=top_k)
    return [VIBE_LABELS[i] for i in top.indices]
