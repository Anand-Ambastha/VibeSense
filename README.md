# 📅 Flickd | VibeSense

**VibeSense** is an AI-powered fashion intelligence feature developed under the **Flickd** project. It analyzes short-form videos to detect fashion products and extract their underlying aesthetic *vibes* — such as *Coquette*, *Streetwear*, or *Vintage* — using a blend of vision-language models.

Developed for the [Flickd AI Hackathon 2025](https://drive.google.com/file/d/1Y1Rsb6670qDuvdi4oElfcCWLeC7HpjSO/view?usp=sharing), VibeSense brings together real-time detection, product matching, and vibe interpretation from captions and visual content.

---

## 🚀 What Does VibeSense Do?

Given a short video (like an Instagram reel or TikTok):

1. **Extract Frames** using FFmpeg
2. **Detect Clothing Items** using YOLOv8 trained on Fashionpedia
3. **Crop Detected Items**
4. **Generate Embeddings** with OpenAI’s CLIP
5. **Match Products** from a pre-indexed FAISS catalog
6. **Read Captions** from `.txt` files
7. **Extract Vibes** (e.g., *Romantic*, *Grunge*) using Sentence Transformers
8. **Output a Structured JSON** with matched products and vibe tags

---

## 🧠 Why VibeSense?

Short-form content is dominating fashion discovery. But brands, creators, and platforms struggle to:

* Recognize product details from video
* Understand the *emotional style* of a look
* Auto-tag reels for trends or shopping

**VibeSense** solves this by fusing:

* 🎞️ Video Frame Analysis
* 👗 Visual Product Matching
* 🧠 Caption-to-Vibe Understanding

All in a single lightweight pipeline.

---

## 🧪 Example Output

```json
{
  "video_id": "2025-05-27_13-46-16_UTC",
  "vibes": ["Coquette", "Clean Girl", "Boho"],
  "products": [
    {
      "Product_ID": "15581",
      "type": "Shirt",
      "color": "Grey",
      "match_type": "Similar Match",
      "confidence": 0.8543
    },
    {
      "Product_ID": "15974",
      "type": "Dress",
      "color": "Purple",
      "match_type": "Similar Match",
      "confidence": 0.84
    }
  ]
}
```

---

## 🔗 Hackathon Problem Statement

📄 [Click to view the official problem statement PDF](https://drive.google.com/file/d/1Y1Rsb6670qDuvdi4oElfcCWLeC7HpjSO/view?usp=sharing)


---

## 🛠 Tech Stack

| Component         | Library / Model                  |
| ----------------- | -------------------------------- |
| Object Detection  | YOLOv8 (Ultralytics)             |
| Caption Embedding | `sentence-transformers` (MiniLM) |
| Product Matching  | CLIP + FAISS                     |
| Image Processing  | OpenCV, Pillow                   |
| Data Handling     | Pandas, NumPy                    |
| Interface         | CLI + JSON Output                |

---

## 🚀 Which Models we used and why ?
1. ViT-B/32 (CLIP) – For extracting semantic image embeddings to enable visual similarity search. Chosen for its strong vision-language understanding.
2. all-MiniLM-L6-v2 – Used for generating compact text embeddings from product metadata. Lightweight and great for semantic text matching.
3. YOLOv8n – Employed for fast object detection on fashion items. Optimized for real-time performance with minimal resource usage.

---

## Dataset
The project uses the Fashionpedia dataset, which includes 45 clothing categories with detailed annotations. It was used for training YOLOv8n and generating embeddings for fashion item analysis.
---

## 📂 Folder Structure

```
flickd-vibesense/
├── main.py                  # Entry point
├── clip_embedding.py        # CLIP catalog embedding
├── cropped_yolo.py          # Frame & crop logic
├── faiss_data_extract.py    # Matching logic
├── vibe_sense.py            # Caption vibe extraction
├── product_data.xlsx        # Product catalog
├── videos/                  # Input videos, Caption Files
├── crops/                   # YOLO crops
├── models/                  # YOLO model
├── outputs/                 # Final JSON outputs
└── runs/                    # YOLO predictions
```

---

## 🔄 How to Run

1. ✅ Clone the repo
2. 📦 Install requirements
3. 🎥 Place your video in `videos/` and its caption in `captions/` with the same filename
4. ▶️ Run:

```bash
python main.py
```

5. 🗂️ Find your results in `outputs/`

---

## 🌟 Credits

Made with ❤️ for the **Flickd**
By \[Anand]
Using open-source tools and custom-trained models

---

> *"From frames to fashion feels — that's VibeSense."*
