import os
import json
from pathlib import Path
from cropped_yolo import extract_frames, run_yolo_on_frames, process_video
from faiss_data_extract import search_similar_products
from vibe_extraction import extract_vibes

def DeduplicateViaConfidence(products):
    product_map = {}
    for prod in products:
        pid = str(prod.get("Product_ID") or prod.get("Product ID"))
        if pid not in product_map or prod["confidence"] > product_map[pid]["confidence"]:
            product_map[pid] = prod
    return list(product_map.values())

def drop_unknown_products(products):
    return [prod for prod in products if prod["type"].lower() != "unknown"]

def full_pipeline(video_path):
    video_name = Path(video_path).stem
    cropped_dir = process_video(video_path)
    cropped_images = list(Path(cropped_dir).glob("*.jpg"))
    caption_file = video_path.replace(".mp4", ".txt")
    products = []
    for img_path in cropped_images:
        matches = search_similar_products(str(img_path))
        class_name = "unknown"
        parts = img_path.name.split("_")
        if len(parts) >= 3:
            class_name = parts[-2]  

        for match in matches["matches"]:
            products.append({
                "type": match["product_type"],
                "color": match["Colour"],  
                "match_type": match["match_type"],
                "Product_ID": match["matched_product_id"],
                "confidence": match["similarity"]
            })
    vibe = extract_vibes(caption_file)
    
    output = {
        "video_id": video_name,
        "vibes": vibe,  
        "products": products
    }
    output["products"] = DeduplicateViaConfidence(output["products"])
    #Because the provided images.csv and product.xlsx have diffrent products some of the data which are in image.csv isn't available in product.xlsx
    output["products"] = drop_unknown_products(output["products"])
    output_dir = f"outputs/{video_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{video_name}_final.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Final structured JSON saved to {output_path}")
    return output_path


if __name__ == "__main__":
    video_n = str(input("Enter Your Relative video path: \n"))
    video_path = video_n 
    full_pipeline(video_path)
