import json
with open("training_annotations_raw.json", "r", encoding="utf-8") as f:
    data = json.load(f)

cleaned = []

for item in data:
    cleaned.append({
        "id" : item["id"],
        "file_name" : item["still"].replace("/data/upload/1/", ""),
        "shot_type" : item["shot_type"]
    })

with open("training_annotations_clean.json", "w") as f:
    json.dump(cleaned, f, indent=2)
