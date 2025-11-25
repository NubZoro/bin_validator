import os
from PIL import Image

input_folder = r"C:\Users\arnav\OneDrive\Desktop\ai_amazon\bin-images"
output_folder = r"C:\Users\arnav\OneDrive\Desktop\ai_amazon\bin-images-resized"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)

        # Resize
        img_resized = img.resize((224, 224))

        # Save with same filename in new folder
        save_path = os.path.join(output_folder, filename)
        img_resized.save(save_path)

print("All images resized successfully to 224x224.")
