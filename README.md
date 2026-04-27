# 🐆 Panthera Pardus Fusca: Indian Leopard Dataset

This dataset was engineered specifically for the detection and monitoring of the **Indian Leopard (*Panthera pardus fusca*)** within the Himalayan and North-Indian landscape. Unlike standard wildlife datasets, this collection utilizes a multi-stage AI verification pipeline to ensure 100% species purity and high-variance visual samples.

## 📊 Dataset Statistics

* **Total Verified Elite Images:** 1450+
* **Negative/Background Samples:** *Note: Please add background images during modeling to help improve accuracy.*
* **Resolution:** 720p / 1080p Full-Frame (a few are in lower resolution to improve processing speeds).
* **Geographic Focus:** Uttarakhand, Himachal Pradesh.

## 🛠️ How It Was Built

### The "Elite" Extraction Pipeline
To achieve this level of accuracy, every image in this dataset passed through a three-layer "Ironclad" verification system:

1. **Discovery Layer (YOLOv8x):** Real-time object detection identifies potential feline shapes in raw video footage.
2. **Discrimination Layer (OpenAI CLIP):** A multimodal Transformer model verifies the subspecies. It explicitly rejects:
   * *Panthera uncia* (Snow Leopards)
   * Melanistic *Panthera pardus* (Black Panthers)
   * *Panthera tigris* (Tigers)
3. **Uniqueness Layer (pHash):** Perceptual Hashing ensures no two images are visually redundant, maximizing the model's ability to generalize.

### 📸 Data Sources
The dataset aggregates high-quality images from multiple sources:
* **YouTube Crawling:** 1200+ images
* **iNaturalist:** 100+ images *(Note: All images are under free use for any commercial usage)*
* **icrawler:** 85+ images
* **Wikimedia:** 85+ images
* **LILA BC (Science):** 50+ images

## 📂 Metadata Specifications

The `dataset_metadata.csv` provides normalized coordinates for ground-truth labeling:
* `x1, y1`: Top-left corner of the leopard detection.
* `x2, y2`: Bottom-right corner of the leopard detection.

> **⚠️ Important Note on Metadata:** Bounding box metadata is only available for around 800 images. 

## 💡 Usage for Training (Recommendations to Improve Accuracy)

* **Data Augmentation:** It is highly recommended to use data augmentation techniques so the effective dataset size is increased to 5000+ images.
* **Background Images:** Add 500+ negative/background images to reduce false positives and improve accuracy.
* **Manual Verification:** These are the best pictures selected from over 7,000+ raw images. While the dataset is >99% correct, a quick manual pass is recommended.
* **Scaling Up:** A YouTube crawling script is included. If you have a powerful GPU, you can update the search queries and re-run the script to further increase the dataset size!

## 💻 Hardware Used
* NVIDIA RTX GPU
* Google Colab (Tesla T4)
*(Note: Initial local extraction faced hardware limitations, hence Colab was heavily utilized despite daily limits.)*

## 👨‍💻 Authors & Credits

* **Lead Developer:** Nitish Joshi (2nd Year B.Tech CSE Student)
* **Institution:** Dev Bhoomi Uttrakhand University, Dehradun
* **Contact:** nitishjoshi0554@gmail.com
* **Website:** [nitishjoshi.in](https://nitishjoshi.in)
