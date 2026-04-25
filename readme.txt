🐆 Panthera Pardus Fusca: Indian Leopard Dataset

This dataset was engineered specifically for the detection and monitoring of the Indian Leopard (Panthera pardus fusca) within the Himalayan and North-Indian landscape. Unlike standard wildlife datasets, this collection utilizes a multi-stage AI verification pipeline to ensure 100% species purity and high-variance visual samples.

Dataset Statistics
Total Verified Elite Images: 1450+ 
Negative/Background Samples: ~pls add background images during modeling it will help in improving accuracy
Resolution: 720p / 1080p Full-Frame, few are in low res to improve speeds.
Geographic Focus: Uttarakhand, Himachal Pradesh.

HOW IT WAS BUILT?

The "Elite" Extraction Pipeline
To achieve this level of accuracy, every image in this dataset passed through a three-layer "Ironclad" verification system:
Discovery Layer (YOLOv8x): Real-time object detection identifies potential feline shapes in raw video footage.
Discrimination Layer (OpenAI CLIP): A multimodal Transformer model verifies the subspecies. It explicitly rejects:
        ~Panthera uncia (Snow Leopards)
        ~Melanistic Panthera pardus (Black Panthers)
        ~Panthera tigris (Tigers)
Uniqueness Layer (pHash): Perceptual Hashing ensures no two images are visually redundant, maximizing the model's ability to generalize.


HEY THERE! I have included other pics from different sources as well:
*inaturalists: 100+ pics       ~all the images are under free use for any commercial usage
*icrawler: 85+ pics
*Wikimedia: 85+ pics
*lila sciences: 50+ pics
*youtube crawling: 1200+ images

Metadata Specifications
The dataset_metadata.csv provides normalized coordinates for ground-truth labeling:
x1, y1: Top-left corner of the leopard detection.
x2, y2: Bottom-right corner of the leopard detection.             don't use. it will help but metadata is only of around 800 images.

Usage for Training:  try this to improve accuracy 
~**** remember to use data augmentation so the dataset will be increased to 5000+ images (important).
** add background images more than 500+ to improve accuracy.
these are the best pics i have selected from 7000+ images the dataset is more than 99% correct but it would be good if you go through the dataset once. 
I am adding the youtube crawling script if you have a good gpu update the queries and rerun the script to increase the dataset size :)

Hardware: Powered by NVIDIA RTX GPU & Google Colab (Tesla T4)
(my own laptop had limitations because of no gpu so the images are less and collab has a per day limit)

Authors & Credits
Lead Developer: Nitish Joshi 2nd Year B.Tech CSE Student
Dev Bhoomi Uttrakhand University Dehradun
Contact: nitishjoshi0554@gmail.com

-- nitishjoshi.in

