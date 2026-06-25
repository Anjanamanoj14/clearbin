# ♻️ ClearBin — AI Waste Classifier

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red)
![Accuracy](https://img.shields.io/badge/Accuracy-92.7%25-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

An AI-powered waste image classifier that identifies the correct bin 
for any waste item using deep learning and computer vision.

🚀 **[Try the live demo](https://anjanamanoj14-clearbin.hf.space)**

## 📸 Demo

Upload any waste image and ClearBin instantly tells you:
- What type of waste it is
- Which bin it belongs to
- Confidence score for all 6 categories


## 🗑️ Waste Categories

| Category | Bin | Emoji |
|----------|-----|-------|
| Cardboard | Brown bin | 📦 |
| Glass | Green bin | 🍶 |
| Metal | Blue bin | 🥫 |
| Paper | Blue bin | 📄 |
| Plastic | Yellow bin | 🧴 |
| Trash | Black bin | 🗑️ |


## 🧠 How It Works

1. **Dataset** — TrashNet dataset with 2,527 labeled waste images
2. **Model** — EfficientNet-B0 pretrained on ImageNet
3. **Transfer Learning** — Froze base layers, trained classifier head
4. **Fine Tuning** — Unfroze all layers with low learning rate (0.0001)
5. **Deployment** — Gradio app hosted on Hugging Face Spaces


## 📊 Results

| Metric | Score |
|--------|-------|
| Overall Accuracy | 92.7% |
| Best Class (Paper) | 95% F1 |
| Weakest Class (Trash) | 90% F1 |
| Total Parameters | 4,015,234 |
| Trainable (Phase 1) | 7,686 |


## 🛠️ Tech Stack

- **Language** — Python 3.10
- **Deep Learning** — PyTorch 2.1
- **Model** — EfficientNet-B0 (torchvision)
- **Data** — TrashNet (2,527 images, 6 classes)
- **App** — Gradio
- **Hosting** — Hugging Face Spaces

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/Anjanamanoj14/clearbin.git
cd clearbin

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

## 📁 Project Structure

```
clearbin/
├── data/
│   ├── train/          # 2019 training images
│   └── val/            # 508 validation images
├── notebooks/
│   ├── day01_foundations.ipynb
│   ├── day02_dataset.ipynb
│   ├── day03_split_and_nn.ipynb
│   ├── day04_training.ipynb
│   ├── day05_finetune.ipynb
│   └── day06_evaluation.ipynb
├── src/
├── app.py              # Gradio web app
├── requirements.txt
└── README.md
```

## 🔍 Key Learnings

- Transfer learning achieves 92.7% accuracy with only 2,527 images
- Fine tuning improved accuracy from 78.9% → 92.7% (+13.8%)
- Glass/Plastic most confused pair due to visual similarity
- Data augmentation critical for handling class imbalance

## 📄 License

MIT License — feel free to use and modify!