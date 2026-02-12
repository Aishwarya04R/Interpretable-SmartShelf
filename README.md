# Interpretable SmartShelf: AI-Driven Food Quality & Shelf-Life Prediction

**SmartShelf** is a multi-modal deep learning framework designed to reduce food waste by accurately predicting the freshness and remaining shelf life of perishable foods. It covers four domains: **Meat & Seafood, Fruits, Vegetables, and Bakery**.

## Features
- **Multi-Task Learning:** Simultaneous prediction of Food Type, Freshness Status, and Shelf Life (Days).
- **Explainable AI (XAI):** Integrated **Grad-CAM** and **LIME** to visualize spoilage patterns (e.g., mold, discoloration).
- **Sustainability:** Real-time Carbon Footprint ($CO_2e$) estimation.
- **Safety Logic:** Automated override system to prevent consumption of spoiled items.

## Tech Stack
- **Deep Learning:** TensorFlow, PyTorch, EfficientNetB4, MobileNetV2, EfficientNetB0, Custom Multi Scale CNN
- **Interface:** Streamlit
- **XAI:** Lime, OpenCV (Grad-CAM)

## Dataset
The system was trained on a stratified dataset of **28,000+ images**.
**[Click Here to View Dataset on Kaggle](https://www.kaggle.com/datasets/msirisha1403/food-quality-and-shelf-life-prediction)**

## Models
Due to file size limits, the pre-trained model weights (`.keras`, `.pth`) are hosted externally.
**[Download Model Weights (Google Drive)](PASTE_YOUR_DRIVE_LINK_HERE)**

## Screenshots
![Dashboard](https://github.com/Aishwarya04R/Interpretable-SmartShelf/blob/main/SmarShelf-Assets/Dashboard.png)
![Input-meat](https://github.com/Aishwarya04R/Interpretable-SmartShelf/blob/main/SmarShelf-Assets/Input-meat-model.png)
![Input-fruit](https://github.com/Aishwarya04R/Interpretable-SmartShelf/blob/main/SmarShelf-Assets/Input-fruit-model.png)
![Decay-Cure-and-AI-recommedation](https://github.com/Aishwarya04R/Interpretable-SmartShelf/blob/main/SmarShelf-Assets/Dashboard.png)
![Explainable_AI(GRAD&LIME)-bakery](https://github.com/Aishwarya04R/Interpretable-SmartShelf/blob/main/SmarShelf-Assets/Dashboard.png)
![Explainable_AI(GRAD&LIME)-fruit](https://github.com/Aishwarya04R/Interpretable-SmartShelf/blob/main/SmarShelf-Assets/Dashboard.png)
![Explainable_AI(GRAD&LIME)-meat](https://github.com/Aishwarya04R/Interpretable-SmartShelf/blob/main/SmarShelf-Assets/Dashboard.png)
![Explainable_AI(GRAD&LIME)-vegetable](https://github.com/Aishwarya04R/Interpretable-SmartShelf/blob/main/SmarShelf-Assets/Dashboard.png)
![Input-bakery](https://github.com/Aishwarya04R/Interpretable-SmartShelf/blob/main/SmarShelf-Assets/Dashboard.png)

![Input-vegetable](https://github.com/Aishwarya04R/Interpretable-SmartShelf/blob/main/SmarShelf-Assets/Dashboard.png)
![Output-bakery](https://github.com/Aishwarya04R/Interpretable-SmartShelf/blob/main/SmarShelf-Assets/Dashboard.png)
![Output-fruit](https://github.com/Aishwarya04R/Interpretable-SmartShelf/blob/main/SmarShelf-Assets/Dashboard.png)
![Output-meat](https://github.com/Aishwarya04R/Interpretable-SmartShelf/blob/main/SmarShelf-Assets/Dashboard.png)
![Output-vegetable](https://github.com/Aishwarya04R/Interpretable-SmartShelf/blob/main/SmarShelf-Assets/Dashboard.png)
![Sustainabilty-feature](https://github.com/Aishwarya04R/Interpretable-SmartShelf/blob/main/SmarShelf-Assets/Dashboard.png)


