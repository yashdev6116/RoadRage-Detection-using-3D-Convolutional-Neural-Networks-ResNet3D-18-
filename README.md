# RoadRage-Detection-using-3D-Convolutional-Neural-Networks-ResNet3D-18-
Detect road rage and violent behavior in road videos using ResNet3D-18 (3D CNN). This AI system classifies uploaded clips as Fight or Non-Fight with confidence scores, and runs on a sleek Streamlit interface. Built with PyTorch, TorchVision, and OpenCV.

🚗 Road Rage Detection Using AI
An AI-powered system to detect violent or aggressive behavior (fights) from road surveillance or dashcam videos in real-time.

🔍 Project Overview
This project leverages deep learning and computer vision to detect road rage incidents, such as fights or violent behavior, from short video clips. It is built using:

PyTorch for modeling

R3D-18 (ResNet3D-18) for spatiotemporal video classification

Streamlit for an interactive and modern web interface

The model classifies a given video as either:

🟩 Non-Fight

🟥 Fight

💡 Features
🎥 Upload any .mp4 video clip (dashcam/CCTV)

🧠 Predicts if the scene contains violent behavior

🔢 Shows prediction confidence score

📜 Logs previous predictions in a CSV

🖤 Fully responsive dark-themed Streamlit interface

✅ All processing is done locally, no external contacts or APIs

🛠 Tech Stack
PyTorch

TorchVision

OpenCV

Streamlit

ResNet3D (r3d_18)

🌐 Use Case Examples
Traffic monitoring systems

Smart city surveillance

Driver safety monitoring

Behavior analysis from road footage

🚀 Run Locally
Clone the repository

Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the app:

bash
Copy
Edit
streamlit run app.py

📁 Dataset
The dataset used includes short video clips labeled as fight and non-fight scenes. For training, ~16 frames are sampled per clip and processed using ResNet3D.
https://www.kaggle.com/datasets/shaiktanveer7/road-rage-dataset

📷 Preview
Upload Interface	
![image](https://github.com/user-attachments/assets/49ff09ef-47b3-4081-b25b-da5ff60257e2)

Prediction Result
![image](https://github.com/user-attachments/assets/4dd46034-ae87-4aa0-918b-95a0115e660b)

👨‍💻 Made with ❤️ by Yash Dev
