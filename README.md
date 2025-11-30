### Driver Drowsiness Detection System Using Deep Learning

This project aims to develop an intelligent system capable of accurately detecting driver fatigue, leveraging Deep Learning models to classify the state of the eyes (open or closed).

#### 1. Project Structure
<pre>
.
├── .venv/                  <!-- Python virtual environment -->
├── data/                   <!-- Folder containing the datasets -->
│   ├── awake/              <!-- Images of open eyes -->
│   └── sleepy/             <!-- Images of closed eyes -->
├── notebooks/              <!-- Jupyter notebooks for exploration and testing -->
│   └── data_preparation.ipynb
├── src/                    <!-- Python source code -->
│   ├── data_loader.py      <!-- Data preparation and loading -->
│   ├── model.py            <!-- CNN model definition -->
│   ├── train.py            <!-- Model training -->
│   ├── evaluate.py         <!-- Model evaluation and testing -->
│   └── predict.py          <!-- Prediction script for new images -->
├── requirements.txt        <!-- Python dependencies -->
├── README.md               <!-- Project documentation -->
└── app.py                  <!-- Gradio deployment -->
</pre>

#### 2. Dataset
The project uses the **MRL Eye Dataset** from Kaggle, which contains infrared images of eyes categorized into two classes: **Awake** and **Sleepy**.  
Source: [Kaggle - MRL Eye Dataset](https://www.kaggle.com/datasets/akashshingha850/mrl-eye-dataset?resource=download)  
Data used: **Awake (25,770 images)** and **Sleepy (25,167 images)**, extracted from the original dataset.

#### 3. Deep Learning Model Training
  - **Data Preparation**: cleaning, resizing, and normalizing the images.  
  - **Model Definition**: CNN architecture tailored for binary classification.  
  - **Training**: optimization and parameter tuning to improve accuracy.  
  - **Evaluation**: performance metrics including accuracy, precision, and recall.  
  - **Testing**: validating the model on an independent dataset to ensure generalization on new images.

#### 4. Deployment
The model is deployed using **Gradio**, providing an interactive interface for real-time eye state detection.

#### 5. Possible Results
 Eyes open  

 One eye closed and the other open  
 Eyes closed  
 No eyes detected

Yassine DARIF | INSEA - 2025