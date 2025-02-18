# 📝 NLP-Remarks-Classification

This project focuses on classifying visit locations using Natural Language Processing (NLP). The goal is to automatically categorize visit remarks written by BFI Finance agents into predefined location types, such as:
- Customer's House
- Customer's Office
- Customer's Business Location
- Other Family House
- Other Rent House
- Others
- Unknown

## 💡 Why This Project?
BFI Finance agents often fill out visit report forms and write remarks after conducting customer visits. These remarks can provide detailed information regarding the agent's visit, including whether the customer was found at the registered address or if alternative locations were identified. Some agents may discover alternative addresses that were not originally provided by BFI. When visiting these alternative locations, they may find the customer at a different place rather than the registered home address. These remarks may provide valuable insights that need to be accurately classified to enhance visit success rates and improve customer tracking efficiency. Our solution:
- Utilizing NLP and Machine Learning to classify remarks into visit location categories automatically.
- Enhancing visit report analysis efficiency with a fast and accurate system.
- Providing a web-based interface (Streamlit) where users can input text or upload CSV/Excel files for batch classification.

## 🚀 Project Overview
- 🔍 Goal: Develop an NLP-based model to classify visit remarks reported by BFI Finance agents into specific location categories.
- 🗂 Dataset: Collected visit remarks from BFI agents.
- 🏗 Approach: Text preprocessing, feature extraction, and machine learning model training.
- 🧠 Model: Support Vector Machine (SVM), Multinomial Naïve Bayes, Random Forest Classifier.
- 🎯 Evaluation Metric: Accuracy.

## 🛠 Tech Stack
- Programming Language: Python
- Web Framework: Streamlit
- Machine Learning & NLP:
  - Text Processing: NLTK, Sastrawi (Indonesian words stemming)
  - Feature Extraction: Scikit-learn (CountVectorizer)
  - Machine Learning Models: Multinomial Naïve Bayes, SVM, Random Forest (Scikit-learn)
  - Data Processing: Pandas, NumPy
- File Handling: Pandas (CSV, Excel)
- Visualization & UI: Streamlit (for interactive web app)
- Model Deployment: Pickle (for loading pre-trained models)

## 📊 Results
- Multinomial Naïve Bayes Accuracy: 75%
- SVM Accuracy: 83%
- Random Forest Accuracy: 85%

## 🏗 How to Run

1️⃣ Clone this repository
```bash
git clone https://github.com/vierikurniawan/NLP-Remarks-Classification.git  
cd NLP-Remarks-Classification  
```
2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```  
3️⃣ Run the Streamlit dashboard
```bash
streamlit run main.py
```
