# 📧 Email Spam Detection System

A complete machine learning web application that classifies email/SMS messages as **SPAM** or **HAM (Not Spam)** using Natural Language Processing and Scikit-learn classifiers, served through a Streamlit frontend.

---

## 🎯 Project Overview

The system trains three classification models (Multinomial Naive Bayes, Logistic Regression, Linear SVC) on a real SMS spam dataset, selects the best performer, and exposes it through a clean web UI where users can paste any message and get an instant prediction with confidence score.

---

## ✅ Prerequisites

- Python **3.8 or higher**
- pip (comes with Python)
- A terminal / command prompt (or PyCharm Terminal)

---

## ⚙️ Installation

```bash
# 1. Navigate to the project folder
cd email_spam_detection

# 2. (Optional but recommended) Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🏋️ Train the Model

Run this **once** before starting the app:

```bash
python train_model.py
```

This will:
1. Load and clean `data/spam.csv`
2. Train MultinomialNB, Logistic Regression, and LinearSVC
3. Print accuracy, precision, recall, F1, and confusion matrix for each
4. Save the best model to `models/spam_detection_model.pkl`

---

## 🚀 Run the Application

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`

---

## 📁 Project Structure

| File / Folder | Role |
|---|---|
| `app.py` | Streamlit frontend — UI, buttons, results display |
| `train_model.py` | Model training script — run once to create the .pkl file |
| `preprocessing.py` | Dataset loading, cleaning, TF-IDF vectorisation helpers |
| `predict.py` | Model loading and prediction functions used by the app |
| `requirements.txt` | Python package dependencies |
| `README.md` | This file |
| `data/spam.csv` | Raw dataset (SMS Spam Collection) |
| `models/spam_detection_model.pkl` | Saved trained pipeline (created after training) |
| `models/.gitkeep` | Keeps the models/ folder tracked in git before training |

---

## 🧪 Sample Test Messages

Try these in the app after training:

| Type | Message |
|---|---|
| Normal | `Hey, are we still meeting for lunch tomorrow at noon?` |
| Spam | `Congratulations! You've won a FREE iPhone. Click here to claim: http://bit.ly/win-now` |
| Link Spam | `URGENT: Your account will be suspended. Verify now at http://secure-login.xyz/verify` |
| Empty | *(leave text area blank and click Detect)* |
| Unusual punctuation | `W0W!!! Y0U WON £5000 — Call NOW: 07XXXXXXXX!!! Limited TIME!!!` |

---

## 🔧 Troubleshooting

### "Model not found" error in the app
You haven't run the training script yet. Run:
```bash
python train_model.py
```

### UnicodeDecodeError when loading dataset
The code automatically retries with `latin-1` encoding. If you still see this error, open `preprocessing.py` and change the fallback encoding to `cp1252`.

### Port already in use
```bash
streamlit run app.py --server.port 8502
```

### ModuleNotFoundError
Make sure you're in the `email_spam_detection/` directory and your virtual environment is activated:
```bash
cd email_spam_detection
pip install -r requirements.txt
```

### PyCharm specific
- Open the `email_spam_detection/` folder as the project root.
- Set the Python interpreter to your virtual environment.
- Use **Tools → Terminal** and run `python train_model.py` then `streamlit run app.py`.

---

## 🗂️ Har File Ka Role — Hinglish Mein (College Students Ke Liye)

| File | Kya karta hai? |
|---|---|
| **app.py** | Yeh hamara main web page hai. Streamlit library se bana hua hai. Isme text box, buttons, aur result dikhane ka kaam hota hai. |
| **train_model.py** | Yeh script dataset ko padhti hai, model train karti hai, aur best model ko save karti hai. Isko sirf ek baar chalana hota hai. |
| **preprocessing.py** | Yeh data ko saaf karta hai — extra columns hatata hai, duplicate rows remove karta hai, text ko TF-IDF features mein convert karta hai. |
| **predict.py** | Yeh saved model ko load karta hai aur naya message dene par spam ya ham predict karta hai. App.py isi ko call karta hai. |
| **requirements.txt** | Isme un sab libraries ke naam hain jo project ke liye chahiye. `pip install -r requirements.txt` se sab install ho jaati hain. |
| **data/spam.csv** | Yeh hamara dataset hai — 5000+ SMS messages jinpe ham/spam labels lage hain. |
| **models/spam_detection_model.pkl** | Train hone ke baad yahan model save hota hai. Joblib library isko save aur load karti hai. |

---

## 📊 Model Evaluation Summary (Typical Results)

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| MultinomialNB | ~97% | ~97% | ~89% | ~93% |
| Logistic Regression | ~98% | ~97% | ~93% | ~95% |
| LinearSVC | ~98% | ~98% | ~94% | ~96% |

*Actual numbers may vary slightly depending on dataset version.*

---

## 📝 Notes

- ML predictions are **not 100% accurate**. Always apply human judgement for important emails.
- This project is intended for **educational purposes** as a college minor project demonstration.
- The model is trained on an SMS spam dataset; performance on formal emails may vary.

---

*Built with ❤️ using Python, Scikit-learn, and Streamlit.*
