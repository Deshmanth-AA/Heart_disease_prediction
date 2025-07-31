# 💓 Heart Disease Prediction Dashboard

This project is an interactive Streamlit dashboard that allows users to explore heart disease data, train machine learning models, and make predictions both interactively and in batches.

---

## 📊 Features

- ✅ Load and explore the dataset
- 📈 Visualize data with histograms and correlation heatmaps
- 🧠 Train machine learning models:
  - Random Forest
  - Logistic Regression
  - Decision Tree
- 🎯 View model metrics: accuracy, confusion matrix, and ROC curve
- 🔍 Predict heart disease using:
  - Manual form input (live prediction)
  - Batch CSV upload (bulk prediction)
- 💾 Automatically saves the trained model to `models/trained_model.pkl`

---

## 🛠️ Requirements

Install required packages using:

```bash
pip install -r requirements.txt
```

Dependencies include:
- streamlit
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

---

## 🚀 Run the App

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## 📥 Batch Prediction Format

Upload a `.csv` with the following columns (excluding `target`):
```csv
age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal
```

Example:
```csv
63,1,3,145,233,1,0,150,0,2.3,0,0,1
```

The app will output predictions and allow you to download the results.

---

## 🔐 Notes

- Make sure the `models/` folder exists before training. If it doesn’t, the app will create it automatically.
- The model is retrained each time unless you manually modify `app.py` to persist across sessions.

---

## 👨‍💻 Author

Developed by Deshmanth A.  
Feel free to modify and expand for your own projects!
