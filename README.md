# Absenteeism Management System Dashboard

**Mini DataCamp Hackathon – First Prize 🏆 (Sep 2026)**

This project was my entry for the Mini DataCamp Hackathon, and I was grateful to receive first place. 🎉 The challenge was to analyse employee absenteeism dataset and look for insights that could help improve employee productivity, all within 22 hours😬. I cleaned the data, trained a machine learning model to predict types of absence, and presented the results in a one-page interactive dashboard.

**🚀 Try the live demo:** [https://absenteeism-dashboard-cadybmdrdwjfktrswdpgen.streamlit.app/]

> **🔒 About the data:** The original competition dataset is private, so this public version runs on synthetic data I generated to match the original's column distributions and key patterns (generate_demo_data.py). No real employee records are included, and the numbers in the demo are not the official competition results.

## Screenshot

![Dashboard](Screenshot_Absent_Dashboard.png)

## 💡 What the dashboard shows
- **KPI cards:** average target achievement, the age group with the most absence hours, average absence per incident and the peak absence month
- **Monthly absence trend:** which months lose the most working hours
- **Workload vs target:** months where the team falls below target
- **Health and commute patterns:** how age, BMI and distance from work relate to absence
- **Today's predictions:** how many employees are likely to take a Time Slip, Medical Leave (MC) or an Abnormal Absence

I added a short insight under each chart to make the results easier to understand.

## 🛠️ How I built it

- **Data cleaning:** found and replaced missing values (including `#` placeholders) across 17 columns, and grouped absence hours into three classes: Time Slip, Medical Leave (MC) and Abnormal Absence.
- **Prediction:** trained a KNN classifier, tuned with GridSearchCV, to predict the absence category of an employee record.
- **Dashboard:** built in Streamlit with Plotly charts, KPI cards and short insight boxes that explain each chart in simple words.

## Tech stack

Python, Pandas, NumPy, Scikit-learn, Plotly, Streamlit

## Project structure

```
app.py                  # Streamlit dashboard
demo_data.csv           # Synthetic demo data (no real records)
knn_model.pkl           # KNN model trained on the demo data
model_features.pkl      # Feature list used by the model
generate_demo_data.py   # Creates demo_data.csv (needs the private dataset)
train_demo_model.py     # Trains the demo model
notebooks/              # Original data cleaning and prediction notebooks
requirements.txt
```

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

To rebuild the demo data and model (requires the private dataset in `data/`):

```bash
python generate_demo_data.py
python train_demo_model.py
```

---

Thanks for checking out my project! 🙌
