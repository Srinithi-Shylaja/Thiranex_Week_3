```
# Thiranex Data Science Internship - Week 3: Exploratory Data Analysis (EDA) Project

## 📌 Project Overview
This repository contains the completed work for **Task 3 (Exploratory Data Analysis Project)** of the Thiranex Data Science Internship.

The goal of this project is to perform a comprehensive 5-step Exploratory Data Analysis (EDA) on an amusement park roller coaster dataset using Python, Pandas, Matplotlib, and Seaborn to uncover key feature distributions, relationships, and actionable insights.

---

## 🛠️ 5-Step EDA Methodology

Following standard data science best practices, the analysis in `Thiranex_Week_3_EDA.ipynb` is structured as follows:

1. **Data Understanding**:
   - Inspected dataset shape (`rows x columns`), data types (`df.info()`), summary statistics (`df.describe()`), and missing value counts (`df.isna().sum()`).

2. **Data Preparation &amp; Cleaning**:
   - Subclassed key features (`Coaster_Name`, `Park`, `Material_Type`, `Speed_MPH`, `Height_FT`, `Length_FT`, `Inversions`, `Year_Introduced`, `Rating`).
   - Renamed headers for clarity and imputed missing numerical values using median strategies.
   - Dropped duplicate entries based on coaster names (`drop_duplicates()`).

3. **Feature Distributions (Univariate Analysis)**:
   - Visualized categorical variables using bar charts (e.g., Top 10 Parks by Coaster Count).
   - Analyzed numerical distributions using histograms and Kernel Density Estimate (KDE) plots (e.g., Coaster Speed Distribution).

4. **Feature Relationships (Bivariate &amp; Multivariate Analysis)**:
   - Created Seaborn scatter plots comparing **Speed vs. Height** categorized by `Material_Type`.
   - Plotted an annotated correlation heatmap across all numerical variables (`sns.heatmap`).

5. **Asking &amp; Answering Data Questions (Insights &amp; Aggregations)**:
   - Formulated and answered a targeted business question: *Which park has the highest average coaster speed for coasters introduced in year 2000 or later?*
   - Calculated group aggregations using `groupby()` and rendered horizontal bar charts of the results.

---

## 📁 Repository Structure

```text
Thiranex_Week_3/
│
├── Thiranex_Week_3_EDA.ipynb  # Main Exploratory Data Analysis Notebook
├── coaster_db.csv             # Project Dataset
└── README.md                  # Task 3 Documentation

```

---

## 💻 Tech Stack &amp; Libraries Used

* **Language**: Python 3
* **Environment**: Google Colab / JupyterLab
* **Libraries**: `pandas`, `numpy`, `matplotlib`, `seaborn`
