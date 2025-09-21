# Indian Kids Screen Time Analysis 📱

## 1. Project Overview
This project analyzes the screen time habits of children in India to understand the impact of digital devices on their lives. The analysis explores patterns based on age, location (urban/rural), and primary device usage. It also uses K-Means clustering to identify distinct user groups based on their screen time behavior.

---

## 2. Dataset
The dataset used is `Indian_Kids_Screen_Time.csv`, a Kaggle dataset containing survey data on children's digital habits.

* **Source:** Kaggle
* **Link:** [https://www.kaggle.com/datasets/prathamtripathi/indian-kids-screen-time-analysis](https://www.kaggle.com/datasets/prathamtripathi/indian-kids-screen-time-analysis)
* **License:** CC0: Public Domain

---

## 3. Tools & Libraries
* **Python:** The core language for the analysis.
* **Pandas:** For data loading and cleaning.
* **Matplotlib & Seaborn:** For creating insightful visualizations.
* **Scikit-learn:** For K-Means clustering to segment users.
* **argparse:** To allow running the script with command-line arguments.

---

## 4. How to Run This Project
This script is designed to be run from the command line.

1.  Clone this repository to your local machine.
2.  Ensure you have Python and the required libraries installed by running:
    ```bash
    pip install -r requirements.txt
    ```
3.  Execute the Python script from your terminal, passing the CSV file as an argument:
    ```bash
    python Indian_kids_screetime_analysis.py Indian_Kids_Screen_Time.csv
    ```
The script will generate and save several visualizations (e.g., boxplots, scatter plots) in the same directory.

---

## 5. Key Findings
* **Device Dominance:** Smartphones and TVs are the primary devices contributing to high screen time among Indian children.
* **Health Correlations:** The analysis shows a correlation between high screen time and reported health impacts like poor sleep and eye strain.
* **User Segmentation:** K-Means clustering identified at least three distinct groups of users: low, moderate, and high-intensity screen users, which can be targeted with different digital wellness strategies.

---

## 6. Contact
Created by Ashwin Yadav - [ashwinyadav2408@gmail.com](mailto:ashwinyadav2408@gmail.com) - [LinkedIn](https://www.linkedin.com/in/ashwin-yadav-1704a1248/)
