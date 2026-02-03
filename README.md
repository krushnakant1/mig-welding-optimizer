# mig-welding-optimizer

# MIG Welding Optimizer (Single-File Edition) 👨‍🏭⚡

**A standalone AI tool for optimizing 316L Austenitic ↔ 430 Ferritic Stainless Steel joints.**

This application is a complete, single-file solution that combines a Python Machine Learning backend with a modern, responsive web interface. It acts as a "Digital Foreman" to predict weld quality and ensure safety standards are met for dissimilar metal joining.

## 🚀 Key Features

* **Zero-Config Deployment:** Entire app (Backend + Frontend) lives in one `.py` file.
* **Machine Learning Engine:** Uses **Random Forest Regressors** (via Scikit-Learn) to predict:
    * **Tensile Strength (MPa)**
    * **Penetration Depth (mm)**
* **Safety "Guardrails":**
    * Real-time **Heat Input (HI)** monitoring ($HI = \frac{V \times I \times \eta}{S}$).
    * Automatic alerts if HI > **1.2 kJ/mm** (Critical for preventing grain coarsening in 430 Ferritic SS).
    * 
* **Optimizer Module:** A grid-search system that finds the "Sweet Spot" parameters (Current/Speed) that maximize strength while respecting safety constraints.

## 🛠️ Technical Architecture

* **Backend:** Python `http.server` (No Flask/Django required) + Scikit-Learn.
* **Frontend:** Embedded HTML5, CSS3 (Dark Mode), and Vanilla JavaScript.
* **Data Handling:** Pandas & NumPy for vectorizing welding parameters.

## 📋 Prerequisites

You only need Python installed and the data science libraries:

```bash
pip install pandas numpy scikit-learn

🏃‍♂️ How to Run
1. Save the script: Save the code as mig_optimizer.py.

2. Run the application:

Bash
python mig_optimizer.py

3. Access the UI: The script will automatically open your default web browser to: http://localhost:8000



## **🧠 Workflow**
1. Train:

Go to the "Train Model" tab.

Click "Generate Demo Data" to simulate 80+ welding experiments instantly.

Click "Train Model" to build the Random Forest.

2. Predict:

Switch to "Predict Quality".

Adjust Current, Voltage, and Speed.

Watch the Heat Input Bar update in real-time.
<img width="3999" height="2283" alt="image" src="https://github.com/user-attachments/assets/6d993097-7089-4233-b992-c1c7178e39e1" />

3. Optimize:

Go to "Optimize Parameters".

Set your constraints (e.g., Max Heat Input 1.2 kJ/mm).

Click "Run Optimization" to find the best settings automatically.


## **📂 Project Structure**
Since this is a single-file solution, the structure is minimal:
/
├── mig_optimizer.py    # Contains Server, ML Logic, and HTML/CSS/JS
├── requirements.txt    # Python dependencies
└── data/               # (Optional) Folder if you export/import CSVs

## **🤝 Contributing**
Feel free to fork this repo. Future improvements could include:

- Adding joblib to save trained models to disk.

- Expanding the dataset to include varying plate thicknesses (currently optimized for 3mm).

## **📄 License**
MIT License
---

### 2. `requirements.txt`
Since you are using the standard library for the server (`http.server`), you only need to list the data science packages.

```text
pandas>=1.5.0
numpy>=1.20.0
scikit-learn>=1.0.0

3. Usage Note
Since you are using webbrowser.open and http.server, this script is designed to run locally on your machine. If you plan to deploy this to a cloud server (like Heroku or AWS), you would typically need to swap the http.server part for a production WSGI framework like Flask or FastAPI, as http.server is not designed for production security or load handling.

For a student project, thesis demo, or internal tool, this implementation is perfect.
