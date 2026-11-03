# 🌱 BeanSort AI - Dry Bean Variety Predictor

A streamlined machine learning project that classifies seven dry bean varieties (`BARBUNYA`, `BOMBAY`, `CALI`, `DERMASON`, `HOROZ`, `SEKER`, and `SIRA`) from 16 geometric measurements using a Random Forest Classifier and an Artificial Neural Network (MLPClassifier).

## 📁 Project Structure

- `beans_dataset.csv`: Clean dataset containing 16 geometric bean features and the target `Class` column.
- `main.py`: Loads the data, prints summary statistics, generates exploratory and evaluation plots, and trains both the Random Forest and Neural Network models.
- `app.py`: Interactive Streamlit web application for comparing model accuracies, viewing plots, and predicting bean varieties from custom measurements.
- `requirements.txt`: Python package dependencies required to run the project.

## 🚀 How to Run Locally

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the training script directly (optional):**
   ```bash
   python main.py
   ```

3. **Launch the Streamlit web app:**
   ```bash
   streamlit run app.py
   ```
