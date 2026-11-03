import sys
import pandas as pd
import streamlit as st

sys.dont_write_bytecode = True  # don't create a __pycache__ folder when importing main.py


with st.spinner("Training the models, please wait..."):
    import main

st.title("🌱 BeanSort AI - Dry Bean Variety Predictor")


st.subheader("Which model is better?")
st.table(pd.DataFrame(main.results.items(), columns=["Model", "Accuracy (%)"]))
st.success(f"Winner: {main.winner}")


st.subheader("Plots")
plots = {
    "Bean variety counts": main.fig1,
    "Area by variety": main.fig2,
    "Roundness by variety": main.fig3,
    "Model accuracy": main.fig4,
}
choice = st.selectbox("Choose a plot", list(plots.keys()))
st.pyplot(plots[choice])


st.subheader("Predict a bean variety")
col1, col2 = st.columns(2)
user_input = {}

for i, col in enumerate(main.X.columns):
    min_val = float(main.X[col].min())
    max_val = float(main.X[col].max())
    med_val = float(main.X[col].median())
    target_col = col1 if i % 2 == 0 else col2
    user_input[col] = target_col.slider(col, min_val, max_val, med_val)

if st.button("Predict"):
    sample = pd.DataFrame([user_input])
    sample_scaled = main.scaler.transform(sample)

    st.write(f"Random Forest: **{main.rf.predict(sample_scaled)[0]}**")
    st.write(f"Neural Network: **{main.nn.predict(sample_scaled)[0]}**")
