import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Title of the app
st.title("Iris Dataset Exploration")

# Display the raw data
st.subheader("Raw Data")
st.write(df)

# Basic operations: Calculate mean sepal length per species
st.subheader("Mean Sepal Length by Species")
mean_sepal_length = df.groupby('species')['sepal length (cm)'].mean()
st.write(mean_sepal_length)

# Visualization: Create a scatter plot
st.subheader("Scatter Plot: Sepal Length vs. Sepal Width")
st.scatter_chart(df, x='sepal length (cm)', y='sepal width (cm)', color='species')
