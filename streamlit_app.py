import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

iris = pd.read_csv('iris.csv')

st.title("Visualisation des données Iris")
st.write("Voici les premières lignes des données Iris :")
st.dataframe(iris.head())


st.write("### Statistiques descriptives :")
st.dataframe(iris.describe())

# Sélection de colonnes pour visualisation
x_axis = st.selectbox("Choisissez la colonne pour l'axe X", iris.columns)
y_axis = st.selectbox("Choisissez la colonne pour l'axe Y", iris.columns)
hue_option = st.selectbox("Choisissez une colonne pour la coloration (hue)", iris.columns)

# Visualisation : Scatter plot
st.write("### Graphique de dispersion :")
fig, ax = plt.subplots()
sns.scatterplot(data=iris, x=x_axis, y=y_axis, hue=hue_option, ax=ax)
plt.title(f"Scatter Plot : {y_axis} vs {x_axis}")
st.pyplot(fig)

# Histogramme
st.write("### Distribution des données (Histogramme) :")
selected_column = st.selectbox("Choisissez une colonne pour l'histogramme", iris.columns)
fig2, ax2 = plt.subplots()
sns.histplot(iris[selected_column], kde=True, ax=ax2)
plt.title(f"Histogramme de la colonne {selected_column}")
st.pyplot(fig2)

# Corrélation
if st.checkbox("Afficher la matrice de corrélation"):
    st.write("### Matrice de corrélation :")
    corr_matrix = iris[:, 1:4].corr()
    st.dataframe(corr_matrix)
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)   
# import streamlit as st
# import pandas as pd
# from sklearn.datasets import load_iris

# # Load the Iris dataset
# iris = load_iris()
# df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# # Title of the app
# st.title("Iris Dataset Exploration")

# # Display the raw data
# st.subheader("Raw Data")
# st.write(df)

# # Basic operations: Calculate mean sepal length per species
# st.subheader("Mean Sepal Length by Species")
# mean_sepal_length = df.groupby('species')['sepal length (cm)'].mean()
# st.write(mean_sepal_length)

# # Visualization: Create a scatter plot
# st.subheader("Scatter Plot: Sepal Length vs. Sepal Width")
# st.scatter_chart(df, x='sepal length (cm)', y='sepal width (cm)', color='species')
