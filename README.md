import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Clustering", layout="centered")

st.title("🛍️ Customer Segmentation using K-Means Clustering")

# Upload CSV
uploaded_file = st.file_uploader("Upload customer purchase history CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Data")
    st.write(df.head())

    if st.checkbox("Standardize the data? (Recommended)", value=True):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df.select_dtypes(include='number'))
    else:
        scaled_data = df.select_dtypes(include='number').values

    st.subheader("Select Number of Clusters (k)")
    k = st.slider("Choose K", 2, 10, value=3)

    model = KMeans(n_clusters=k, random_state=42)
    clusters = model.fit_predict(scaled_data)
    df['Cluster'] = clusters

    # PCA for 2D Visualization
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(scaled_data)
    df['PCA1'] = reduced[:, 0]
    df['PCA2'] = reduced[:, 1]

    st.subheader("Cluster Visualization (PCA)")
    fig, ax = plt.subplots()
    for cluster in range(k):
        cluster_data = df[df['Cluster'] == cluster]
        ax.scatter(cluster_data['PCA1'], cluster_data['PCA2'], label=f"Cluster {cluster}")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Clustered Data Sample")
    st.write(df.head())

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Clustered Data", data=csv, file_name="clustered_customers.csv", mime="text/csv")
else:
    st.info("Please upload a CSV file to proceed.")
