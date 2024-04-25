# gp-octant
Visualization for projects participating in the GreenPill Network x Octant round deployed at https://gp-octant.streamlit.app/

# Approach
Project descriptions are first converted into numerical vectors using a Sentence Transformer model. Next, UMAP reduces this high-dimensional data to a two-dimensional space, maintaining the intrinsic relationships between projects. Finally, the HDBSCAN algorithm clusters these projects based on their descriptions' similarities, identifying dense groups and distinguishing outliers, which helps in understanding the natural categorizations and thematic consistencies within the project dataset.

# Files
**"create summary.ipynb"**
Script is designed to process textual grantee project descriptions from a JSON file, generate both long and short summaries of project descriptions using the OpenAI API, and then output the results into a CSV file. 

**build cluster.py**
This script visualizes clusters of project descriptions by processing text data through a series of machine learning steps:
- Text Embedding: Utilizes a pre-trained model from sentence_transformers to convert text data into numerical embeddings. These embeddings represent the semantic meaning of the project descriptions.
- Dimensionality Reduction: Applies UMAP (Uniform Manifold Approximation and Projection) to reduce the high-dimensional embeddings into a 2D space, facilitating easier visualization and clustering.
- Clustering: Uses HDBSCAN, a density-based clustering algorithm, to group similar project descriptions into clusters based on their reduced embeddings.
- Visualization Preparation: Constructs a new DataFrame with UMAP results and cluster labels, along with project metadata.
- Data Export and Visualization: Exports the clustered data to a new CSV file and creates an interactive scatter plot using Plotly Express. This plot visually represents the clusters and allows interactive exploration of project names and descriptions based on their semantic similarities.

**radial-report**
This script creates an interactive web application using Streamlit to visualize and explore project data, consuming the UMAP results from the previous step. It allows users to select a project from a dropdown menu, computes distances between the selected project and all others to find similarities and differences and displays the three most similar and three least similar projects, providing clickable links and short descriptions for each.

<img width="1539" alt="Screenshot 2024-04-25 at 6 56 39 PM" src="https://github.com/rohitmalekar/gp-octant/assets/20112313/6887713e-3e87-4b97-aa7f-32fdc227e7f8">

<img width="1564" alt="Screenshot 2024-04-25 at 6 57 10 PM" src="https://github.com/rohitmalekar/gp-octant/assets/20112313/97a80a4b-f27e-429b-a77c-dfbfa6eb638c">


