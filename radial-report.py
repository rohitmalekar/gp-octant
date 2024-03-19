import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(layout="wide")

st.markdown("# GreenPill x Octant Community Round")
st.markdown("The [GreenPill Network](https://greenpill.network/) and [Octant](https://octant.app/) are partnering to support community-driven initiatives that impact their local communities. Select a project you are most curious about or familiar with to explore the solution space the participating projects are working in.")

col1, col2, col3, col4 = st.columns([1,1,2,2])
with col1:
    st.link_button("Explore all projects", "https://explorer.gitcoin.co/#/round/10/0xe9459565709c5e856ffbc3cc8715824945d92de7",type="primary")
with col2:
    st.link_button("About GrantsScope", "https://grantsscope.xyz",type="secondary")


# Function to re-center and convert all points based on a new center project
def recenter_coordinates(df, project_index):
    # The project to center on
    center_project = df.iloc[project_index]

    # Adjusting all points relative to the center project
    adjusted_df = df.copy()
    adjusted_df['UMAP_1'] -= center_project['UMAP_1']
    adjusted_df['UMAP_2'] -= center_project['UMAP_2']

    return adjusted_df

# Function to calculate Cartesian distances from the selected project to all others
def calculate_distances(df, project_index):
    selected_project = df.iloc[project_index]
    df['Distance'] = np.sqrt((df['UMAP_1'] - selected_project['UMAP_1'])**2 + (df['UMAP_2'] - selected_project['UMAP_2'])**2)
    return df

def make_clickable(name, link):
    return f"[{name}]({link})"

# Load your CSV file
df = pd.read_csv('./cluster_greenpill.csv')

# Streamlit widget to choose the project to center on, sorted alphabetically
project_to_center = st.selectbox('Pick a project:', sorted(df['Project Name'].unique()))


# Find the index of the selected project
project_index = df[df['Project Name'] == project_to_center].index[0]

# Display the selected project's description
selected_project_desc = df.loc[project_index, 'Project Desc']
center_project_link = df.loc[project_index, 'Link']
st.markdown(f"About [{project_to_center}]({center_project_link}):")
st.info(selected_project_desc)

# Calculate distances from the selected project to all others
df_with_distances = calculate_distances(df, project_index)

# Exclude the selected project itself from the comparison
df_with_distances = df_with_distances[df_with_distances['Project Name'] != project_to_center]

# Sort by distance to get the most similar and most different projects
top_3_similar = df_with_distances.nsmallest(3, 'Distance')[['Project Name', 'Short Project Desc', 'Link']]
top_3_different = df_with_distances.nlargest(3, 'Distance')[['Project Name', 'Short Project Desc', 'Link']]

# Apply the function to your dataframes
top_3_similar['Project Name'] = top_3_similar.apply(lambda x: make_clickable(x['Project Name'], x['Link']), axis=1)
top_3_different['Project Name'] = top_3_different.apply(lambda x: make_clickable(x['Project Name'], x['Link']), axis=1)

# Use Streamlit's columns to display tables side by side
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"### 3 most similar projects to {project_to_center}:")
    for _, row in top_3_similar.reset_index(drop=True).iterrows():
        st.markdown(f"- {row['Project Name']}: {row['Short Project Desc']}")

with col2:
    st.markdown(f"### 3 most distinct projects to {project_to_center}:")
    for _, row in top_3_different.reset_index(drop=True).iterrows():
        st.markdown(f"- {row['Project Name']}: {row['Short Project Desc']}")

# Recenter and convert the data based on the selected project
recentered_df = recenter_coordinates(df, project_index)

st.markdown(f"## Interactive Scatter Plot Centered on {project_to_center}")
st.markdown("Hover over the project for a short description. Zoom in to explore interesting clusters. Double-click to zoom out.")

# Create the scatter plot with specified width and height
fig = px.scatter(recentered_df, x='UMAP_1', y='UMAP_2', 
                 color='Cluster',
                 text='Project Name',
                 hover_data={'UMAP_1': False, 'UMAP_2': False, 'Project Name': True, 'Short Project Desc': True, 'Cluster': False},
                 color_continuous_scale=px.colors.sequential.Viridis,
                 width=1200,  # Specify the width here
                 height=900)  # Specify the height here


# Adjust axes to include 0,0 at the center and update other plot elements
x_range = [recentered_df['UMAP_1'].min(), recentered_df['UMAP_1'].max()]
y_range = [recentered_df['UMAP_2'].min(), recentered_df['UMAP_2'].max()]
fig.update_xaxes(range=[min(x_range[0], -x_range[1]), max(-x_range[0], x_range[1])])
fig.update_yaxes(range=[min(y_range[0], -y_range[1]), max(-y_range[0], y_range[1])])
fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers+text', text=[""],
                         marker=dict(symbol='x', size=12, color='orange')))

fig.update_traces(textfont_size=14, textposition='top center', marker=dict(size=13))

# Display the plot in Streamlit
st.plotly_chart(fig, use_container_width=True)

st.caption("Technical Notes: Project descriptions are first converted into numerical vectors using a Sentence Transformer model. \
            Next, UMAP reduces this high-dimensional data to a two-dimensional space, maintaining the intrinsic relationships between projects.\
            Finally, the HDBSCAN algorithm clusters these projects based on their descriptions' similarities, \
            identifying dense groups and distinguishing outliers, which helps in understanding the natural categorizations and thematic consistencies within the project dataset.")

