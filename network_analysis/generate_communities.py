import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community import girvan_newman
from collections import defaultdict
import seaborn as sns
from collections import defaultdict, Counter
import json
import community.community_louvain as community_louvain
import re

data=pd.read_json('network_data.json')

while True:
    start_date = str(input("Specify the starting date: "))
    if re.match(r"^\d{4}\-\d{2}\-\d{2}$",start_date):
        break
    else:
        print("Please enter a valid date")

while True:
    end_date = str(input("Specify the ending date: "))
    if re.match(r"^\d{4}\-\d{2}\-\d{2}$",end_date):
        break
    else:
        print("Please enter a valid date")

while True:
    n_authors = int(input("Specify the number of authors to consider: "))
    if isinstance(n_authors,int):
        break
    else:
        print("Please enter a valid integer")

def filter_data(df,start_date,end_date):

    mask = (data['son_date'] >= start_date) & (data['son_date'] <= end_date) & (data['mother_date'] >= start_date) & (data['mother_date'] <= end_date) & (data.author_son != "AutoModerator") & (data.author_mother != "AutoModerator")

    return df[mask].reset_index(drop = True)

def define_top_authors(df, n):
    list_author_son = df.groupby("author_son")["text_son"].count().reset_index().sort_values("text_son",ascending = False).head(n).author_son.tolist()
    list_author_mother = df.groupby("author_mother")["text_mother"].count().reset_index().sort_values("text_mother",ascending = False).head(n).author_mother.tolist()
    top_authors = set(list_author_son + list_author_mother)

    return top_authors

filtered_data = filter_data(data,start_date,end_date)
top_authors = define_top_authors(filtered_data,n_authors)
cleaned_data = filtered_data[filtered_data.author_son.isin(top_authors) & (filtered_data.author_mother.isin(top_authors))].dropna(subset=['author_son', 'author_mother'])

# GENERATE THE GRAPH FROM THE DATA, WEIGHTING EDGES BY THE TOTAL NUMBER OF INTERACTIONS BETWEEN NODES (users)
G = nx.Graph()
for index, row in cleaned_data.iterrows():
    son = row['author_son']
    mother = row['author_mother']
    if G.has_edge(son, mother):
        G[son][mother]['weight'] += 1
    else:
        G.add_edge(son, mother, weight=1)

# REMOVE EDGES WITH WEIGHT 1 - if you want to

accepted = ["y","n"]
while True:
    wish_to_remove = str(input("Do you wish to remove edges between users who interacted only once ? (Y/N)"))
    if wish_to_remove.lower() in accepted:
        break
    else:
        print("Please enter a valid answer (Y/N)")

if wish_to_remove == "y":
    edges_to_remove = [edge for edge in G.edges() if G.edges[edge]["weight"] < 2]
    G.remove_edges_from(edges_to_remove)

# REMOVE ISOLATED COMPONENTS (1 or 2 nodes)
connected_components = list(nx.connected_components(G))
isolated_components = [component for component in connected_components if (len(component) == 2) or (len(component) == 1)]
nodes_to_remove = [node for component in isolated_components for node in component]
G.remove_nodes_from(nodes_to_remove)

plt.figure(figsize=(18, 12))
pos = nx.spring_layout(G, k=0.1)
edges = G.edges(data=True)
weights = [edge[2]['weight'] for edge in edges]

# GENERATE THE COMMUNITIES
partition = community_louvain.best_partition(G,weight = 'weight')
community_dict = defaultdict(list)
for node, comm in partition.items():
    community_dict[comm].append(node)

communities = list(community_dict.values())

# SAVE GENERATED COMMUNITIES
with open(f"communities{start_date}_{end_date}.json","w") as f:
    json.dump(dict(community_dict),f)

plt.figure(figsize=(16, 10))
pos = nx.spring_layout(G, k=0.2, seed=42)  # Increased k value for better separation
original_club_color = 'black'  # Assuming a default color for edges

# Graph with communities
nx.draw_networkx_nodes(G,
                       pos,
                       node_color=list(partition.values()),
                       edgecolors='black',  # Distinctive edge color for nodes
                       linewidths=1.5,
                       cmap=plt.cm.tab20,
                       node_size= [10 * G.degree(v) for v in G]  # Increased node size
                    )
nx.draw_networkx_edges(G,
                       pos,
                       alpha=0.5)
#nx.draw_networkx_labels(G, pos, font_size=12)  # Increased font size

# Add a descriptive box with the value
mod_level = community_louvain.modularity(partition, G, weight='weight')
text_str = f"Modularity = {round(mod_level,2)}\nCommunities: {len(set(partition.values()))}"
plt.xlabel(text_str, fontsize = 16)
plt.title(f"Detected Communities from {start_date} to {end_date}", fontsize=20)
plt.tight_layout(pad=5.0)
plt.savefig(f"Communities{start_date}_{end_date}.png")


# Induced graph
plt.figure(figsize = (18,12))
ind = community_louvain.induced_graph(partition, G)  
pos_ind = nx.spring_layout(ind, seed=20)
no = nx.draw_networkx_nodes(ind,
                            node_color=pd.Series(partition).drop_duplicates().values,
                            cmap=plt.cm.tab20,
                            pos=pos_ind,
                            node_size= [100 * ind.degree(v) for v in ind],  # Increased node size
                            edgecolors='black',  # Distinctive edge color for nodes
                            linewidths=1.5)
ed = nx.draw_networkx_edges(ind,
                            pos=pos_ind,
                            alpha=0.5)
# Plot the edge labels
el = nx.draw_networkx_edge_labels(ind,
                                  edge_labels=nx.get_edge_attributes(ind, 'weight'),
                                  verticalalignment='baseline',
                                  horizontalalignment='left',
                                  font_size=10,
                                  pos=pos_ind)
no.set_zorder(1)


plt.title(f"Induced Graph from {start_date} to {end_date}", fontsize = 20)
# Adjust layout to separate the plots
text_str = f"Modularity = {round(mod_level,2)}\nCommunities: {len(set(partition.values()))}"
plt.xlabel(text_str, fontsize = 16)
plt.tight_layout(pad=5.0)
plt.savefig(f"InducedGraph{start_date}_{end_date}.png")


