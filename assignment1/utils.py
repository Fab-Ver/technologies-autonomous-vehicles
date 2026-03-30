import json
import os
import re
import osmnx as ox
import matplotlib.pyplot as plt

def compact_json(data):
    """Serialize data to JSON with objects indented but arrays on a single line."""
    raw = json.dumps(data, indent=2)
    # Collapse any array that spans multiple lines into a single line
    raw = re.sub(
        r'\[\n\s+((?:[^\[\]]*?\n\s+)*[^\[\]]*?)\]',
        lambda m: '[' + ', '.join(t.rstrip(',') for t in m.group(1).split()) + ']',
        raw,
    )
    return raw

def reset_graph(G):
    """Reset graph attributes for a new run."""
    for node in G.nodes:
        G.nodes[node]["visited"] = False
        G.nodes[node]["distance"] = float("inf")
        G.nodes[node]["previous"] = None
        G.nodes[node]["size"] = 0
    for edge in G.edges:
        style_unvisited_edge(G, edge)

def get_global_max_speed(G):
    """Find the maximum speed limit in the graph, used for normalizing heuristic values in A*."""
    max_speed = 1
    for u, v, data in G.edges(data=True):
        if "maxspeed" in data and data["maxspeed"] > max_speed:
            max_speed = data["maxspeed"]
    return max_speed / 3.6  # convert km/h to m/s

def compute_weights(G):
    """Compute the "weight" attribute for each edge based on its length and maxspeed."""
    for edge in G.edges:
        # Cleaning the "maxspeed" attribute
        maxspeed = 40
        if "maxspeed" in G.edges[edge]:
            maxspeed = G.edges[edge]["maxspeed"]
            if isinstance(maxspeed, list):
                speeds = [int(s) if s != "walk" else 1 for s in maxspeed]
                maxspeed = min(speeds)
            elif isinstance(maxspeed, str):
                if maxspeed == "walk":
                    maxspeed = 1
                else:
                    # take the numeric part only
                    maxspeed = int(maxspeed.split()[0])
        G.edges[edge]["maxspeed"] = maxspeed
        # Adding the "weight" attribute (time = distance / speed)
        maxspeed_ms = maxspeed * 1000 / 3600  # convert km/h to m/s
        G.edges[edge]["weight"] = G.edges[edge]["length"] / maxspeed_ms

def style_unvisited_edge(G, edge):        
    G.edges[edge]["color"] = "gray"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 0.2

def style_visited_edge(G, edge):
    G.edges[edge]["color"] = "green"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 1

def style_active_edge(G, edge):
    G.edges[edge]["color"] = "red"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 1

def style_path_edge(G, edge):
    G.edges[edge]["color"] = "white"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 5

def reconstruct_path(G, orig, dest, plot=False, algorithm=None, filepath=None):
    for edge in G.edges:
        style_unvisited_edge(G, edge)
    dist = 0
    travel_time = 0
    curr = dest
    while curr != orig:
        prev = G.nodes[curr]["previous"]
        dist += G.edges[(prev, curr, 0)]["length"]
        travel_time += G.edges[(prev, curr, 0)]["weight"]
        style_path_edge(G, (prev, curr, 0))
        if algorithm:
            G.edges[(prev, curr, 0)][f"{algorithm}_uses"] = G.edges[(prev, curr, 0)].get(f"{algorithm}_uses", 0) + 1
        curr = prev
    dist /= 1000
    if plot:
        plot_graph(G, filepath=filepath)
    return dist, travel_time

def plot_graph(G, filepath=None):
    os.makedirs(os.path.dirname(filepath), exist_ok=True) if filepath else None
    fig, ax = ox.plot_graph(
        G,
        node_size =  [ G.nodes[node]["size"] for node in G.nodes ],
        edge_color = [ G.edges[edge]["color"] for edge in G.edges ],
        edge_alpha = [ G.edges[edge]["alpha"] for edge in G.edges ],
        edge_linewidth = [ G.edges[edge]["linewidth"] for edge in G.edges ],
        node_color = "white",
        bgcolor = "black",
        show = False,
        save = False,
    )
    if filepath:
        fig.savefig(filepath, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    else:
        plt.show()
    plt.close("all")