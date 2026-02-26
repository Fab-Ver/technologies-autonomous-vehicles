# Loads a driving network from OpenStreetMap (osmnx), computes edge weights as travel time
# (length / maxspeed), and runs Dijkstra's shortest-path algorithm (heapq) between 10 random
# node pairs per city. Tracks iterations per run, computes the average, and saves
# graph visualizations to the plots/ folder. Cities: Aosta, Turin.

import argparse
import os
import osmnx as ox
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import heapq

NUM_RUNS = 10   # number of random pairs per city

PLACES = {
    "Aosta": "Aosta, Aosta, Italy",
    "Turin": "Turin, Piedmont, Italy",
}

def style_unvisited_edge(edge):        
    G.edges[edge]["color"] = "gray"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 0.2

def style_visited_edge(edge):
    G.edges[edge]["color"] = "green"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 1

def style_active_edge(edge):
    G.edges[edge]["color"] = "red"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 1

def style_path_edge(edge):
    G.edges[edge]["color"] = "white"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 5

def clean_maxspeed():
    for edge in G.edges:
        # Cleaning the "maxspeed" attribute, some values are lists, some are strings, some are None
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
                    # take the numeric part only (handles "50", "50 mph", "50 km/h", etc.)
                    maxspeed = int(maxspeed.split()[0])
        G.edges[edge]["maxspeed"] = maxspeed
        # Adding the "weight" attribute (time = distance / speed)
        G.edges[edge]["weight"] = G.edges[edge]["length"] / maxspeed

def plot_graph(filepath=None):
    os.makedirs(os.path.dirname(filepath), exist_ok=True) if filepath else None
    ox.plot_graph(
        G,
        node_size =  [ G.nodes[node]["size"] for node in G.nodes ],
        edge_color = [ G.edges[edge]["color"] for edge in G.edges ],
        edge_alpha = [ G.edges[edge]["alpha"] for edge in G.edges ],
        edge_linewidth = [ G.edges[edge]["linewidth"] for edge in G.edges ],
        node_color = "white",
        bgcolor = "black",
        show = filepath is None,
        save = filepath is not None,
        filepath = filepath,
    )
    plt.close("all")

def plot_heatmap(filepath):
    """Color edges by dijkstra_uses frequency using a heatmap colormap."""
    uses = [G.edges[e]["dijkstra_uses"] for e in G.edges]
    max_uses = max(uses) if max(uses) > 0 else 1
    norm = mcolors.Normalize(vmin=0, vmax=max_uses)
    cmap = plt.get_cmap("plasma")
    edge_colors = []
    edge_widths = []
    for e in G.edges:
        u = G.edges[e]["dijkstra_uses"]
        edge_colors.append(mcolors.to_hex(cmap(norm(u))) if u > 0 else "#1a1a1a")
        edge_widths.append(0.5 + 4.5 * norm(u))  # 0.5 (unused) → 5.0 (max used)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    ox.plot_graph(
        G,
        node_size=0,
        edge_color=edge_colors,
        edge_linewidth=edge_widths,
        edge_alpha=1,
        node_color="white",
        bgcolor="black",
        show=False,
        save=True,
        filepath=filepath,
    )
    plt.close("all")

def dijkstra(orig, dest):
    # Reset all node and edge state — ensures no contamination between runs
    for node in G.nodes:
        G.nodes[node]["visited"] = False
        G.nodes[node]["distance"] = float("inf")
        G.nodes[node]["previous"] = None
        G.nodes[node]["size"] = 0
    for edge in G.edges:
        style_unvisited_edge(edge)
    G.nodes[orig]["distance"] = 0
    G.nodes[orig]["size"] = 50
    G.nodes[dest]["size"] = 50
    pq = [(0, orig)]
    step = 0
    while pq:
        _, node = heapq.heappop(pq)
        if node == dest:
            return step
        if G.nodes[node]["visited"]: continue
        G.nodes[node]["visited"] = True
        for edge in G.out_edges(node):
            style_visited_edge((edge[0], edge[1], 0))
            neighbor = edge[1]
            weight = G.edges[(edge[0], edge[1], 0)]["weight"]
            if G.nodes[neighbor]["distance"] > G.nodes[node]["distance"] + weight:
                G.nodes[neighbor]["distance"] = G.nodes[node]["distance"] + weight
                G.nodes[neighbor]["previous"] = node
                heapq.heappush(pq, (G.nodes[neighbor]["distance"], neighbor))
                for edge2 in G.out_edges(neighbor):
                    style_active_edge((edge2[0], edge2[1], 0))
        step += 1
    return None # No path found

def reconstruct_path(orig, dest, plot=False, algorithm=None, filepath=None):
    for edge in G.edges:
        style_unvisited_edge(edge)
    dist = 0
    curr = dest
    while curr != orig:
        prev = G.nodes[curr]["previous"]
        dist += G.edges[(prev, curr, 0)]["length"]
        style_path_edge((prev, curr, 0))
        if algorithm:
            G.edges[(prev, curr, 0)][f"{algorithm}_uses"] = G.edges[(prev, curr, 0)].get(f"{algorithm}_uses", 0) + 1
        curr = prev
    dist /= 1000
    if plot:
        plot_graph(filepath=filepath)
    return dist

def main():
    global G
    parser = argparse.ArgumentParser(description="Run Dijkstra on OSM city graphs.")
    parser.add_argument(
        "-n", "--runs",
        type=int,
        default=NUM_RUNS,
        help=f"Number of random pairs per city (default: {NUM_RUNS})",
    )
    args = parser.parse_args()
    num_runs = args.runs

    os.makedirs("plots", exist_ok=True)

    for city_name, place_query in PLACES.items():
        city_abbr = city_name.lower()
        print(f"\n{'='*55}")
        print(f"City : {city_name}")
        print(f"Query: {place_query}")
        print(f"Loading graph...")
        G = ox.graph_from_place(place_query, network_type="drive")
        clean_maxspeed()
        for edge in G.edges:
            G.edges[edge]["dijkstra_uses"] = 0
        print(f"Graph loaded: {len(G.nodes):,} nodes, {len(G.edges):,} edges")

        i = 0
        tot_steps = []
        while i < num_runs:
            start = random.choice(list(G.nodes))
            end = random.choice(list(G.nodes))
            steps = dijkstra(start, end)
            if steps is not None:
                i += 1
                tot_steps.append(steps)
                fp = f"plots/dijkstra_run{i:02d}_{city_abbr}.png"
                reconstruct_path(start, end, plot=True, algorithm="dijkstra", filepath=fp)
                print(f"  Run {i:2d}/{num_runs} | Iterations: {steps:6d} | Plot saved: {fp}")
            else:
                print(f"  Run {i+1:2d}/{num_runs} | No path found between selected nodes, retrying...")

        avg = sum(tot_steps) / len(tot_steps)
        print(f"\n  Steps per run : {tot_steps}")
        print(f"  Average steps : {avg:.1f}")

        hm_fp = f"plots/dijkstra_heatmap_{city_abbr}.png"
        plot_heatmap(hm_fp)
        print(f"  Heatmap saved : {hm_fp}")


if __name__ == "__main__":
    main()

