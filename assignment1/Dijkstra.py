# Loads a driving network from OpenStreetMap (osmnx), computes edge weights as travel time
# (length / maxspeed), and runs Dijkstra's shortest-path algorithm (heapq) between 10 random
# node pairs per city. Tracks iterations per run, computes the average, and saves
# graph visualizations to the plots/ folder. Cities: Aosta, Turin.

import argparse
import json
import osmnx as ox
import random
import heapq
import os 
from  utils import compute_weights, style_visited_edge, style_active_edge, reset_graph, reconstruct_path, compact_json

NUM_RUNS = 10   # number of random pairs per city

PLACES = {
    "Aosta": "Aosta, Aosta, Italy",
    "Turin": "Turin, Piedmont, Italy",
}

def dijkstra(G, orig, dest):
    """Run Dijkstra's algorithm from orig to dest, return number of iterations."""
    reset_graph(G)
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
            style_visited_edge(G, (edge[0], edge[1], 0))
            neighbor = edge[1]
            weight = G.edges[(edge[0], edge[1], 0)]["weight"]
            if G.nodes[neighbor]["distance"] > G.nodes[node]["distance"] + weight:
                G.nodes[neighbor]["distance"] = G.nodes[node]["distance"] + weight
                G.nodes[neighbor]["previous"] = node
                heapq.heappush(pq, (G.nodes[neighbor]["distance"], neighbor))
                for edge2 in G.out_edges(neighbor):
                    style_active_edge(G, (edge2[0], edge2[1], 0))
        step += 1
    return None # No path found

def main():
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
    os.makedirs("results", exist_ok=True)

    results_fp = "results/results.json"
    all_results = {}

    for city_name, place_query in PLACES.items():
        city_abbr = city_name.lower()
        print(f"\n{'='*55}")
        print(f"  CITY  : {city_name}")
        print(f"  Query : {place_query}")
        print(f"  Loading graph...")
        G = ox.graph_from_place(place_query, network_type="drive")
        compute_weights(G)  # compute "weight" attribute for each edge based on length and maxspeed
        for edge in G.edges:
            G.edges[edge]["dijkstra_uses"] = 0
        print(f"  Graph : {len(G.nodes):,} nodes, {len(G.edges):,} edges")
        print(f"\n  {'─'*49}")
        print(f"  Algorithm : Dijkstra")
        print(f"  {'─'*49}")

        node_list = list(G.nodes)
        pairs = []       # valid (start, end) pairs, saved to file for A*
        run_results = [] # per-run iteration counts
        i = 0
        while i < num_runs:
            start = random.choice(node_list)
            end   = random.choice(node_list)
            steps = dijkstra(G, start, end)
            if steps is not None:
                i += 1
                pairs.append([start, end])
                run_results.append(steps)
                city_acronym = "D" + city_name[:2].upper()
                fp = f"plots/dijkstra/{city_abbr}/run{i:02d}_{city_acronym}.png"
                dist, travel_time = reconstruct_path(G, start, end, plot=True, filepath=fp)
                run_results[-1] = {"iterations": steps, "distance_km": round(dist, 4), "travel_time_s": round(travel_time, 2)}
                print(f"    Run {i:2d}/{num_runs} | Iterations: {steps:6d} | Distance: {dist:.2f} km | Time: {travel_time:.1f} s | Plot saved: {fp}")
            else:
                print(f"    Run {i+1:2d}/{num_runs} | No path found, retrying...")

        avg = sum(r["iterations"] for r in run_results) / len(run_results)
        print(f"\n    Steps per run       : {[r['iterations'] for r in run_results]}")
        print(f"    Avg distance        : {sum(r['distance_km'] for r in run_results) / len(run_results):.2f} km")
        print(f"    Avg travel time (s) : {sum(r['travel_time_s'] for r in run_results) / len(run_results):.1f} s")
        print(f"    Average steps       : {avg:.1f}")

        # Save valid pairs for A* reuse
        pairs_fp = f"results/pairs_{city_abbr}.json"
        with open(pairs_fp, "w") as f:
            json.dump(pairs, f, indent=2)
        print(f"    Pairs saved   : {pairs_fp}")

        # Accumulate results
        all_results[city_name] = {
            "nodes": len(G.nodes),
            "edges": len(G.edges),
            "results": {
                "dijkstra": {
                    "iterations":    [r["iterations"]    for r in run_results],
                    "distance_km":   [r["distance_km"]   for r in run_results],
                    "travel_time_s": [r["travel_time_s"] for r in run_results],
                    "average":       round(avg, 2),
                }
            }
        }

    with open(results_fp, "w") as f:
        f.write(compact_json(all_results))
    print(f"\n  Results saved : {results_fp}")


if __name__ == "__main__":
    main()

