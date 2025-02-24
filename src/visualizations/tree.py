import sys

sys.path.append("src")
sys.path.append("")
import datetime
import uuid
import random
import itertools
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, deque
from base import initialize_session, Scaffold, Population
import matplotlib.colors as mcolors
from dotenv import load_dotenv

load_dotenv(override=True)


def plot_tree(scaffolds):
    """
    Plot a multi-layer graph from a list of Scaffold objects.
    Each unique 'generation_timestamp' becomes its own layer.
    Each 'cluster_id' is represented by a different color.
    Node labels are shown as 'scaffold_name' rather than 'scaffold_id'.
    Node sizes are proportional to scaffold_capability_ci_median (0 to 1).
    If a node has scaffold_capability_ci_median = -1 or None, treat it as 0.

    Labels for each layer are alternately drawn above and below nodes
    to reduce overlap of long names.
    """

    # 1. Collect and sort the distinct generation timestamps
    unique_gen_timestamps = sorted({s.generation_timestamp for s in scaffolds})

    # 2. Map each timestamp to an integer "layer" index
    timestamp_to_layer = {
        gen_ts: idx for idx, gen_ts in enumerate(unique_gen_timestamps)
    }

    # 3. Initialize the graph
    G = nx.Graph()

    # 4. Add nodes with layer, cluster_id, and fitness attributes
    for sys in scaffolds:
        # Handle fitness edge cases
        if (
            sys.scaffold_capability_ci_median is None
            or sys.scaffold_capability_ci_median == -1
        ):
            fitness_val = 0.0
        else:
            fitness_val = sys.scaffold_capability_ci_median  # assumed in [0, 1]

        G.add_node(
            sys.scaffold_id,  # node identifier = scaffold_id
            layer=timestamp_to_layer[sys.generation_timestamp],
            cluster_id=sys.cluster_id,  # for color grouping
            fitness=fitness_val,  # store normalized fitness
        )

    # 5. Add edges from parents to child (if they exist in this population)
    scaffold_ids = set(s.scaffold_id for s in scaffolds)
    for sys in scaffolds:
        if (
            sys.scaffold_first_parent_id
            and sys.scaffold_first_parent_id in scaffold_ids
        ):
            G.add_edge(sys.scaffold_first_parent_id, sys.scaffold_id)
        if (
            sys.scaffold_second_parent_id
            and sys.scaffold_second_parent_id in scaffold_ids
        ):
            G.add_edge(sys.scaffold_second_parent_id, sys.scaffold_id)

    # 6. Use the built-in multipartite layout keyed by 'layer'
    pos = nx.multipartite_layout(G, subset_key="layer")

    # 7. Assign colors by cluster_id

    all_clusters = list({G.nodes[n]["cluster_id"] for n in G.nodes()})

    cmap = plt.cm.get_cmap("rainbow", len(all_clusters))
    node_color = []
    for n in G.nodes():
        cluster = G.nodes[n]["cluster_id"]
        cluster_index = all_clusters.index(cluster)
        node_color.append(cmap(cluster_index))

    # 8. Create a label dictionary mapping scaffold_id -> scaffold_name
    label_dict = {sys.scaffold_id: sys.scaffold_name for sys in scaffolds}

    # 9. Create a node_size list based on fitness
    #    e.g. map fitness=0 to size=100, fitness=1 to size=2100
    node_size = []
    for n in G.nodes():
        f = G.nodes[n]["fitness"]  # 0.0 <= f <= 1.0
        size = 100 + 2000 * f  # scale the node size
        node_size.append(size)

    # 10. Draw the graph WITHOUT labels
    plt.figure(figsize=(12, 8))
    nx.draw(
        G,
        pos,
        node_color=node_color,
        edge_color="#ddddd4",
        with_labels=False,  # don't draw labels here
        font_size=8,
        node_size=node_size,
    )

    # 11. Prepare a custom position dictionary for labels
    #     that shifts them above/below the node per layer
    label_pos = {}
    for node, data in G.nodes(data=True):
        x, y = pos[node]
        layer_idx = data["layer"]
        # If layer is even, move label above (+), else below (-)
        offset = 0.07 if (layer_idx % 2 == 0) else -0.07
        # Create a new position with y-offset
        label_pos[node] = (x, y + offset)

    # 12. Draw labels separately, at the offset positions
    nx.draw_networkx_labels(
        G,
        label_pos,
        labels=label_dict,
        font_size=8,
    )

    plt.title(
        "Scaffolds by Generation Timestamp (colored by cluster_id, sized by fitness)"
    )
    plt.show()


if __name__ == "__main__":
    random.seed(42)
    population_id = "d44a351c-d454-4d2c-ae74-f7e0e88b9ce8"
    population_id = "dd43526d-9a36-41c3-89bb-2f71c7738040"

    for session in initialize_session():
        population = (
            session.query(Population)
            .order_by(Population.population_timestamp.desc())
            .limit(2)
            .all()
        )[0]
        population_id = population.population_id

        print(population_id)
        # /home/#/Documents/AgentBreeder/src/logs/test/20250129-083111/GPQA-13e8fb8c-4d16-4487-85ce-e40249da8422
        population_id = "cfda0d48-e4aa-439b-a348-b1433d27d344"

        # Suppose you have a list of scaffolds from your DB:
        scaffolds = session.query(Scaffold).filter_by(population_id=population_id).all()
        print(len(scaffolds))
        print(scaffolds[0])
        print(population_id)
        plot_tree(scaffolds)


# def assign_generations(scaffolds):
#     base_generation = scaffolds[:7]
#     generation_timestamp = datetime.datetime.utcnow()
#     for scaffold in base_generation:
#         print(scaffold.scaffold_name, scaffold.scaffold_id)
#         scaffold.update(generation_timestamp=generation_timestamp)

#     other_generations = scaffolds[7:]

#     # each generation has 10 scaffolds
#     for i in range(5):
#         generation = other_generations[i * 10 : (i + 1) * 10]
#         generation_timestamp = datetime.datetime.utcnow()
#         for scaffold in generation:
#             print(scaffold.scaffold_name, scaffold.scaffold_id)
#             scaffold.update(generation_timestamp=generation_timestamp)
