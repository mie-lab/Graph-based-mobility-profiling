import pickle
from future_trackintel.activity_graph import activity_graph
import os

AG_dict = pickle.load(open(os.path.join(".", "data_out", "graph_data", "gc1", "counts_full.pkl"), 'rb'))

output_spring = os.path.join(".", "graph_images", "gc1", "spring")
if not os.path.exists(output_spring):
    os.makedirs(output_spring)

output_coords = os.path.join(".", "graph_images", "gc1", "coords")
if not os.path.exists(output_coords):
    os.makedirs(output_coords)

for user_id_this, AG in AG_dict.items():

    AG.plot(filename=os.path.join(output_spring, str(user_id_this)),
        filter_node_importance=25,
        draw_edge_label=False,
    )
    AG.plot(filename=os.path.join(output_coords, str(user_id_this)),
        filter_node_importance=25,
        draw_edge_label=False,
        layout="coordinate",
    )
G = AG.G