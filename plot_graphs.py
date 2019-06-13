#
## save figures
#
#for user_id, G in G_dict.items():
#    print(user_id)
#    # edge color management
#    weights = [G[u][v]['weight']+1 for u,v in G.edges()]
#    norm_width = np.log(weights)*2
#
#    deg = nx.degree(G)
#    node_sizes = [5 * deg[iata] for iata in G.nodes]
#    
#    # draw geographic representation
#    ax, smap = draw_smopy_basemap(G)
#    nx.draw_networkx(G, ax=ax,
#                 font_size=20,
#                 width=1,
#                 edge_width=norm_width,
#                 with_labels=False,
#                 node_size=node_sizes,
#                 pos=nx_coordinate_layout_smopy(G,smap))
#    
#
#    filename = os.path.join(IMAGE_OUTPUT, str(user_id) + "_coordinate_layout" + ".png")
#    plt.savefig(filename)
#    plt.close()
#    
#    # draw spring layout 
#    plt.figure()
#    pos = nx.spring_layout(G)
#    nx.draw(G, pos=pos, width=norm_width/2, node_size=node_sizes)
#    filename = os.path.join(IMAGE_OUTPUT, str(user_id) + "_spring_layout" + ".png")
#    plt.savefig(filename)
#    plt.close()
#    
    
