# aligned_cayley

def draw_aligned_graphs(g1, g2, layout=nx.spring_layout, shift_x=2.0, special_edges=[], filename='aligned_cayley.pgf'):

    pos_g1 = layout(g1)
    pos_g2 = {node: (x + shift_x, y) for node, (x, y) in layout(g2).items()}

    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    # left graph
    nx.draw(g1, pos=pos_g1, with_labels=True, node_color='white', edge_color='k', ax=ax, edgecolors ='black', node_size=50)
    
    # right graph
    nx.draw_networkx_nodes(g2, pos=pos_g2, node_color='white', ax=ax, edgecolors ='black', node_size=50)
    nx.draw_networkx_labels(g2, pos=pos_g2, ax=ax)
    
    # draw normal and special edges with different styles
    normal_edges = [edge for edge in g2.edges() if edge not in special_edges]
    nx.draw_networkx_edges(g2, pos=pos_g2, edgelist=normal_edges, edge_color='k', ax=ax)
    nx.draw_networkx_edges(g2, pos=pos_g2, edgelist=special_edges, edge_color='k', width=5, ax=ax)

    # draw lines for alignments
    for node in g1.nodes():
        if node in pos_g1 and node in pos_g2:
            start_pos = pos_g1[node]
            end_pos = pos_g2[node]
            plt.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], "k--", linewidth=0.5, zorder=-1, color='gray', alpha=0.5)

    plt.axis('equal')
    plt.axis('off')  # Optionally turn off the axis for a cleaner look
    plt.savefig(filename, bbox_inches='tight')
    plt.show()

# cayley_clusters

def draw_aligned_graphs(g1, g2, layout=nx.spring_layout, shift_x=2.0, filename='cayley_clusters.pgf'):

    pos_g1 = layout(g1)
    pos_g2 = {node: (x + shift_x, y) for node, (x, y) in layout(g2).items()}

    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # left graph
    nx.draw(g1, pos=pos_g1, node_color=node_colours, edge_color='k', ax=ax, node_size=50, edgecolors ='black')

    # right graphs
    nx.draw(g2, pos=pos_g2, node_color=node_colours, edge_color='k', ax=ax, node_size=50, edgecolors ='black')

    # draw lines for alignments
    for node in g1.nodes():
        if node in pos_g1 and node in pos_g2:  # Ensure node exists in both layouts
            start_pos = pos_g1[node]
            end_pos = pos_g2[node]
            plt.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], "k--", linewidth=0.5, zorder=-1, color='gray', alpha=0.5)

    plt.axis('equal')
    plt.axis('off')  # Optionally turn off the axis for a cleaner look
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
