import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as pp
import matplotlib.ticker as ticker
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout


def add_2d_scatter(ax, points, title=None, c=None):
    x, y = points.T
    if c is None:
        ax.scatter(x, y, s=50, alpha=0.8)
    else:
        ax.scatter(x, y, s=50, alpha=0.8, c=c)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())


def plot_2d(points, title, rgbs):
    fig, ax = pp.subplots(figsize=(8, 8), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    add_2d_scatter(ax, points, c=rgbs)
    return fig, ax


def map_it(ds_var, add_colorbar=False, center=None, **isel_kwds):
    f, ax = pp.subplots(figsize=(14, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    var = ds_var.isel(**isel_kwds)
    var.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), x='lon', y='lat', add_colorbar=add_colorbar, center=center)
    ax.coastlines()
    ax.add_feature(cfeature.LAND, color='grey')
    ax.set_ylim([-80, 80])
    title = ax.get_title().split('= ')[-1]
    ax.set_title(f'{ds_var.name}, {title}')
    return f, ax



def visualize_hierarchical_model(output_vars):
    """
    Creates a visualization of the hierarchical modeling framework using NetworkX and Graphviz.

    Parameters
    ----------
    output_vars : list of str
        Names of the output variables.
    """

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes for input features, first-level model, and second-level models
    G.add_node("Input Features", shape='box')  # Input features
    G.add_node("First-Level XGBoost", shape='ellipse')  # First-level model

    for output_var in output_vars:
        G.add_node(f"Second-Level XGBoost\n({output_var})", shape='ellipse')  # Second-level models

    # Add edges between nodes
    for node in G.nodes:
        if node.startswith("Second-Level"):
            G.add_edge("Input Features", node)
            G.add_edge("First-Level XGBoost", node)
        elif node == "First-Level XGBoost":
            G.add_edge("Input Features", node)

    # Add output variable nodes and edges
    for output_var in output_vars:
        G.add_node(output_var)
        G.add_edge(f"Second-Level XGBoost\n({output_var})", output_var)

    # Use Graphviz for layout
    pos = graphviz_layout(G, prog='dot')

    # Draw the graph
    plt.figure(figsize=(10, 6))  # Adjust figure size as needed
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, arrowsize=20)
    plt.title("Hierarchical Modeling Framework")
    plt.show()

# Example usage (assuming you have 'output_vars' defined)
visualize_hierarchical_model(output_vars)