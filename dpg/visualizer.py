import os
import numpy as np
from graphviz import Source
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from .utils import highlight_class_node, change_node_color, delete_folder_contents, change_edge_color
import re

def plot_dpg(plot_name, dot, df, df_edge, df_dpg, save_dir="examples/", attribute=None, variant=False, communities=False, class_flag=False, edge_attribute=None):
    """
    Plots a Decision Predicate Graph (DPG) with various customization options.

    Args:
    plot_name: The name of the plot.
    dot: A Graphviz Digraph object representing the DPG.
    df: A pandas DataFrame containing node metrics.
    df_dpg: A pandas DataFrame containing DPG metrics.
    save_dir: Directory to save the plot image. Default is "examples/".
    attribute: A specific node attribute to visualize. Default is None.
    communities: Boolean indicating whether to visualize communities. Default is False.
    class_flag: Boolean indicating whether to highlight class nodes. Default is False.

    Returns:
    None
    """

    # Basic color scheme if no attribute or communities are specified
    if attribute is None and not communities and edge_attribute is None:

        for index, row in df.iterrows():
            if 'Class' in row['Label']:
                change_node_color(dot, row['Node'], "#{:02x}{:02x}{:02x}".format(157, 195, 230))  # Light blue for class nodes
            else:
                change_node_color(dot, row['Node'], "#{:02x}{:02x}{:02x}".format(222, 235, 247))  # Light grey for other nodes


    # Color nodes based on a specific attribute
    elif attribute is not None and not variant and not communities and edge_attribute is None:

        colormap = cm.Blues  # Choose a colormap
        norm = None

        # Highlight class nodes if class_flag is True
        if class_flag:
            for index, row in df.iterrows():
                if 'Class' in row['Label']:
                    change_node_color(dot, row['Node'], '#ffc000')  # Yellow for class nodes
            df = df[~df.Label.str.contains('Class')].reset_index(drop=True)  # Exclude class nodes from further processing
        
        # Normalize the attribute values if norm_flag is True
        max_score = df[attribute].max()
        norm = mcolors.Normalize(0, max_score)
        colors = colormap(norm(df[attribute]))  # Assign colors based on normalized scores
        
        for index, row in df.iterrows():
            color = "#{:02x}{:02x}{:02x}".format(int(colors[index][0]*255), int(colors[index][1]*255), int(colors[index][2]*255))
            change_node_color(dot, row['Node'], color)
        
        plot_name = plot_name + f"_{attribute}".replace(" ","")


    elif attribute is not None and variant and not communities and edge_attribute is None:

        # colors = [(0, "#742881"), (0.5, "#f9f9f9"), (1, "#1b7939")]  # green-purple
        colors = [(0, "#b31529"), (0.5, "#f9f9f9"), (1, "#1065ab")]  # blu-red
        colormap = LinearSegmentedColormap.from_list("my_cmap", colors)

        norm = plt.Normalize(vmin=-1, vmax=1)

        # Highlight class nodes if class_flag is True
        if class_flag:
            for index, row in df.iterrows():
                if 'Class -1' in row['Label']:
                    change_node_color(dot, row['Node'], '#b31529')
                elif 'Class 1' in row['Label']:
                    change_node_color(dot, row['Node'], '#1065ab')
            df = df[~df.Label.str.contains('Class')].reset_index(drop=True)  # Exclude class nodes from further processing
        
        # Normalize the attribute values if norm_flag is True
               
        color_values = norm(df[attribute])  # Convert attribute values directly to the colormap range
        
        colors = colormap(color_values)  # Get corresponding RGBA colors

        for index, row in df.iterrows():
            color = "#{:02x}{:02x}{:02x}".format(
                int(colors[index][0]*255), 
                int(colors[index][1]*255), 
                int(colors[index][2]*255)
            )
            change_node_color(dot, row['Node'], color)
        
        plot_name = plot_name + f"_{attribute}".replace(" ","")
    

    # Color nodes based on community detection
    elif communities and attribute is None and edge_attribute is None:

        colormap = cm.YlOrRd  # Choose a colormap
        
        # Highlight class nodes if class_flag is True
        if class_flag:
            for index, row in df.iterrows():
                if 'Class' in row['Label']:
                    change_node_color(dot, row['Node'], '#ffc000')  # Yellow for class nodes
            df = df[~df.Label.str.contains('Class')].reset_index(drop=True)  # Exclude class nodes from further processing

        # Map labels to community indices
        label_to_community = {label: idx for idx, s in enumerate(df_dpg['Communities']) for label in s}
        df['Community'] = df['Label'].map(label_to_community)
        
        max_score = df['Community'].max()
        norm = mcolors.Normalize(0, max_score)  # Normalize the community indices        
        colors = colormap(norm(df['Community']))  # Assign colors based on normalized community indices

        for index, row in df.iterrows():
            color = "#{:02x}{:02x}{:02x}".format(int(colors[index][0]*255), int(colors[index][1]*255), int(colors[index][2]*255))
            change_node_color(dot, row['Node'], color)

        plot_name = plot_name + "_communities"
    

    # Color edges based on edge attributes
    elif edge_attribute is not None and attribute is None:

        colormap = cm.Reds  # Choose a colormap
        max_edge_value = df_edge[edge_attribute].max()
        norm = mcolors.Normalize(vmin=0, vmax=max_edge_value)  # Normalize edge attribute values

        for index, row in df_edge.iterrows():
            edge_value = row[edge_attribute]
            color = colormap(norm(edge_value))  # Calculate color based on normalized values
            color_hex = "#{:02x}{:02x}{:02x}".format(int(color[0]*255), int(color[1]*255), int(color[2]*255))
            
            # Change edge color in the Graphviz graph
            change_edge_color(dot, row['Source_id'], row['Target_id'], new_color=color_hex)  # Corrected argument name
            
        for index, row in df.iterrows():
            if 'Class -1' in row['Label']:
                change_node_color(dot, row['Node'], "#{:02x}{:02x}{:02x}".format(255, 153, 153))  # Light red for class nodes
            else:
                change_node_color(dot, row['Node'], "#{:02x}{:02x}{:02x}".format(255, 255, 255))  # White for other nodes
    
    else:
        raise AttributeError("The plot can show the basic plot, communities or a specific node-metric")







    











    # Highlight class nodes
    highlight_class_node(dot)

    def to_sci_notation(match):
        num = float(match.group(1))
        return f'label="{num:.2e}"'
    
    pattern = r'label=([0-9]+\.?[0-9]*)'

    for i in range(len(dot.body)):
        dot.body[i] = re.sub(pattern, to_sci_notation, dot.body[i])
            
    # Render the graph and display it
    dot.render("temp/"+plot_name, format="pdf")
    graph = Source(dot.source, format="png")
    graph.render("temp/" + plot_name + "_temp", view=False)

    # Open and display the rendered image
    img = Image.open("temp/" + plot_name + "_temp.png")
    plt.figure(figsize=(16, 8))
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.title(plot_name)
    plt.imshow(img)
    
    # Add a color bar if an attribute or edge attribute is specified
    if attribute is not None or edge_attribute is not None:
        cax = plt.axes([0.11, 0.07, 0.8, 0.025])  # Define color bar position
        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap), cax=cax, orientation='horizontal')
        if edge_attribute is not None:
            cbar.set_label(edge_attribute)
        else:
            cbar.set_label(attribute)

    # Save the plot to the specified directory
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, plot_name + ".png"))
    plt.show()
    
    # Close the current figure to prevent overlapping with future plots
    plt.close()

    # Clean up temporary files
    # delete_folder_contents("temp")

