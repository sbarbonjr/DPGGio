import os
import shutil



def highlight_class_node(dot):
    """
    Highlights nodes in the Graphviz Digraph that contain "Class" in their identifiers by changing their fill color
    and adding a rounded shape.

    Args:
    dot: A Graphviz Digraph object.

    Returns:
    dot: The modified Graphviz Digraph object with the class nodes highlighted.
    """
    # Iterate over each line in the dot body
    for i, line in enumerate(dot.body):
        # Extract the node identifier from the line
        line_id = line.split(' ')[1].replace("\t", "")
        # Check if the node identifier contains "Class"
        if "Class" in line_id:
            # Replace the current color with the new color and add rounded shape attribute
            dot.body[i] = dot.body[i].replace("filled", '"rounded, filled" shape=box ')
    
    # Return the modified Graphviz Digraph object
    return dot



def change_node_color(graph, node_id, new_color):
    """
    Changes the fill color of a specified node in the Graphviz Digraph.

    Args:
    graph: A Graphviz Digraph object.
    node_id: The identifier of the node whose color is to be changed.
    new_color: The new color to be applied to the node.

    Returns:
    None
    """
    # Look for the existing node in the graph body
    for i, line in enumerate(graph.body):
        if f'{node_id} [' in line:  # Find the line that defines the node
            # Modify the existing fillcolor attribute or add it if it doesn't exist
            if 'fillcolor' in line:
                # Replace the existing fillcolor
                new_line = line.rstrip().replace(
                    'fillcolor="#ffc3c3"', f'fillcolor="{new_color}"'
                )
            else:
                # Add fillcolor if it doesn't exist
                new_line = line.rstrip().replace(
                    ']', f' fillcolor="{new_color}"]'
                )
            
            # Update the line in the graph body
            graph.body[i] = new_line
            break
    
    
    
def change_edge_color(graph, source_id, target_id, new_color):
    """
    Changes the color of a specified edge in the Graphviz Digraph.

    Args:
    graph: A Graphviz Digraph object.
    source: The source node of the edge.
    target: The target node of the edge.
    new_color: The new color to be applied to the edge.

    Returns:
    None
    """
    # Look for the existing edge in the graph body
    for i, line in enumerate(graph.body):
        if f'{source_id} -> {target_id}' in line:
            # Modify the existing edge attributes to include color
            new_line = line.rstrip().replace(']', f' color="{new_color}"]')
            graph.body[i] = new_line
            break



def delete_folder_contents(folder_path):
    """
    Deletes all contents of the specified folder.

    Args:
    folder_path: The path to the folder whose contents are to be deleted.

    Returns:
    None
    """
    # Iterate over each item in the folder
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)  # Get the full path of the item
        try:
            # Check if the item is a file or a symbolic link
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)  # Remove the file or link
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # Remove the directory and its contents
        except Exception as e:
            # Print an error message if the deletion fails
            print(f'Failed to delete {item_path}. Reason: {e}')
