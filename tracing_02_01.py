def tracing_if(case_id, sample, iforest, feature_names, decimal_threshold, mode_graph, mode_class): #EDITED
    '''
    This function traces the decision paths taken by each iTree in a Isolation Forest for a given sample.
    It records the path of decisions made by each tree, including the comparisons at each node and the resulting class (inlier or outlier).

    Args:
    case_id: An identifier for the sample being traced.
    sample: The input sample for which the decision paths are traced.
    iforest: The Isolation Forest containing the iTrees.
    feature_names: The names of the features used in the iTrees.
    decimal_threshold: The number of decimal places to which thresholds are rounded (default is 2)

    Returns:
    event_log: A list of the decision steps taken by each tree in the forest for the given sample.
    '''
    event_log = []
    
    def find_n_sample_post(tree, next_node):
        node = tree.tree_
        n_sample_next = node.n_node_samples[next_node]
        return n_sample_next
        

    def build_path(tree, node_index, path, depth):
         
        node = tree.tree_
        is_leaf = node.children_left[node_index] == node.children_right[node_index]
        feature_index = node.feature[node_index]
        feature_name = feature_names[feature_index]
        threshold = round(node.threshold[node_index], decimal_threshold)
        sample_val = sample[feature_index]
        
        n_samples_parent = node.n_node_samples[node_index]

        if is_leaf:
            path.append(f"Class -1")
            path.append(anomaly_score)
                                                        
        else:
            depth += 1
            next_node = node.children_left[node_index] if sample_val <= threshold else node.children_right[node_index]
            n_samples_post = find_n_sample_post(tree, next_node)
            score = math.log2(n_samples_parent / n_samples_post) - 1
            condition = f"{feature_name} <= {threshold} {score} {depth}" if sample_val <= threshold else f"{feature_name} > {threshold} {score} {depth}"
            path.append(condition)
            build_path(tree, next_node, path, depth)
                  
 
    for i, tree in enumerate(iforest.estimators_):
        sample_path = []
        anomaly_score = iforest.decision_function([sample])[0]
        
        # Filtra in base a mode_class e anomaly_score
        if (mode_class == "outliers" and anomaly_score < 0) or (mode_class != "outliers" and anomaly_score >= 0):
            build_path(tree, 0, sample_path, 0)
            
            # Se mode_graph è "last", trova l'ultimo nodo significativo
            if mode_graph == "last":
                try:
                    class_index = sample_path.index("Class -1")
                    significant_path = sample_path[max(0, class_index - 1):]  # Conserva solo l'ultimo nodo e la classificazione
                except ValueError:
                    significant_path = sample_path  # Se "Class -1" non è presente, usa tutto il percorso
            else:
                significant_path = sample_path
            
            # Crea gli eventi per il log
            tree_events = [[f"sample{case_id}_dt{i}", step] for step in significant_path]
            event_log.extend(tree_events)
            
    return event_log