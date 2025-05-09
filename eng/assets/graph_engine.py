import json, re, random
from datetime import datetime
from pathlib import Path
from typing import Callable, Tuple, List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from ..fsm.state_machine_module import FiniteStateMachine
from ..core.qr_doc_engine import xqe_kg_load
from ..interfaces.lil_wrapper import process_output_to_str

def save_results_json(project,
                     fsm_state: str,
                     workflow_index: int,
                     save_dir: str = 'results') -> Path:
    """
    Save the project results to a JSON file.
    
    Args:
        project (QroissantProject): Project containing results to save
        fsm_state (str): Current state of the workflow
        workflow_index (int): Index of the workflow
        save_dir (str): Directory to save results in (relative to current directory)
        
    Returns:
        Path: Path to the saved JSON file
    """
    from pathlib import Path
    import json
    import networkx as nx
    from networkx.readwrite import json_graph
    from datetime import datetime
    
    # Get current formula data
    try:
        formula_data = project.formulas[project.indx]
    except Exception as e:
        print(f"Error getting formula data: {e}")
        formula_data = project.formulas[-1]
    
    # Convert networkx graphs to JSON-serializable format with explicit edges parameter
    evaluated_params_g = (json_graph.node_link_data(formula_data['evaluated_parameters_g'], edges="edges") 
                         if isinstance(formula_data.get('evaluated_parameters_g'), nx.Graph) else None)
    # Collect all logs
    log_entries = project.wf_status.log_entries
    
    summary = [{
        "project_signature": str(project.config.project_signature),
        "calculation_description": formula_data.get('calculation_description'),
        "analysis_dict": formula_data.get('analysis_dict'),
        # Parameters and graphs
        "get_conditions_prompts": formula_data.get('get_conditions_prompts'),
        "condition_tables": formula_data.get('condition_tables'),
        "relevant_conditions": formula_data.get('relevant_conditions'),
        "parameter_evaluation_prompts": formula_data.get('parameter_evaluation_prompts'),
        "parameters_collection": formula_data.get('parameters_collection'),
        "evaluated_parameters": formula_data.get('evaluated_parameters'),
        "evaluated_parameters_graph": evaluated_params_g,
        "trimmed_documentation": formula_data.get('trimmed_documentation'),
        "trimmed_documentation_string": formula_data.get('trimmed_documentation_string'),
        
        # QE related data
        "qe_generation_template": formula_data.get('qe_generation_template'),
        "qe_initialization": formula_data.get('qe_initialization'),
        "generated_input": formula_data.get('generated_input'),
        
        # Logs
        "error_msg": formula_data.get('error_msg', []),
        "log_qe_gen_prompt": formula_data.get('log_qe_gen_prompt', []),
        "log_qe_input": formula_data.get('log_qe_input', []),
        "log_qe_solution": formula_data.get('log_qe_solution', []),
        "rest_formula_data": {k: str(formula_data.get(k)) for k in [
            'formula', 'k_points_2d', 'uuid', 'ase_atom_object', 'indx'
        ]
        },

        "workflow_log": [
            {
                'status': entry.get('status', '').name if hasattr(entry.get('status', ''), 'name') else str(entry.get('status', '')),
                'message': entry.get('message', ''),
                'timestamp': str(entry.get('timestamp', '')),
                'level': entry.get('level', '')
            }
            for entry in log_entries
        ],

        # workflow data
        "workflow_state": fsm_state,

    }]
    
    # Create output directory
    save_path = Path.cwd() / save_dir
    print(f"Saving results to: {save_path}")
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename using timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'qe_results_{timestamp}_{workflow_index}.json'
    
    # Save to file
    with open(save_path / filename, 'w') as f:
        json.dump(summary, f, indent=4, default=str)
    return save_path / filename

def visualize_log_timeline(logs, save_path:Path=None):
    """
    Create a timeline visualization of log entries with relative time (MM:SS)
    
    Parameters:
    logs (list): List of log dictionaries with nested message structure
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    # Create DataFrame with relevant fields
    records = []
    for log in logs:
        msg_dict = log['message']
        records.append({
            'status': msg_dict['status'].split('WorkflowStatus.')[1],
            'message': msg_dict['message'],
            'start_time': pd.to_datetime(msg_dict['start_time']),
            'end_time': pd.to_datetime(msg_dict.get('end_time', None))
        })
    
    df = pd.DataFrame(records)
    
    # Calculate relative times from the first start_time
    first_time = df['start_time'].min()
    df['start_time_rel'] = (df['start_time'] - first_time).dt.total_seconds()
    df['end_time_rel'] = (df['end_time'] - first_time).dt.total_seconds() if 'end_time' in df else None
    
    df['message'] = df['status'] + ': ' + df['message'].str[:100]
    
    plt.figure(figsize=(12, 8))
    
    status_colors = {
        'PENDING': 'orange',
        'SUCCESS': 'green',
        'RETRY': 'gray',
        'ERROR': 'red'
    }
    
    # Plot timeline
    for i, row in df.iterrows():
        color = status_colors.get(row['status'], 'red')
        
        # Plot start point
        plt.scatter(row['start_time_rel'], i, color=color, s=100, zorder=3)
        
        # Plot duration line and end point if end_time exists
        if pd.notna(row['end_time_rel']):
            plt.hlines(i, row['start_time_rel'], row['end_time_rel'],
                      colors=color, linestyles='solid', linewidth=5, zorder=2)
            plt.scatter(row['end_time_rel'], i, color=color, s=100, zorder=3)
            
            # Add circles at start and end
            circle_start = plt.Circle((row['start_time_rel'], i), 0.2, 
                                    fill=False, color=color, linewidth=2, zorder=4)
            circle_end = plt.Circle((row['end_time_rel'], i), 0.2,
                                  fill=False, color=color, linewidth=2, zorder=4)
            plt.gca().add_artist(circle_start)
            plt.gca().add_artist(circle_end)
    
    # Customize plot
    plt.title('Quantum Espresso Log Timeline', fontsize=15)
    plt.xlabel('Time (MM:SS)', fontsize=12)
    plt.ylabel('Log Entries', fontsize=12)
    
    # Create colored tick labels
    ax = plt.gca()
    ax.set_yticks(range(len(df)))
    
    # Create tick labels with colors
    plt.yticks(range(len(df)), df['message'], fontsize=8)
    
    # Format x-axis ticks as MM:SS
    def format_time(x, _):
        minutes = int(x // 60)
        seconds = int(x % 60)
        return f'{minutes:02d}:{seconds:02d}'
    
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_time))
    plt.xticks(rotation=45)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color=color, label=status, marker='o')
        for status, color in status_colors.items()
    ]
    plt.legend(handles=legend_elements)
    
    # Add grid for both axes with improved visibility
    plt.grid(True, which='both', axis='both', linestyle='--', alpha=0.7)
    
    # Adjust layout to accommodate colored labels
    plt.subplots_adjust(left=0.4)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #plt.show()


def visualize_fsm(fsm: FiniteStateMachine, save_path: Path = None):
    """
    Visualize the finite state machine using NetworkX and Matplotlib.
    Shows state transitions in a clean, simple layout.
    
    Args:
        fsm: An instance of the FiniteStateMachine class
        save_path: Optional path to save the visualization
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    if not fsm.transitions:
        print("No transitions to visualize.")
        return
    
    # Create a directed graph
    graph = nx.DiGraph()
    edges = []
    
    # Add all states as nodes
    for state_name in fsm.states:
        graph.add_node(state_name)
    
    # Add transitions, filtering out 'NEXT' events
    for from_state, events in fsm.transitions.items():
        for event, to_state in events.items():
            if event != 'RETRY':  # Skip RETRY transitions
                edges.append((from_state, to_state))
                if event != 'NEXT': # Only add non-NEXT event labels
                    graph.add_edge(from_state, to_state, label=event)
                else:
                    graph.add_edge(from_state, to_state)  # Add edge without label
    
    # Set positions for nodes
    pos = nx.kamada_kawai_layout(graph)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Draw edges
    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=edges,
        edge_color="gray",
        arrowsize=20,
        connectionstyle="arc3,rad=0.1",
        min_source_margin=25,
        min_target_margin=25
    )
    
    # Draw regular nodes
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=2500,
        node_color="lightblue",
        alpha=0.7
    )
    
    # Draw special states
    special_states = {
        'INIT': {'color': 'yellow'},
        'ERROR': {'color': 'red'},
        'FINISHED': {'color': 'green'}
    }
    
    for state, props in special_states.items():
        if state in fsm.states:
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=[state],
                node_color=props['color'],
                node_shape="o",
                node_size=2500,
                alpha=0.7
            )
    
    # Highlight current state if it exists
    if fsm.current_state_name:
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=[fsm.current_state_name],
            node_color="lightgreen",
            node_shape="o",
            node_size=2500,
            alpha=1
        )
    
    # Add labels
    nx.draw_networkx_labels(
        graph,
        pos,
        font_size=8,
        font_weight="normal"
    )
    
    edge_labels = nx.get_edge_attributes(graph, "label")
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        font_size=6,
        alpha=0.7
    )
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='gray', label='Transition'),
        plt.Line2D([0], [0], marker='o', color='w', label='Normal State',
                  markerfacecolor='lightblue', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Current State',
                  markerfacecolor='lightgreen', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Init State',
                  markerfacecolor='yellow', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Error State',
                  markerfacecolor='red', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Finished State',
                  markerfacecolor='green', markersize=10),
    ]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Finalize the plot
    plt.title("Finite State Machine Visualization", fontsize=14, pad=20)
    plt.axis("off")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()



def visualize_parameter_graph(G: nx.DiGraph, figsize: tuple[int, int] = (12, 8), dpi: int = 300, save_path: Path = None):
    """Enhanced parameter graph visualization with improved styling"""
    
    plt.figure(figsize=figsize, dpi=dpi)
    style = random.choice(plt.style.available)
    plt.style.use('seaborn-v0_8-notebook')
    
    # Color scheme by node type (namelist/card)
    colors = {
        '&CONTROL': '#90EE90',    # Light green
        '&SYSTEM': '#A5D8FF',     # Light blue
        '&ELECTRONS': '#FFB6C1',   # Light red
        '&IONS': '#FFD700',       # Gold
        '&CELL': '#DDA0DD',       # Plum
        '&FCP': '#98FB98',        # Pale green
        '&RISM': '#F0E68C',       # Khaki
        'Card': '#FFA07A',        # Light salmon
        #'default': '#D3D3D3'      # Light gray, fallback
    }
    
    # Enhanced layout with more spacing
    pos = nx.spring_layout(G, k=2.5, iterations=100)
    
    # Calculate plot bounds with padding
    padding = 0.1
    xs = [coord[0] for coord in pos.values()]
    ys = [coord[1] for coord in pos.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_range = x_max - x_min
    y_range = y_max - y_min
    plt.xlim(x_min - padding * x_range, x_max + padding * x_range)
    plt.ylim(y_min - padding * y_range, y_max + padding * y_range)
    
    # Draw edges with curved arrows
    nx.draw_networkx_edges(G, pos,
                          edge_color='#2B4C7E',
                          width=1.5,
                          arrows=True,
                          arrowstyle='->',
                          arrowsize=20,
                          connectionstyle="arc3,rad=0.2",
                          min_source_margin=40,
                          min_target_margin=40)
    
    # Draw nodes by type
    for node in G.nodes():
        node_data = G.nodes[node]
        namelist = node_data.get('Card_Name') or node_data.get('Namelist')
        color = colors.get(namelist, colors['Card'])
        
        # Draw main node
        ns = 5000
        nx.draw_networkx_nodes(G, pos,
                             nodelist=[node],
                             node_color=color,
                             node_size=ns,
                             edgecolors='#1A365D',
                             linewidths=2)    
    # Draw node labels with namelist info
    labels = {
        node: f"{node}"#\n({G.nodes[node].get('Namelist', 'Card')})"
        for node in G.nodes()
    }
    nx.draw_networkx_labels(G, pos,
                           labels=labels,
                           font_size=10,
                           font_weight='bold')
    
    # Draw relationship labels with enhanced styling
    edge_labels = {
        (source, target): G.edges[source, target]['condition']
        for source, target in G.edges()
    }
    nx.draw_networkx_edge_labels(G, pos,
                                edge_labels=edge_labels,
                                font_size=9,
                                font_weight='normal',
                                bbox=dict(
                                    facecolor='white',
                                    edgecolor='none',
                                    alpha=0.7,
                                    pad=0.3,  
                                  
                                ))
    
    plt.title("Parameter Relationships Graph",
             fontsize=16,
             fontweight='bold',
             pad=20)
    
    # Add legend for namelists/cards
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor=color, markersize=15,
                                 label=name)
                      for name, color in colors.items()]
    plt.legend(handles=legend_elements,
              loc='center left',
              bbox_to_anchor=(1, 0.5))
    
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def g_maker(param: list[str], main_data: list[dict] = xqe_kg_load, verbose: bool = False,) -> nx.DiGraph:
    """
    Creates a directed graph representing parameter relationships.
    
    Args:
        param: List of parameters to analyze
        main_data: Collection of parameter documentation data
        verbose: Whether to print warning messages, default False
        
    Returns:
        NetworkX DiGraph
    """
    all_valid_nodes = set()
    for item in main_data:
        node_name = item.get('Parameter_Name') or item.get('Card_Name')
        if node_name:
            all_valid_nodes.add(node_name)

    # Add special system cards/namelists that might not be in main_data
    system_elements = {
        '&CONTROL', '&SYSTEM', '&ELECTRONS', '&IONS', '&CELL', 
        '&FCP', '&RISM', 'ATOMIC_SPECIES', 'ATOMIC_POSITIONS', 
        'K_POINTS', 'CELL_PARAMETERS', 'OCCUPATIONS', 'CONSTRAINTS', 
        'ATOMIC_FORCES', 'ATOMIC_VELOCITIES', 'ADDITIONAL_K_POINTS', 
        'SOLVENTS', 'HUBBARD'
    }
    all_valid_nodes.update(system_elements)

    connection_dict = 'Relationships_Conditions_to_Other_Parameters_Cards'

    # 1. Create initial dictionary list from input parameters
    initial_dict_list = []
    for item in main_data:
        if item.get('Parameter_Name', '') in param or item.get('Card_Name', '') in param:
            initial_dict_list.append(item)

    # 2. Create a new NetworkX graph
    G = nx.DiGraph()  # Changed to DiGraph to better represent relationship direction

    # Add nodes first
    for item in initial_dict_list:
        node_ = item.get('Parameter_Name') or item.get('Card_Name')
        node_attrs = {k: v for k, v in item.items() if k != connection_dict}
        G.add_node(node_, **node_attrs)

    # Improved edge creation with source information
    for item in initial_dict_list:
        source_node = item.get('Parameter_Name') or item.get('Card_Name')
        if source_node is None:
            continue

        relationships = item.get(connection_dict, {})
        if not relationships:
            continue

        # Get source node's namelist/card type for additional context
        source_type = item.get('Namelist') or item.get('Card_Name')

        for target_node, condition in relationships.items():
            if isinstance(condition, str):
                condition_txt = condition[:30]
            elif condition is None:
                condition_txt = ''
            else:
                condition_txt = process_output_to_str(condition)[:30]
            
            if target_node in all_valid_nodes:
                edge_attrs = {
                    'condition': condition_txt,
                    'source_node': source_node,
                    'source_type': source_type,
                    'relationship_defined_by': source_node
                }
                G.add_edge(source_node, target_node, **edge_attrs)
            elif verbose:
                print(f"Warning: Skipping invalid connection from {source_node} to {target_node}")

    # Create updated dictionary list based on graph nodes
    updated_dict_list = []
    for item in main_data:
        if item.get('Parameter_Name', '') in G.nodes() or item.get('Card_Name', '') in G.nodes():
            updated_dict_list.append(item)

    # Update node attributes with complete information
    for item in updated_dict_list:
        node_ = item.get('Parameter_Name') or item.get('Card_Name')
        node_attrs = {k: v for k, v in item.items() if k != connection_dict}
        nx.set_node_attributes(G, {node_: node_attrs})
        
    return G


def cond_graph(conditions, graph_data = xqe_kg_load ):
    main_key_str = 'Parameter_Name'
    main_key_str2 = 'Card_Name'
    connection_dict = 'Relationships_Conditions_to_Other_Parameters_Cards'
    node_attrs_key = 'Possible_Usage_Conditions'

    # Create a networkx graph
    G_cond = nx.Graph()
    
    def operation(node_attrs, node_attrs_key, b_set):
        cond_1 = node_attrs['Required/Optional'] == 'required'
        a_set = set(node_attrs[node_attrs_key])
        cond_2 = a_set.intersection(b_set).__len__() > 0
        condition =  cond_1 or cond_2
        return condition 

    # Add nodes and edges
    for item in graph_data:
        node_ = item.get(main_key_str) or item.get(main_key_str2)
        if node_ is None:
            continue    
        node_attrs = {k: v for k, v in item.items() if k != connection_dict}
        # compare list and set node_attrs['Possible_Usage_Conditions'] not in CONDITIONS:
        # convert to set, then perform set operation

        if not operation(node_attrs, node_attrs_key, conditions):
            continue
        G_cond.add_node(node_, **node_attrs)

    # add edges
    for item in graph_data:
        node_ = item.get(main_key_str) or item.get(main_key_str2)
        if node_ is None:
            continue

        node_attrs = {k: v for k, v in item.items() if k != connection_dict}
        if not operation(node_attrs, node_attrs_key, conditions):
            continue
        
        for k, v in item.get(connection_dict, {}).items():
            if (k.lower() in [i.lower() for i in list(G_cond.nodes())]) or (k in ['&CONTROL', '&SYSTEM', '&ELECTRONS', '&IONS', '&CELL', '&FCP', '&RISM', 'ATOMIC_SPECIES', 'ATOMIC_POSITIONS', 'K_POINTS', 'CELL_PARAMETERS', 'OCCUPATIONS', 'CONSTRAINTS', 'ATOMIC_FORCES', 'ATOMIC_VELOCITIES', 'ADDITIONAL_K_POINTS', 'SOLVENTS', 'HUBBARD']):
                G_cond.add_edge(node_, k, condition=v)
            
            
    return G_cond
