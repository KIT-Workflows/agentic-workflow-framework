
class StateMachineVisualizer:
    """
    Visualize finite state machines using Graphviz.
    """
    @staticmethod
    def visualize(state_machine, filename='state_machine', format='png'):
        """
        Create a visual representation of the state machine.
        
        Args:
            state_machine: FiniteStateMachine instance
            filename: Output filename
            format: Output image format (png, svg, etc.)
        
        Returns:
            Path to the generated visualization
        """
        dot = graphviz.Digraph(comment='State Machine')
        
        # Add nodes (states)
        for state_name in state_machine.states:
            is_current = state_name == state_machine.current_state_name
            node_attrs = {
                'style': 'filled',
                'fillcolor': 'lightblue' if is_current else 'white'
            }
            dot.node(state_name, state_name, **node_attrs)
        
        # Add edges (transitions)
        for from_state, transitions in state_machine.transitions.items():
            for event, to_state in transitions.items():
                dot.edge(from_state, to_state, label=event)
        
        # Render and save
        output_path = dot.render(filename=filename, format=format, cleanup=True)
        return output_path
