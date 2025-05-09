# Finite State Machine Library

## Overview

This Python library provides a flexible and easy-to-use implementation of Finite State Machines (FSMs) with visualization capabilities.

## Features

- Create complex state machines with custom states and transitions
- Add entry and exit callbacks for states
- Visualize state machine structure
- Error handling for invalid transitions

## Installation

```bash
pip install finite-state-machine
```

## Quick Example

```python
from finite_state_machine import FiniteStateMachine, StateMachineVisualizer

# Create a state machine
fsm = FiniteStateMachine()

# Define states
fsm.add_state("idle")
fsm.add_state("processing")
fsm.add_state("completed")

# Define transitions
fsm.add_transition("idle", "processing", "start")
fsm.add_transition("processing", "completed", "finish")

# Set initial state
fsm.set_initial_state("idle")

# Trigger transitions
fsm.trigger("start")  # Goes to processing
fsm.trigger("finish")  # Goes to completed

# Visualize the state machine
StateMachineVisualizer.visualize(fsm)
```

## License

MIT License
