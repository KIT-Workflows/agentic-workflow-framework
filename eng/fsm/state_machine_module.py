import asyncio
import typing
from dataclasses import dataclass, field
from typing import Dict, Callable, Optional, Coroutine, Union, Any
from enum import Enum, auto

class StateTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""
    pass

@dataclass
class State:
    """Represents a state in the finite state machine."""
    name: str
    on_enter: Optional[Union[Callable, Callable[[], Coroutine]]] = None
    on_exit: Optional[Union[Callable, Callable[[], Coroutine]]] = None
    data: Dict[str, Any] = field(default_factory=dict) # ADDED THIS LINE

class WorkflowEvent(Enum):
    NEXT = auto()
    RETRY = auto()
    ERROR = auto()
    COMPLETE = auto()
    JUMP = auto()  # New event type for jumps


class FiniteStateMachine:
    """
    An asynchronous-capable Finite State Machine.
    
    Key Features:
    - Supports both synchronous and asynchronous state entry/exit callbacks
    - Non-blocking state transitions
    - Flexible event handling
    """
    def __init__(self):
        self.states: Dict[str, State] = {}
        self.current_state: Optional[State] = None
        self.transitions: Dict[str, Dict[str, str]] = {}
        
        # Add support for async event queues and processing
        self._event_queue = asyncio.Queue()
        self._processing_task = None

    def add_state(self, 
                  name: str, 
                  on_enter: Optional[Union[Callable, Callable[[], Coroutine]]] = None, 
                  on_exit: Optional[Union[Callable, Callable[[], Coroutine]]] = None) -> None:
        """
        Add a new state with support for sync or async callbacks.
        
        Args:
            name: Unique state identifier
            on_enter: Callback when entering the state (sync or async)
            on_exit: Callback when exiting the state (sync or async)
        """
        if name in self.states:
            raise ValueError(f"State {name} already exists")
        
        state = State(name, on_enter, on_exit)
        self.states[name] = state

    def add_transition(self, from_state: str, to_state: str, event: str) -> None:
        """
        Define a transition between states.
        
        Args:
            from_state: Source state name
            to_state: Destination state name
            event: Trigger for the transition
        """
        if from_state not in self.states:
            raise ValueError(f"From state {from_state} does not exist")
        if to_state not in self.states:
            raise ValueError(f"To state {to_state} does not exist")
        
        if from_state not in self.transitions:
            self.transitions[from_state] = {}
        
        self.transitions[from_state][event] = to_state

    def set_initial_state(self, state_name: str) -> None:
        """
        Set the initial state for the machine.
        
        Args:
            state_name: Name of the initial state
        """
        if state_name not in self.states:
            raise ValueError(f"State {state_name} does not exist")
        
        self.current_state = self.states[state_name]

    async def _call_callback(self, callback: Optional[Union[Callable, Callable[[], Coroutine]]]):
        """
        Safely call both synchronous and asynchronous callbacks.
        
        Args:
            callback: Function to call (sync or async)
        """
        if callback is None:
            return
        
        if asyncio.iscoroutinefunction(callback):
            await callback()
        else:
            callback()

    async def jump_to_state(self, target_state: str) -> None:
        """
        Directly jump to a target state, bypassing normal transitions.
        
        Args:
            target_state: Name of the state to jump to
        """
        if target_state not in self.states:
            raise ValueError(f"Target state {target_state} does not exist")
            
        # Exit current state if it exists
        if self.current_state:
            await self._call_callback(self.current_state.on_exit)
            
        # Enter new state
        self.current_state = self.states[target_state]
        await self._call_callback(self.current_state.on_enter)

    async def trigger(self, event: str, jump_target: str = None) -> None:
        """
        Enhanced trigger method that handles both normal transitions and jumps.
        
        Args:
            event: Transition event to trigger
            jump_target: Optional target state for jump events
        """
        if event == WorkflowEvent.JUMP.name:
            if not jump_target:
                raise ValueError("Jump event requires a target state")
            await self.jump_to_state(jump_target)
            return

        # Normal transition logic
        if not self.current_state:
            raise StateTransitionError("No initial state set")
        
        if (self.current_state.name not in self.transitions or 
            event not in self.transitions[self.current_state.name]):
            raise StateTransitionError(
                f"No transition from {self.current_state.name} on event {event}"
            )
        
        old_state = self.current_state
        new_state_name = self.transitions[old_state.name][event]
        new_state = self.states[new_state_name]
        
        await self._call_callback(old_state.on_exit)
        self.current_state = new_state
        await self._call_callback(new_state.on_enter)

    async def process_events(self):
        """
        Continuously process events from the queue.
        Useful for long-running state machines with multiple events.
        """
        while True:
            event = await self._event_queue.get()
            try:
                await self.trigger(event.name)
            except StateTransitionError as e:
                print(f"Transition error: {e}")
            finally:
                self._event_queue.task_done()

    async def start_processing(self):
        """
        Start processing events in the background.
        """
        if self._processing_task is None or self._processing_task.done():
            self._processing_task = asyncio.create_task(self.process_events())

    async def stop_processing(self):
        """
        Stop processing events.
        """
        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

    async def add_event(self, event: str):
        """
        Add an event to the processing queue.
        
        Args:
            event: Event to be processed
        """
        await self._event_queue.put(event)

    @property
    def current_state_name(self) -> Optional[str]:
        """
        Get the name of the current state.
        
        Returns:
            Current state name or None
        """
        return self.current_state.name if self.current_state else None