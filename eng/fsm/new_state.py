from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional, Callable, Union, Any, Coroutine
import asyncio
import logging

class StateType(Enum):
    INITIAL = auto()
    NORMAL = auto()
    HALT = auto()
    FINAL = auto()

@dataclass
class FSMEvent:
    name: str
    source: str
    target: str
    conditions: List[Callable] = field(default_factory=list)
    
    def can_trigger(self) -> bool:
        return all(cond() for cond in self.conditions)

@dataclass
class State:
    name: str
    type: StateType = StateType.NORMAL
    on_enter: Optional[Union[Callable, Callable[..., Coroutine]]] = None
    on_exit: Optional[Union[Callable, Callable[..., Coroutine]]] = None
    data: Dict[str, Any] = field(default_factory=dict)

class FSMException(Exception):
    pass

class FiniteStateMachine:
    def __init__(self, name: str):
        self.name = name
        self.states: Dict[str, State] = {}
        self.events: Dict[str, FSMEvent] = {}
        self.current_state: Optional[State] = None
        self.history: List[str] = []
        self.logger = logging.getLogger(f"FSM.{name}")

    def add_state(self, name: str, state_type: StateType = StateType.NORMAL, 
                 on_enter: Optional[Callable] = None, 
                 on_exit: Optional[Callable] = None) -> None:
        if name in self.states:
            raise FSMException(f"State {name} already exists")
        
        state = State(name, state_type, on_enter, on_exit)
        self.states[name] = state
        
        if state_type == StateType.INITIAL and self.current_state is None:
            self.current_state = state
            self.history.append(name)

    def add_transition(self, event_name: str, source: str, target: str, 
                      conditions: List[Callable] = None) -> None:
        if source not in self.states or target not in self.states:
            raise FSMException(f"Invalid states: {source} -> {target}")
            
        event = FSMEvent(event_name, source, target, conditions or [])
        self.events[event_name] = event

    async def trigger(self, event_name: str) -> None:
        if not self.current_state:
            raise FSMException("FSM not initialized")
            
        event = self.events.get(event_name)
        if not event:
            raise FSMException(f"Unknown event: {event_name}")
            
        if event.source != self.current_state.name:
            raise FSMException(f"Cannot trigger {event_name} from {self.current_state.name}")
            
        if not event.can_trigger():
            raise FSMException(f"Event conditions not met: {event_name}")

        # Execute exit actions
        if self.current_state.on_exit:
            if asyncio.iscoroutinefunction(self.current_state.on_exit):
                await self.current_state.on_exit()
            else:
                self.current_state.on_exit()

        # Transition
        old_state = self.current_state
        self.current_state = self.states[event.target]
        self.history.append(self.current_state.name)
        
        self.logger.info(f"Transition: {old_state.name} -> {self.current_state.name}")

        # Execute entry actions
        if self.current_state.on_enter:
            if asyncio.iscoroutinefunction(self.current_state.on_enter):
                await self.current_state.on_enter()
            else:
                self.current_state.on_enter()

    @property
    def current_state_name(self) -> Optional[str]:
        return self.current_state.name if self.current_state else None

    def get_state_history(self) -> List[str]:
        return self.history.copy()

    def can_trigger(self, event_name: str) -> bool:
        event = self.events.get(event_name)
        return (event and self.current_state and 
                event.source == self.current_state.name and 
                event.can_trigger())