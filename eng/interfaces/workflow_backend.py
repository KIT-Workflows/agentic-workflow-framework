from dotenv import get_key
import sys, json, ast, re
from pathlib import Path
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from datetime import datetime
from ..interfaces.llms_engine import quick_chat, a_quick_chat
from ..core.qr_doc_engine import process_output_to_str
from ..interfaces.qr_input import parse_chemical_formula
from ..assets.text_prompt import ask_ai_2_, ask_ai_distill
from ..assets.graph_engine import visualize_log_timeline, save_results_json
from ..fsm.state_machine_module import FiniteStateMachine
from ..core.qr_project_engine import QroissantProject, WorkflowStatus
from ..core.project_config import ProjectConfig, default_qe_settings
from ..fsm.state_machine_module import WorkflowEvent
import asyncio
import logging

lm_api = get_key('.env', 'LM_API')
gen_model_hierarchy=['dbrx', 'meta405o', 'referee']
model_config={'parameter_evaluation': 'mistral',
              'condition_finder': 'mistral'}
interface_agent_kwargs={'model': 'mistral', 'extras':{'max_tokens': 8000}}

logger = logging.getLogger(__name__)

async def interface_agent(user_prompt: str, 
                    lm_function: callable = a_quick_chat, 
                    base_prompt: str = ask_ai_2_, 
                    lm_func_kwargs: dict = None) -> tuple:
    
    """
    Interface with an LLM to analyze a user prompt and extract structured information.
    
    This function takes a user prompt, sends it to a language model with a base prompt template,
    and processes the response to extract both markdown content and a Python dictionary containing
    the analysis results.

    Args:
        user_prompt (str): The input prompt to be analyzed
        lm_function (callable, optional): Function to call the language model. Defaults to quick_chat.
        base_prompt (str, optional): Template prompt to format with user input. Defaults to ask_ai_2_.
        lm_func_kwargs (dict, optional): Additional kwargs for the LM function. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - str: The markdown portion of the LLM response
            - dict: The extracted analysis dictionary, or None if parsing fails
    """
    
    import re, json, ast
    lm_func_kwargs_ref = {'lm_api': lm_api, 'model': 'mistral'}

    if lm_func_kwargs is not None:
        lm_func_kwargs_ref.update(lm_func_kwargs)
    tx_ = base_prompt.format(INPUT=user_prompt)
    lm_output = await lm_function(query = tx_, **lm_func_kwargs_ref)
    # cleanup

    md_part = lm_output[:lm_output.find('```python')]
    
    find_dict = re.findall(r'```python(.*?)```', lm_output, re.DOTALL)
    if find_dict:
        py_part = find_dict[0]
        try:
            analysis_dict = json.loads(py_part)
        except json.JSONDecodeError:
            analysis_dict = ast.literal_eval(py_part.split('analysis_dict = ')[1])
    else:
        analysis_dict = None

    return md_part, analysis_dict

def prepare_for_kw_extraction(analysis_dict: dict):
    base = process_output_to_str(analysis_dict['modified_description'])
    return base

#with open('out_interface.json', 'r') as f:
#    out_interface = json.load(f)
#
#with open('ok_items.json', 'r') as f:
#    data = json.load(f)
#
#items = [out_interface[i][1]['description'] for i in range(len(out_interface))]
#tmp_ = [list(i.values())[0] for i in data]

#items_for_workflow = items + tmp_

with open('kmeans_cluster.json', 'r') as f:
    data = json.load(f)

#items_for_workflow = data['Cluster 1'] + data['Cluster 2']

current_dir = Path.cwd()
out_dir = current_dir / 'out_dir'
pseudopotentials_dir = current_dir / 'new_pp'

print(current_dir, out_dir, pseudopotentials_dir, sep='\n----------------\n')

updated_qe_settings = default_qe_settings.copy()
updated_qe_settings['n_cores'] = 8
updated_qe_settings['use_slurm'] = False # preferred 
updated_qe_settings['qe_prefix'] = 'arch -x86_64'

config = ProjectConfig(name='test',
                       participants=['test'],
                       metadata={'test': 'test'},
                       main_dir= current_dir,
                       output_dir=out_dir,
                       pseudopotentials=pseudopotentials_dir,
                       project_signature='test',
                       num_retries_permodel=3,
                       async_batch_size=10,
                       MAXDocToken=19000,
                       gen_model_hierarchy=gen_model_hierarchy,
                       model_config=model_config,
                       lm_api= lm_api,
                       qe_settings=updated_qe_settings
                       )


class WorkflowState(Enum):
    INIT = auto()
    INTERFACE = auto()
    NEW_CALCULATION = auto()
    DOC_COLLECTION = auto()
    CONDITION_EXTRACTOR = auto()
    PARAMETER_GRAPH = auto()
    COLLECT_AND_PROCESS_PARAMETERS = auto()
    EVALUATE_PARAMETERS = auto()
    QE_INPUT_GENERATION_TEMPLATE = auto()
    QE_INPUT_GENERATION = auto()
    QE_RUN = auto()
    SWITCH_MODEL_IF_NEEDED = auto()
    QE_CRASH_HANDLE = auto()
    FINISHED = auto()
    ERROR = auto()

@dataclass
class WorkflowContext:
    project: QroissantProject  # QroissantProject instance
    workflow_index: Optional[int] = None
    items: Optional[List[Dict[str, str]]] = None
    calculation_prompt: Optional[str] = None
    selected_item: Optional[str] = None
    analysis_dict: Optional[Dict[str, Any]] = None
    formula_index: Optional[int] = None
    md_part: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    jump_target: Optional[str] = None  # New field for jump targets
    lm_func_kwargs: Optional[dict] = field(default_factory=lambda: interface_agent_kwargs)
    
    def get_selected_item(self, index: int) -> str:
        """Get the selected item at the given index."""
        if not self.items:
            raise ValueError("No items available")
        if index >= len(self.items):
            raise ValueError(f"Index {index} out of range")
        return self.items[index]

class WorkflowError(Exception):
    """Base exception for workflow errors"""
    pass

class ValidationError(WorkflowError):
    """Raised when validation fails"""
    pass

class ProcessingError(WorkflowError):
    """Raised when processing fails"""
    pass

class WorkflowSteps:
    @staticmethod
    async def interface_step(context: WorkflowContext) -> WorkflowEvent:
        try:
            # Get selected item
            if not context.calculation_prompt:
                context.selected_item = context.get_selected_item(context.workflow_index)
            else:
                context.selected_item = context.calculation_prompt
            
            # Call interface agent
            context.project._update_status(WorkflowStatus.PENDING, 
                                'Interface agent')
            context.md_part, context.analysis_dict = await interface_agent(
                user_prompt=context.selected_item,
                base_prompt=ask_ai_distill,
                lm_func_kwargs=context.lm_func_kwargs
            )
            
            # Validate analysis dict
            required_keys = ['description', 'formula', 'database']#, 'analysis', 'modified_description']
            
            if context.analysis_dict is None:
                raise ValidationError("analysis_dict is None")
                
            missing_keys = [key for key in required_keys if key not in context.analysis_dict]
            if missing_keys:
                raise ValidationError(f"Missing required keys: {missing_keys}")
            
            print(json.dumps(context.analysis_dict, indent=4))

            context.project._update_status(WorkflowStatus.SUCCESS, 
                                'Interface agent')
            return WorkflowEvent.NEXT
                
        except Exception as e:
            context.error_message = str(e)
            return WorkflowEvent.ERROR if context.retry_count >= context.max_retries else WorkflowEvent.RETRY
        
    @staticmethod
    async def new_calculation_step(context: WorkflowContext) -> WorkflowEvent:
        try:

            context.project.new_calculation(
                name='calc_1',
                formula=context.analysis_dict['formula'],
                calculation_description=context.analysis_dict['description'],
                helper_retriever=context.analysis_dict['database']
            )
            
            status = context.project.wf_status.status.name
            if status == 'SUCCESS':
                return WorkflowEvent.NEXT
            else:
                context.error_message = str(context.project.wf_status.log_entries[-1])
                print(context.error_message)
                context.jump_target = WorkflowState.INTERFACE.name
                return WorkflowEvent.JUMP if context.retry_count >= context.max_retries else WorkflowEvent.ERROR
            
        except Exception as e:
            context.error_message = str(e)
            return WorkflowEvent.ERROR 

    @staticmethod
    async def doc_collection_step(context: WorkflowContext) -> WorkflowEvent:
        try:    
            context.formula_index = None
            if context.analysis_dict['database'] == 'mc3d':
                try:    
                    b = parse_chemical_formula(context.analysis_dict['formula'])
                    for formual_index, itm in enumerate(context.project.find_out[1]):
                        if parse_chemical_formula(itm) == b:
                            context.formula_index = formual_index
                            break
                
                except Exception as e:
                    context.formula_index = 0

            elif context.analysis_dict['database'] == 'mc2d':
                context.formula_index = 0

            if context.formula_index is None:
                context.formula_index = 0

            print(f"context.formula_index: {context.formula_index}")
            print('Starting document collection')
            await context.project.initialize_document_collection(indx=context.formula_index)
            print('Document collection completed')
            context.project.formulas[context.formula_index]['analysis_dict'] = context.analysis_dict
            status = context.project.wf_status.status.name
            if status == 'SUCCESS':
                return WorkflowEvent.NEXT
            else:
                context.error_message = str(context.project.wf_status.log_entries[-1])
                return WorkflowEvent.ERROR if context.retry_count >= context.max_retries else WorkflowEvent.RETRY
        except Exception as e:
            context.error_message = str(e)
            return WorkflowEvent.ERROR if context.retry_count >= context.max_retries else WorkflowEvent.RETRY
        
    @staticmethod
    async def condition_extractor_step(context: WorkflowContext) -> WorkflowEvent:
        try:
            await context.project.condition_extractor()
            status = context.project.wf_status.status.name
            if status == 'SUCCESS':
                return WorkflowEvent.NEXT
            else:
                context.error_message = str(context.project.wf_status.log_entries[-1])
                return WorkflowEvent.ERROR if context.retry_count >= context.max_retries else WorkflowEvent.RETRY
                
        except Exception as e:
            context.error_message = str(e)
            return WorkflowEvent.ERROR if context.retry_count >= context.max_retries else WorkflowEvent.RETRY
    
    @staticmethod
    async def parameter_graph_step(context: WorkflowContext) -> WorkflowEvent:
        try:
            context.project.create_parameter_graph()
            status = context.project.wf_status.status.name
            if status == 'SUCCESS':
                return WorkflowEvent.NEXT
            else:
                context.error_message = str(context.project.wf_status.log_entries[-1])
                return WorkflowEvent.ERROR if context.retry_count >= context.max_retries else WorkflowEvent.RETRY
        except Exception as e:
            context.error_message = str(e)
            return WorkflowEvent.ERROR if context.retry_count >= context.max_retries else WorkflowEvent.RETRY
        
    #collect_and_process_parameters
    @staticmethod
    async def collect_and_process_parameters_step(context: WorkflowContext) -> WorkflowEvent:
        try:
            context.project.collect_and_process_parameters()
            status = context.project.wf_status.status.name
            if status == 'SUCCESS':
                return WorkflowEvent.NEXT
            else:
                context.error_message = str(context.project.wf_status.log_entries[-1])
                return WorkflowEvent.ERROR if context.retry_count >= context.max_retries else WorkflowEvent.RETRY
        except Exception as e:
            context.error_message = str(e)
            return WorkflowEvent.ERROR if context.retry_count >= context.max_retries else WorkflowEvent.RETRY
        
    #evaluate_parameters
    @staticmethod
    async def evaluate_parameters_step(context: WorkflowContext) -> WorkflowEvent:
        try:
            await context.project.evaluate_parameters()
            status = context.project.wf_status.status.name
            if status == 'SUCCESS':
                return WorkflowEvent.NEXT
            else:
                context.error_message = str(context.project.wf_status.log_entries[-1])
                return WorkflowEvent.ERROR if context.retry_count >= context.max_retries else WorkflowEvent.RETRY
        except Exception as e:
            context.error_message = str(e)
            return WorkflowEvent.ERROR if context.retry_count >= context.max_retries else WorkflowEvent.RETRY
    #qe_input_generation_template
    @staticmethod
    async def qe_input_generation_template_step(context: WorkflowContext) -> WorkflowEvent:
        try:
            context.project.qe_input_generation_template()
            status = context.project.wf_status.status.name
            if status == 'SUCCESS':
                return WorkflowEvent.NEXT
            elif status == 'RETRY':
                context.jump_target = WorkflowState.EVALUATE_PARAMETERS.name
                return WorkflowEvent.JUMP if context.retry_count >= context.max_retries else WorkflowEvent.RETRY
            else:
                context.error_message = str(context.project.wf_status.log_entries[-1])
                return WorkflowEvent.ERROR if context.retry_count >= context.max_retries else WorkflowEvent.RETRY
        except Exception as e:
            context.error_message = str(e)
            return WorkflowEvent.ERROR if context.retry_count >= context.max_retries else WorkflowEvent.RETRY
        
    #qe_input_generation
    @staticmethod
    async def qe_input_generation_step(context: WorkflowContext) -> WorkflowEvent:
        try:
            context.project.qe_input_generation()
            status = context.project.wf_status.status.name
            if status == 'SUCCESS':
                return WorkflowEvent.NEXT
            else:
                context.error_message = str(context.project.wf_status.log_entries[-1])
                return WorkflowEvent.ERROR if context.retry_count >= context.max_retries else WorkflowEvent.RETRY
        except Exception as e:
            context.error_message = str(e)
            return WorkflowEvent.ERROR if context.retry_count >= context.max_retries else WorkflowEvent.RETRY

    #qe_run
    @staticmethod
    async def qe_run_step(context: WorkflowContext) -> WorkflowEvent:
        try:
            context.project.qe_run()
            status = context.project.wf_status.status.name
            if status == 'SUCCESS':
                return WorkflowEvent.COMPLETE
            elif status == 'ERROR':
                return WorkflowEvent.NEXT
            elif status == 'UNEXPECTED_ERROR':
                return WorkflowEvent.ERROR if context.retry_count >= context.max_retries else WorkflowEvent.RETRY
        except Exception as e:
            context.error_message = str(e)
            return WorkflowEvent.ERROR if context.retry_count >= context.max_retries else WorkflowEvent.RETRY

    #switch_model_if_needed
    @staticmethod
    async def switch_model_if_needed_step(context: WorkflowContext) -> WorkflowEvent:
        try:
            context.project.switch_model_if_needed()
            status = context.project.wf_status.status.name # one or retry, switch, max_retries_exceeded
            if status == 'RETRY':
                return WorkflowEvent.NEXT
            elif status == 'SWITCH':
                context.jump_target = WorkflowState.QE_INPUT_GENERATION_TEMPLATE.name
                return WorkflowEvent.JUMP
            elif status == 'MAX_RETRIES_EXCEEDED':
                return WorkflowEvent.ERROR
            else:
                return WorkflowEvent.ERROR if context.retry_count >= context.max_retries else WorkflowEvent.RETRY
        except Exception as e:
            context.error_message = str(e)
            return WorkflowEvent.ERROR if context.retry_count >= context.max_retries else WorkflowEvent.RETRY

    #qe_crash_handle
    @staticmethod
    async def qe_crash_handle_step(context: WorkflowContext) -> WorkflowEvent:
        try:
            await context.project.qe_crash_handle()
            status = context.project.wf_status.status.name # one of: success -> qe run, error -> halt
            if status == 'SUCCESS':
                return WorkflowEvent.NEXT
            else:
                context.error_message = str(context.project.wf_status.log_entries[-1])
                return WorkflowEvent.ERROR if context.retry_count >= context.max_retries else WorkflowEvent.RETRY
        
        except Exception as e:
            context.error_message = str(e)
            return WorkflowEvent.ERROR if context.retry_count >= context.max_retries else WorkflowEvent.RETRY

class WorkflowController:
    def __init__(self, context: WorkflowContext):
        self.fsm = FiniteStateMachine()
        self.context = context
        self._setup_fsm()
        self.is_running = False
        self._stop_requested = False
        
    def _setup_fsm(self):
        # Add states
        for state in WorkflowState:
            self.fsm.add_state(state.name)
            
        # Add transitions
        self._add_transitions()
        
        # Set initial state
        self.fsm.set_initial_state(WorkflowState.INIT.name)

    async def jump_to_state(self, target_state: str) -> None:
        """
        Jump to a specific state in the workflow.
        
        Args:
            target_state: The WorkflowState to jump to
        """
        await self.fsm.trigger(WorkflowEvent.JUMP.name, jump_target=target_state)
        self.context.jump_target = None

    def _add_transitions(self):
        # Normal flow transitions
        self.fsm.add_transition(
            WorkflowState.INIT.name,
            WorkflowState.INTERFACE.name,
            WorkflowEvent.NEXT.name
        )
        self.fsm.add_transition(
            WorkflowState.INTERFACE.name,
            WorkflowState.NEW_CALCULATION.name,
            WorkflowEvent.NEXT.name
        )
        self.fsm.add_transition(
            WorkflowState.NEW_CALCULATION.name,
            WorkflowState.DOC_COLLECTION.name,
            WorkflowEvent.NEXT.name
        )
        self.fsm.add_transition(
            WorkflowState.DOC_COLLECTION.name,
            WorkflowState.CONDITION_EXTRACTOR.name,
            WorkflowEvent.NEXT.name
        )
        self.fsm.add_transition(
            WorkflowState.CONDITION_EXTRACTOR.name,
            WorkflowState.PARAMETER_GRAPH.name,
            WorkflowEvent.NEXT.name
        )
        self.fsm.add_transition(
            WorkflowState.PARAMETER_GRAPH.name,
            WorkflowState.COLLECT_AND_PROCESS_PARAMETERS.name,
            WorkflowEvent.NEXT.name
        )
        self.fsm.add_transition(
            WorkflowState.COLLECT_AND_PROCESS_PARAMETERS.name,
            WorkflowState.EVALUATE_PARAMETERS.name,
            WorkflowEvent.NEXT.name
        )
        self.fsm.add_transition(
            WorkflowState.EVALUATE_PARAMETERS.name,
            WorkflowState.QE_INPUT_GENERATION_TEMPLATE.name,
            WorkflowEvent.NEXT.name
        )
        self.fsm.add_transition(
            WorkflowState.QE_INPUT_GENERATION_TEMPLATE.name,
            WorkflowState.QE_INPUT_GENERATION.name,
            WorkflowEvent.NEXT.name
        )
        self.fsm.add_transition(
            WorkflowState.QE_INPUT_GENERATION.name,
            WorkflowState.QE_RUN.name,
            WorkflowEvent.NEXT.name
        )
        self.fsm.add_transition(
            WorkflowState.QE_RUN.name,
            WorkflowState.SWITCH_MODEL_IF_NEEDED.name,
            WorkflowEvent.NEXT.name
        )
        self.fsm.add_transition(
            WorkflowState.QE_RUN.name,
            WorkflowState.FINISHED.name,
            WorkflowEvent.COMPLETE.name
        )
        self.fsm.add_transition(
            WorkflowState.SWITCH_MODEL_IF_NEEDED.name,
            WorkflowState.QE_CRASH_HANDLE.name,
            WorkflowEvent.NEXT.name

        )
        self.fsm.add_transition(
            WorkflowState.QE_CRASH_HANDLE.name,
            WorkflowState.QE_RUN.name,
            WorkflowEvent.NEXT.name
        )

        
        # Error and retry transitions
        for state in WorkflowState:
            if state not in [WorkflowState.INIT, WorkflowState.FINISHED, WorkflowState.ERROR]:
                self.fsm.add_transition(state.name, state.name, WorkflowEvent.RETRY.name)
                self.fsm.add_transition(state.name, WorkflowState.ERROR.name, WorkflowEvent.ERROR.name)
        
    async def stop_workflow(self):
        """Request graceful workflow stop"""
        logger.info("Stop requested for workflow")
        self._stop_requested = True
        
    async def run_workflow(self) -> None:
        try:
            self.is_running = True
            logger.info("Starting workflow processing")
            await self.fsm.start_processing()
            
            while (self.fsm.current_state_name not in 
                   [WorkflowState.FINISHED.name, WorkflowState.ERROR.name]):
                
                # Check for stop request
                if self._stop_requested:
                    logger.info("Stop request detected, shutting down gracefully")
                    await self.fsm.trigger(WorkflowEvent.ERROR.name)
                    break
                
                logger.info(f"Current state: {self.fsm.current_state_name}")
                event = await self._execute_current_state()
                
                if event == WorkflowEvent.RETRY:
                    self.context.retry_count += 1
                    logger.warning(f"Retrying... Attempt {self.context.retry_count}")
                    logger.warning(f"Error message: {self.context.error_message}")
                    
                    await asyncio.sleep(10)
                    await self.fsm.trigger(event.name)
                elif event == WorkflowEvent.ERROR:
                    logger.error(f"Error occurred: {self.context.error_message}")
                    await self.fsm.trigger(event.name)
                    break
                elif event == WorkflowEvent.JUMP:
                    if self.context.jump_target:
                        logger.info(f"Jumping to state: {self.context.jump_target}")
                        self.context.retry_count += 1
                        await self.jump_to_state(target_state=self.context.jump_target)
                    else:
                        logger.warning("Warning: Jump event received but no target state specified")
                        continue
                else:
                    self.context.retry_count = 0
                    await self.fsm.trigger(event.name)
            
            logger.info(f"Workflow finished in state: {self.fsm.current_state_name}")
            # if the current state is error, print and dump all project wf state log entries
            if self.fsm.current_state_name == WorkflowState.ERROR.name:
                logger.error(f"Project wf state log entries: {json.dumps(self.context.project.wf_status.log_entries, indent=4 , default=str)}")
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                logger.info(f"Timestamp: {timestamp}")
                log_path = self.context.project.wf_status.log_entries_path.with_stem(f"{self.context.project.wf_status.log_entries_path.stem}_{timestamp}")
                with open(log_path, 'w') as f:
                    json.dump(self.context.project.wf_status.log_entries, f, indent=4, default=str)
            await self.fsm.stop_processing()
            
        except Exception as e:
            logger.error(f"Workflow failed with error: {str(e)}")
            raise
        finally:
            self.is_running = False
            self._stop_requested = False

    async def _execute_current_state(self) -> WorkflowEvent:
        state_name = self.fsm.current_state_name
        logger.info(f"\nExecuting state: {state_name}")  # Add detailed logging
        
        state_handlers = {
            WorkflowState.INTERFACE.name: WorkflowSteps.interface_step,
            WorkflowState.NEW_CALCULATION.name: WorkflowSteps.new_calculation_step,
            WorkflowState.DOC_COLLECTION.name: WorkflowSteps.doc_collection_step,
            WorkflowState.CONDITION_EXTRACTOR.name: WorkflowSteps.condition_extractor_step,
            WorkflowState.PARAMETER_GRAPH.name: WorkflowSteps.parameter_graph_step,
            WorkflowState.COLLECT_AND_PROCESS_PARAMETERS.name: WorkflowSteps.collect_and_process_parameters_step,
            WorkflowState.EVALUATE_PARAMETERS.name: WorkflowSteps.evaluate_parameters_step,
            WorkflowState.QE_INPUT_GENERATION_TEMPLATE.name: WorkflowSteps.qe_input_generation_template_step,
            WorkflowState.QE_INPUT_GENERATION.name: WorkflowSteps.qe_input_generation_step,
            WorkflowState.QE_RUN.name: WorkflowSteps.qe_run_step,
            WorkflowState.SWITCH_MODEL_IF_NEEDED.name: WorkflowSteps.switch_model_if_needed_step,
            WorkflowState.QE_CRASH_HANDLE.name: WorkflowSteps.qe_crash_handle_step,
        }
        
        if state_name == WorkflowState.INIT.name:
            return WorkflowEvent.NEXT
            
        if state_name in state_handlers:
            result = await state_handlers[state_name](self.context)
            logger.info(f"State {state_name} completed with event: {result}")  # Add result logging
            return result
        
        return WorkflowEvent.NEXT




async def main(index: int = 0, calculation_prompt: str = None):
    # Initialize workflow data
    workflow_context = None
    save_path = None

    try:
        workflow_context = WorkflowContext(
            #items = items_for_workflow,
            calculation_prompt = calculation_prompt,
            project= QroissantProject(config),
            workflow_index=index
        )

        # Create and run workflow
        controller = WorkflowController(workflow_context)
        await controller.run_workflow()
    except Exception as e:
        print(f"Error occurred during workflow execution: {str(e)}")
        raise
    
    # save results
    try:
        if workflow_context:
            print(f"Saving results for workflow {index}")
            save_path_filename = save_results_json(project=workflow_context.project,
                                                    fsm_state=controller.fsm.current_state_name,
                                                    workflow_index=index,
                                                    save_dir='wf_api_results')
            save_path = save_path_filename.parent
            print(f"Json saved to: {save_path_filename}")
    except Exception as e:
        print(f"Error occurred during saving results: {str(e)}")
        

    # visualize results
    try:
        if save_path:
            with open(save_path_filename, 'r') as f:
                data = json.load(f)
        
        visualize_log_timeline(data[0]['workflow_log'], 
                           save_path=save_path / f'qe_results_{index}.png')
        print(f"Visualization saved to: {save_path / f'qe_results_{index}.png'}")

    except Exception as e:
        print(f"Error occurred during visualization: {str(e)}")
        

    print(f"Workflow finished in state: {controller.fsm.current_state_name}")


if __name__ == "__main__":
    import asyncio
    # input from argparse
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workflow_index', type=int, default=0)
    parser.add_argument('--calculation_prompt', type=str, default=None)
    args = parser.parse_args()
    asyncio.run(main(index=args.workflow_index, calculation_prompt=args.calculation_prompt))