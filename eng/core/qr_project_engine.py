
from concurrent.futures import TimeoutError
from dataclasses import dataclass, field, asdict
import dataclasses
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Literal, Optional
from IPython.display import display, Markdown
import hashlib
import pandas as pd
import re, ast
from pathlib import Path
from ..interfaces.qr_input import HelperRetrieverMc2d, HelperRetrieverMc3d
from ..assets.text_prompt import qe_input_prompt_, qe_param_prompt_, markdown_table
from ..core.project_config import ProjectConfig
from ..interfaces.lil_wrapper import process_output_to_str
from ..core.qr_doc_engine import QrDocInstance
from ..interfaces.llms_engine import process_in_batches, a_quick_chat
from ..core.tag_finder import TagFinder, TagFinderResult
from ..core.qr_engine import llm_in_and_out, run_, QeRunError, qe_crash_handle, QeRunErrorSolution
from ..assets.graph_engine import g_maker
from ..assets.text_prompt import qe_param_prompt_

class WorkflowStatus(Enum):
    """
    Status states for a Quantum Espresso workflow.
    
    - PENDING: Workflow is initialized but not yet running
    - RUNNING: Workflow is currently in progress
    - SUCCESS: Workflow completed successfully
    - ERROR: Workflow encountered an error
    - RETRY: Workflow is being retried
    - MAX_RETRIES_EXCEEDED: Workflow failed after maximum retries
    """
    PENDING = auto()
    RUNNING = auto()
    SUCCESS = auto()
    ERROR = auto()
    RETRY = auto()
    SWITCH = auto()
    MAX_RETRIES_EXCEEDED = auto()
    UNEXPECTED_ERROR = auto()

@dataclass
class WorkflowResult:
    """
    Result of a Quantum Espresso workflow attempt.
    
    Attributes:
        status: Current status of the calculation
        error: Exception if an error occurred
        retries_used: Number of retry attempts used
        final_run_data: Results data if successful
        timestamp: When this result was created
        log_entries: Collection of log messages
    """
    status: WorkflowStatus = WorkflowStatus.PENDING
    error: Optional[Exception] = None
    retries_used: int = 0
    final_run_data: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    log_entries: List[Dict[str, Any]] = field(default_factory=list)
    log_entries_path: Path = field(default=Path('log_entries.json'))

    def add_log_entry(self, message: str, level: str = 'INFO') -> None:
        """Add a timestamped log entry."""
        self.log_entries.append({
            'timestamp': datetime.now(),
            'level': level,
            'message': message
        })

@dataclass
class CompoundData:
    """Data class to store compound information"""
    name: str
    formula: str
    calculation_description: str
    atoms_obj: Optional[Any] = None
    k_points_2d: Optional[List[int]] = None
    atomic_positions: Optional[str] = None
    atomic_species: Optional[str] = None
    cell_parameters: Optional[str] = None
    set_parameters: Optional[list[str]] = None
    qe_initialization: Optional[str] = None
    uuid: Optional[str] = None
    atoms: Optional[Any] = None
    extras: str = ''


@dataclass
class ProcessingResult:
    """Container for processing results and logs"""
    evaluated_parameters: List[str]
    success: bool
    error: Optional[str] = None


class QroissantProject:
    """Quantum Espresso Project Made with Qroissant."""
    
    def __init__(self, config: ProjectConfig):
        """Create a new Quantum Espresso Project."""
        self.config = config
        self.name = config.name
        self.participants = config.participants
        self.metadata = config.metadata
        self.main_dir = config.main_dir
        self.output_dir = config.output_dir
        self.pseudopotentials = config.pseudopotentials
        self.project_signature = config.project_signature
        self.formulas = config.formulas
        self.main_documentation = config.main_documentation
        
        self._create_dirs()
        self._create_id()
        
        self.num_retries = config.num_retries_permodel
        self.async_batch_size = config.async_batch_size
        self.sleep_time = config.sleep_time
        self.input_generation_model_hierarchy = config.gen_model_hierarchy
        self.model_config = config.model_config
        self.num_change = len(config.gen_model_hierarchy) - 1
        self.lm_api = config.lm_api
        self.qe_settings = config.qe_settings
        self.MAXDocToken = config.MAXDocToken
        self.wf_status = WorkflowResult()
        self.qr_doc_instance = QrDocInstance(lm_api=self.lm_api, 
                                             kw_model_name=config.kw_model_name,
                                             chat_model_name=config.chat_model_name,
                                             MAXDocToken=self.MAXDocToken)

    def _create_dirs(self):
        self.main_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pseudopotentials.mkdir(parents=True, exist_ok=True)

    def _create_id(self):
        self.id =  hashlib.md5(self.project_signature.encode()).hexdigest()

    def __repr__(self) -> str:
        """Return a string representation of the project."""
        return (
            f"QroissantProject("
            f"name='{self.name}', "
            f"participants={self.participants}, "
            f"main_dir='{self.main_dir}', "
            f"id='{self.id}'"
            f")"
        )

    def __str__(self) -> str:
        """Return a detailed string description of the project."""
        return "\n".join([
            f"Quantum Espresso Project: {self.name}",
            f"Participants: {', '.join(self.participants)}",
            f"Metadata: {process_output_to_str(self.metadata)}",
            f"Main Directory: {self.main_dir}",
            f"Output Directory: {self.output_dir}",
            f"Pseudopotentials Directory: {self.pseudopotentials}",
            f"Project Signature: {self.project_signature}",
            f"Project ID: {self.id}"
        ])

    def __eq__(self, other):
        if isinstance(other, QroissantProject):
            return self.id == other.id
        return False
    
    def _update_status(self, 
                       status: WorkflowStatus,
                       message: str, 
                       error: Exception = None) -> None:
        """
        Updates either generation or calculation status and adds a log entry with timing information.
        
        Args:
            status: Status string ('PENDING', 'SUCCESS', 'ERROR', 'RETRY', 'MAX_RETRIES_EXCEEDED')
            message: Base message for the status update
            error: Optional exception if status is ERROR
        """
        

        # Get current time for all cases
        current_time = datetime.now()
        
        # Initialize start_time if this is a new operation
        if status == WorkflowStatus.PENDING:
            self.wf_status.status = status
            self._start_time = current_time  # Store start time for later use
            self.wf_status.add_log_entry({
                'status': status,
                'message': message,
                'start_time': current_time.strftime("%Y-%m-%d %H:%M:%S")
            })
            return

        # For all other statuses, we need both start and end times
        if not hasattr(self, '_start_time'):
            raise ValueError("No start time found. Must call with PENDING status first.")

        # Prepare the base log entry
        log_entry = {
            'status': status,
            'start_time': self._start_time.strftime("%Y-%m-%d %H:%M:%S"),
            'end_time': current_time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Handle different status types
        if status == WorkflowStatus.SUCCESS:
            self.wf_status.status = WorkflowStatus.SUCCESS
            log_entry['message'] = f"{message}"

        elif status == WorkflowStatus.ERROR:
            self.wf_status.status = WorkflowStatus.ERROR
            error_msg = str(error) if error else ""
            log_entry['message'] = f"{message}: {error_msg}"

        elif status == WorkflowStatus.RETRY:
            self.wf_status.status = WorkflowStatus.RETRY
            log_entry['start_time'] = current_time.strftime("%Y-%m-%d %H:%M:%S")
            log_entry['message'] = f"{message}"
            # remove end_time
            log_entry.pop('end_time', None)

        elif status == WorkflowStatus.SWITCH:
            self.wf_status.status = WorkflowStatus.SWITCH
            log_entry['start_time'] = current_time.strftime("%Y-%m-%d %H:%M:%S")
            log_entry['message'] = f"{message}"
            log_entry.pop('end_time', None)

        elif status == WorkflowStatus.MAX_RETRIES_EXCEEDED:
            self.wf_status.status = WorkflowStatus.MAX_RETRIES_EXCEEDED
            log_entry['message'] = f"{message}"

        else:
            raise ValueError(f"Invalid status: {status}")

        # Add the log entry
        self.wf_status.add_log_entry(log_entry)
        

    
    def new_calculation(self,
                    name: str,
                    formula: str,
                    calculation_description: str,
                    helper_retriever: Optional[Literal['mc3d', 'mc2d']] = None,
                    **structure_params) -> None:
        """Create a new calculation with the given parameters.
        
        Args:
            name: Name of the calculation
            formula: Chemical formula
            calculation_description: Description of the calculation
            helper_retriever: Type of structure helper ('mc3d' or 'mc2d')
            **structure_params: Additional structure parameters (K_POINTS, ATOMIC_POSITIONS, etc.)
        """
        try:
            
            compound_data = CompoundData(
                name=name,
                formula=formula,
                calculation_description=calculation_description,
                **structure_params
            )
            self.compound_data = compound_data
            if not helper_retriever:
                self.formulas.append(dataclasses.asdict(compound_data))
                self.formulas[-1]['SET_PARAMS'] = set(compound_data.set_parameters)
                
                self._update_status(WorkflowStatus.SUCCESS, 
                                    'Compound information collection')
                return

            # Handle different structure types
            if helper_retriever == 'mc3d':
                self._handle_mc3d_structure(compound_data)
                
            elif helper_retriever == 'mc2d':
                self._handle_mc2d_structure(compound_data)
            

        except Exception as e:
            self._update_status(WorkflowStatus.UNEXPECTED_ERROR, 
                                'Unexpected error in Compound information collection', 
                                error=e)
            raise e
        
    def _handle_mc3d_structure(self, compound_data: CompoundData) -> None:
        """Handle 3D structure generation"""
        try:
            self._update_status(WorkflowStatus.PENDING, 
                                '3D structure generation')
            #find_out = find_compound(compound_data.formula, STOICHIOMETRIC=True)
            helper = HelperRetrieverMc3d.get_qe_input_mc3d({'main':compound_data.formula},
                                       pp_dir=self.pseudopotentials)
            # Store compound collection data
            self.collected_compounds = self._create_compounds_dataframe(helper, '3D Compounds Data')
            self._display_compounds_table()

            self.helper = helper
            #- atoms (ase.Atoms): An ASE Atoms object representing the structure. 
            # - symmetry.number (int): The space group number.
            # - sg_dict.get(str(symmetry.number)) (str): The space group symbol. 
            # - qe_inp_mod (str): The modified Quantum Espresso input string with corrected pseudopotential names. 
            # - carde_st (str): A string containing the processed card information. 
            # - uuid (str): A string representing the uuid of the structure.

            for elm in helper:
                atoms_obj = elm[0]
                formula = str(elm[0].symbols)
                sg_num = elm[1]
                sg_sym = elm[2]
                qe_init = elm[4]
                uuid = elm[5]

                self.formulas.append({
                    'formula': formula,
                    'calculation_description': compound_data.calculation_description,
                    'ase_atom_object': atoms_obj,
                    'qe_initialization': qe_init,
                    'space_group_number': sg_num,
                    'space_group_symbol': sg_sym,
                    'uuid': uuid,
                    'extras': '',
                    'SET_PARAMS' : {
                            'ecutwfc', 'ecutrho', 'nat', 'ibrav',
                            'ATOMIC_POSITIONS', 'ATOMIC_SPECIES',
                            'K_POINTS', 'CELL_PARAMETERS'
                    }
                })

            self._update_status(WorkflowStatus.SUCCESS, 
                                '3D structure generation')
            
        except Exception as e:
            self._update_status(WorkflowStatus.ERROR, 
                                '3D structure generation', 
                                error=e)

    def _handle_mc2d_structure(self, compound_data: CompoundData) -> None:
        """Handle MC2D structure generation"""
        try:
            self._update_status(WorkflowStatus.PENDING, 
                                '2D structure generation')
            
            qe_initialization, uuid, atoms = HelperRetrieverMc2d.get_qe_input_mc2d(
                compound_data.formula, 
                compound_data.k_points_2d, 
                self.pseudopotentials
            )
            compound_data.qe_initialization = qe_initialization
            compound_data.uuid = uuid
            compound_data.ase_atom_object = atoms
            print(qe_initialization)
            
            self.formulas.append(dataclasses.asdict(compound_data))
            self.formulas[-1]['SET_PARAMS'] = {
                            'ecutwfc', 'ecutrho', 'nat', 'ibrav',
                            'ATOMIC_POSITIONS', 'ATOMIC_SPECIES',
                            'K_POINTS', 'CELL_PARAMETERS'
                    }
            
            data = {
                'formula': [compound_data.formula],
                'k_points_2d': [compound_data.k_points_2d or [7, 7, 2, 0, 0, 0]],
                'uuid': [compound_data.uuid]
            }

            df = pd.DataFrame(data)
            df.index.name = 'index'
            df.title = '2D Compounds Data'  # Adding a title to the DataFrame
            self.collected_compounds = df
            self._display_compounds_table()
            
            self._update_status(WorkflowStatus.SUCCESS, 
                                '2D structure generation')
            
        except Exception as e:
            self._update_status(WorkflowStatus.ERROR, 
                                '2D structure generation', 
                                error=e)

    @staticmethod
    def _create_compounds_dataframe(helper: List[Any], title: str) -> pd.DataFrame:
        """Create a DataFrame from compound data"""
        data = {
            i: {
                'formula': str(item[0].symbols),
                'sg': item[1],
                'sg2': item[2],
                'uuid': item[5]
            }
            for i, item in enumerate(helper)
        }
        df = pd.DataFrame(data).T
        df.index.name = 'index'
        df.title = title
        return df

    def _display_compounds_table(self) -> None:
        """Display compounds table in markdown format"""

        display(Markdown(markdown_table.format(
            text=self.collected_compounds.title + '\n' + self.collected_compounds.to_markdown()
        )))

    async def initialize_document_collection(self, indx: int) -> None:
        """
        Initialize and process quantum espresso documentation for a specific formula index.
        
        Args:
            indx: Index of the formula to process
            
        Raises:
            IndexError: If indx is out of range
            ValueError: If calculation_description is missing
        """

        self.indx = indx
        
        # Start status tracking
        self._update_status(WorkflowStatus.PENDING, 
                            'Starting Quantum Espresso documentation collection')
        
        try:
            # Validate input
            if not 0 <= indx < len(self.formulas):
                self._update_status(WorkflowStatus.ERROR, 
                                    'Quantum Espresso documentation collection', 
                                    error=f"Formula index {indx} is out of range")
                
                raise IndexError(f"Formula index {indx} is out of range")
                
            formula_data = self.formulas[indx]
            calculation_description = formula_data.get('calculation_description')
            if not calculation_description or calculation_description == '':
                self._update_status(WorkflowStatus.ERROR, 
                                    'Quantum Espresso documentation collection', 
                                    error="No calculation description found")
                
                raise ValueError("No calculation description found")
                
            # Initialize document processor
            print("starting keyword time")
            await self.qr_doc_instance.keyword_time(formula_data['calculation_description'])
            print("starting doc")
            await self.qr_doc_instance.doc()
            print("starting parameters list")
            # Extract parameters using regex pattern matching
            parameters_list = []
            pattern = re.compile(r'(?:Parameter_Name|Card_Name):\n(\S+)')
            
            #for doc in self.qr_doc_instance.curated_docs:
            #    match = pattern.search(doc)
            #    if match:
            #        param = match.group(1).replace('"', '')
            #        parameters_list.append(param)
            
            # get the parameters list from the curated_docs_raw, list of dicts
            for doc in self.qr_doc_instance.curated_docs_raw:

                name = doc.get('Parameter_Name') or doc.get('Card_Name')
                if name:
                    parameters_list.append(name)

            # Update formula data
            self.formulas[indx].update({
                'curated_docs': self.qr_doc_instance.curated_docs,
                'curated_docs_raw': self.qr_doc_instance.curated_docs_raw,
                'parameters_list': parameters_list,
                'doc_': self.qr_doc_instance
            })
            
            # Log success
            self._update_status(WorkflowStatus.SUCCESS, 
                                'Quantum Espresso documentation collection')
            
        except Exception as e:
            # Log error and re-raise
            self._update_status(WorkflowStatus.ERROR, 
                                'Quantum Espresso documentation collection', 
                                error=e)
            raise
        
    async def condition_extractor(self) -> bool:
        """
        Extract conditions from calculation description and update formula data.
        
        Returns:
            bool: True if extraction was successful, False otherwise
        """

        try:
            # Input validation
            if not hasattr(self, 'indx'):
                self._update_status(WorkflowStatus.ERROR, 
                                  'Quantum Espresso condition extraction', 
                                  error="No index set for condition extraction")
                raise AttributeError("No index set for condition extraction")
                
            formula_data = self.formulas[self.indx]
            if 'calculation_description' not in formula_data:
                self._update_status(WorkflowStatus.ERROR, 
                                  'Quantum Espresso condition extraction', 
                                  error="No calculation description found")
                raise KeyError("No calculation description found")
                
            # Start status tracking
            self._update_status(WorkflowStatus.PENDING, 
                              'Quantum Espresso condition extraction')
            
            # Initialize TagFinder with project configuration
            finder = TagFinder(
                model=self.model_config['condition_finder'],
                batch_size=self.async_batch_size,
                lm_api=self.lm_api,
                sleep_time=self.sleep_time
            )
            
            # Process description
            result = await finder.process_description(
                calculation_description=formula_data['calculation_description']
            )
            
            # Handle errors if any
            if result.errors:
                print(result)
                error_messages = "\n".join(f"- {error['error']}" for error in result.errors)
                self._update_status(WorkflowStatus.ERROR, 
                                  'Quantum Espresso condition extraction', 
                                  error=error_messages)
                return False
            
            # Update formula data with results
            self.formulas[self.indx].update({
                'condition_tables': result.dataframe,
                'relevant_conditions': result.relevant_conditions,
                'get_conditions_prompts': result.items
            })
            
            # Process and update item tags
            if not result.dataframe.empty:
                item_tags = dict(zip(
                    result.dataframe['Item'],
                    result.dataframe['Relevant_tags']
                ))
                self.formulas[self.indx].update(item_tags)
            
            self._update_status(WorkflowStatus.SUCCESS, 
                              'Quantum Espresso condition extraction')
            return True
            
        except Exception as e:
            self._update_status(WorkflowStatus.ERROR, 
                              'Quantum Espresso condition extraction', 
                              error=e)
            return False

    def _validate_tag_finder_result(self, result: TagFinderResult) -> None:
        """Validate tag finder results and raise appropriate errors."""
        if result.errors:
            error_messages = "\n".join(f"- {error['error']}" for error in result.errors)
            self._update_status(WorkflowStatus.ERROR, 
                                'Quantum Espresso condition extraction', 
                                error=error_messages)
            raise ValueError(f"Errors during tag finding:\n{error_messages}")

    def _process_tag_finder_result(self, result: TagFinderResult) -> bool:
        """Process successful tag finder results and update formula data."""
        if result.dataframe.empty:
            self._update_status(WorkflowStatus.ERROR, 
                                'Quantum Espresso condition extraction', 
                                error="No conditions found from the given calculation description")
            raise ValueError("No conditions found from the given calculation description")
        
        # Update formula data with results
        self.formulas[self.indx].update({
            'condition_tables': result.dataframe,
            'relevant_conditions': result.relevant_conditions
        })
        
        # Process and update item tags
        item_tags = dict(zip(
            result.dataframe['Item'],
            result.dataframe['Relevant_tags']
        ))
        self.formulas[self.indx].update(item_tags)

        # Log success with metrics
        self._update_status(WorkflowStatus.SUCCESS, 
                            'Quantum Espresso condition extraction')
        
        return True

    def _handle_condition_extraction_error(self, error: Exception) -> None:
        """Handle errors during condition extraction."""
        # Log error
        self._update_status(WorkflowStatus.ERROR, 
                            'Quantum Espresso condition extraction', 
                            error=error)
        
        # Initialize empty results
        self.formulas[self.indx].update({
            'condition_tables': pd.DataFrame(),
            'relevant_conditions': []
        })
        
        # Log specific error details
        print(f"Error during condition extraction: {str(error)}")
        

    def create_parameter_graph(self) -> None:
        """
        Creates parameter graphs based on documentation and usage conditions.
        Updates formula data with full and trimmed parameter graphs.
        """
        try:
            self._update_status(WorkflowStatus.PENDING, 
                                'Parameter graph generation')
            
            formula_data = self.formulas[self.indx]
            param_list = formula_data['parameters_list']
            relevant_conditions = formula_data['relevant_conditions']
            

            # Filter parameters based on usage conditions
            #selected_params = set()
            #for item in self.main_documentation:
            #    param_name = item.get('Parameter_Name') or item.get('Card_Name')
            #    if not param_name:
            #        continue
            #        
            #    relationships = item.get('Relationships_Conditions_to_Other_Parameters_Cards') or {}
            #    
            #    # Check if parameter is in list or related
            #    if (param_name in param_list or 
            #        any(rel in param_list for rel in relationships)):
            #        
            #        # Check usage conditions
            #        usage_conditions = item.get('Possible_Usage_Conditions', [])
            #        if (isinstance(usage_conditions, list) and 
            #            any(cond in relevant_conditions for cond in usage_conditions)):
            #            selected_params.add(param_name)
            relevant_conditions_set = set(relevant_conditions)
            param_list_set = set(param_list)
            if not relevant_conditions_set or not param_list_set:
                selected_params = set()
            else:
                # Use set comprehension for cleaner code
                selected_params = {
                    item.get('Parameter_Name') or item.get('Card_Name')
                    for item in self.main_documentation
                    if (item.get('Parameter_Name') or item.get('Card_Name')) and  # Skip None values
                    (
                        (item.get('Parameter_Name') or item.get('Card_Name')) in param_list_set or
                        any(rel in param_list_set for rel in (item.get('Relationships_Conditions_to_Other_Parameters_Cards') or {}))
                    ) and
                    isinstance(item.get('Possible_Usage_Conditions', []), list) and
                    any(cond in relevant_conditions_set for cond in item.get('Possible_Usage_Conditions', []))
                }
            # Generate and store graphs
            formula_data.update({
                'trimmed_parameters': selected_params,
                'total_parameters': g_maker(param_list),
                'trimmed_G': g_maker(list(selected_params))
            })
            
            self._update_status(WorkflowStatus.SUCCESS, 
                                'Parameter graph generation')
            
        except Exception as e:
            self._update_status(WorkflowStatus.ERROR, 
                                'Parameter graph generation', 
                                error=e)
            raise


    def collect_and_process_parameters(self) -> None:
        """
        Collects and processes documentation for trimmed parameters.
        
        This method:
        1. Filters out already set parameters
        2. Collects documentation for trimmed and required parameters
        3. Generates parameter evaluation prompts
        4. Updates formula data with results
        
        Raises:
            KeyError: If required formula data is missing
            ValueError: If parameter processing fails
        """
        try:
            # Start status tracking
            self._update_status(WorkflowStatus.PENDING, 
                                'Documentation collection for trimmed parameters')
            
            # Get required data with validation
            formula_data = self.formulas[self.indx]
            trimmed_params = set(formula_data.get('trimmed_parameters', []))
            if not trimmed_params:
                self._update_status(WorkflowStatus.ERROR, 
                                    'Documentation collection for trimmed parameters', 
                                    error="No trimmed parameters found")
                raise ValueError("No trimmed parameters found")
                
            # Define constants
            ALREADY_SET_PARAMS = formula_data.get('SET_PARAMS', set())
            REQUIRED_KEYS = {
                'Parameter_Name', 'Card_Name', 'Namelist', 'Description',
                'Possible_Usage_Conditions', 'Usage_Conditions',
                'Parameter_Value_Conditions', 'Final_comments', 'Default_Values'
            }
            
            # Process documentation more efficiently
            trimmed_doc = []
            for item in self.main_documentation:
                param_name = item.get('Parameter_Name', item.get('Card_Name'))
                if param_name in ALREADY_SET_PARAMS:
                    continue
                    
                if param_name in trimmed_params or item.get('Required/Optional') == 'required':
                    trimmed_doc.append(item)
                    trimmed_params.add(param_name)
            
            # Generate documentation strings more efficiently
            doc_strings = set()  # Use set for automatic deduplication
            for item in trimmed_doc:
                # Create filtered copy with only required keys
                filtered_item = {k: v for k, v in item.items() if k in REQUIRED_KEYS}
                doc_strings.add(process_output_to_str(filtered_item))
            
            # Generate parameter evaluation prompts
            try:
                param_prompts = [
                    qe_param_prompt_.format(
                        proj=formula_data['qe_initialization'],
                        conditions='\n'.join(formula_data['relevant_conditions']),
                        parameter=doc_str
                    )
                    for doc_str in doc_strings
                ]
            except KeyError as e:
                raise ValueError(f"Missing required formula data: {e}")
            
            # Update formula data
            formula_data.update({
                'trimmed_documentation': trimmed_doc,
                'trimmed_documentation_string': list(doc_strings),
                'parameter_evaluation_prompts': param_prompts
            })
            
            self._update_status(WorkflowStatus.SUCCESS, 
                                'Documentation collection for trimmed parameters')
            
        except Exception as e:
            self._update_status(WorkflowStatus.ERROR, 
                                'Documentation collection for trimmed parameters', 
                                error=e)
            raise



    async def evaluate_parameters(self) -> ProcessingResult:
        """
        Process quantum espresso input parameters using LLM.
        
        Handles both initial processing and retry attempts for failed items.
        Updates formula data with processing results and initializes log containers.
        
        Returns:
            ProcessingResult if successful, None if processing failed
        """
        try:
            # Validate index and get formula data
            if not hasattr(self, 'indx'):
                self._update_status(WorkflowStatus.ERROR, 
                                    'Quantum Espresso input parameter evaluation', 
                                    error="No index set for parameter evaluation")
                raise AttributeError("No index set for parameter evaluation")
                
            formula_data = self.formulas[self.indx]
            failed_indices = formula_data.get('failed_indx')
            
            # Start status tracking
            self._update_status(WorkflowStatus.PENDING, 
                                'Quantum Espresso input parameter evaluation')
            
            # Process parameters
            results = await self._process_parameters(
                formula_data=formula_data,
                failed_indices=failed_indices
            )

            # Update formula data with results
            self.formulas[self.indx].update({
                'evaluated_parameters': results,
                'log_qe_solution': [],
                'log_qe_gen_prompt': [],
                'log_qe_input': [],
                'error_msg': []
            })
        
        # Log success
            self._update_status(WorkflowStatus.SUCCESS, 
                                'Quantum Espresso input parameter evaluation')
            
            return ProcessingResult(
                evaluated_parameters=results,
                success=True
            )
            
        except Exception as e:
            error_msg = f"Error during Quantum Espresso input parameter evaluation: {str(e)}"
            print(f"Error: {error_msg}")
            # Update status and log error
            self._update_status(WorkflowStatus.ERROR, 
                                'Quantum Espresso input parameter evaluation', 
                                error=error_msg)

            return ProcessingResult(
                evaluated_parameters=[],
                success=False,
                error=str(e)
            )

    async def _process_parameters(
        self,
        formula_data: Dict[str, Any],
        failed_indices: Optional[List[int]] = None
    ) -> List[str]:
        """Process parameters, handling both initial and retry attempts."""

        # Determine which items to process
        if failed_indices is None:
            items = formula_data['parameter_evaluation_prompts']
        else:
            items = [formula_data['parameter_evaluation_prompts'][i] for i in failed_indices]
        
        # Common processing configuration
        processing_config = {
            'batch_size': self.async_batch_size,
            'async_func': a_quick_chat,
            'func_args': {
                'model': self.model_config['parameter_evaluation'],
                'lm_api': self.lm_api,
                'extras': {
                    'temperature': 0.0,
                    'max_tokens': 5000
                }
            }
        }
        
        # Process items
        results = await process_in_batches(items=items, **processing_config, sleep_time=self.sleep_time)
        
        # Extract and return results
        return [result for _, result in results]

    def qe_input_generation_template(self) -> Optional[bool]:
        """
        Generate a Quantum Espresso input file template by processing evaluated parameters.
        
        This method:
        1. Processes evaluated parameters to extract parameter values
        2. Builds a template string combining parameters with initialization data
        3. Updates the formula data with processed parameters and template
        
        Returns:
            bool: True if successful, None if processing failed
        
        Updates formula fields:
            - parameters_collection: List of processed parameter dictionaries
            - qe_generation_template: Combined template string
            - failed_indx: List of indices where processing failed (if any)
        """
        indx = self.indx
        param_coll = []
        failed = []
        
        # Start status tracking
        self._update_status(WorkflowStatus.PENDING, 
                            'Quantum Espresso input generation template')
        
        try:
            # Process evaluated parameters more efficiently
            evaluated_params = self.formulas[indx]['evaluated_parameters']
            for itm_indx, itm in enumerate(evaluated_params):
                try:
                    # Extract parameter value using regex and ast
                    match = re.search(r'```python(.*?)```', itm, re.DOTALL)
                    param_str = match.group(1).split('parameter_value = ')[-1]
                    param_value = ast.literal_eval(param_str)
                    param_coll.append(param_value)
                    
                except Exception as e:
                    print(f"Error processing parameter {itm_indx}: {e}")
                    failed.append(itm_indx)
            
            # i want to sort the param_coll by the same order of appearance in the main_documentation
            # compare main_documentation namelist with param_coll namelist, get index of similar namelist
            coll_idx = []
            sorted_pairs = []
            for doc_idx, doc in enumerate(self.main_documentation):
                for param_dict in param_coll:
                    for param_name, param_value in param_dict.items():
                        if param_value is not None:
                            if doc.get('Parameter_Name', doc.get('Card_Name')) == param_name:
                                sorted_pairs.append((doc_idx, param_dict))
                                break
            
            param_coll = [pair[1] for pair in sorted_pairs]

            # Handle failed parameters
            if failed:
                self.formulas[indx]['failed_indx'] = failed
                self._update_status(WorkflowStatus.RETRY, 
                                    'Quantum Espresso input generation template')
                return None
            
                        
            tmp_results = []
            for itm in param_coll:
                tmp_results.extend(itm.keys())
            self.formulas[indx]['evaluated_parameters_g'] = g_maker(tmp_results)
            
            # Build template string more efficiently
            trimmed_docs = {
                doc.get('Parameter_Name', doc.get('Card_Name')): doc 
                for doc in self.formulas[indx]['trimmed_documentation']
            }
            
            # Process parameters and build template
            template_lines = []
            for param_dict in param_coll:
                for param_name, param_value in param_dict.items():
                    if param_value is not None:
                        doc = trimmed_docs.get(param_name)
                        if doc:
                            template_lines.append(
                                f"{doc['Namelist']} {param_name}, "
                                f"value: {param_value}, value_type: {doc['Value_Type']}"
                            )
            
            # Combine template with initialization
            template = '\n'.join(template_lines)
            combined_template = f"{template}\n{self.formulas[indx]['qe_initialization']}"
            
            # Update formula data
            self.formulas[indx].update({
                'parameters_collection': param_coll,
                'qe_generation_template': combined_template,
            })
            
            self._update_status(WorkflowStatus.SUCCESS, 
                                'Quantum Espresso input generation template')
            return True
            
        except Exception as e:
            self._update_status(WorkflowStatus.ERROR, 
                                'Quantum Espresso input generation template', 
                                error=e)
            return False


    def qe_input_generation(self) -> bool:
        """
        Generate Quantum Espresso input file using the specified model.

        Returns:
            bool: True if generation was successful, False otherwise
            
        Raises:
            ValueError: If required formula data is missing
            RuntimeError: If LLM generation fails
        """

        try:
            
            self._update_status(WorkflowStatus.PENDING, 
                                'Quantum Espresso input generation')
            # Input validation
            if not hasattr(self, 'indx'):
                self._update_status(WorkflowStatus.ERROR, 
                                    'Quantum Espresso input generation', 
                                    error="No index set for input generation")
                return False
                
            formula_data = self.formulas[self.indx]
            required_fields = {
                'extras', 'Calculation_types', 'formula',
                'qe_generation_template', 'calculation_description'
            }
            
            missing_fields = required_fields - set(formula_data.keys())
            if missing_fields:
                self._update_status(WorkflowStatus.ERROR, 
                                    'Quantum Espresso input generation', 
                                    error=f"Missing required formula data: {missing_fields}")
                return False


            # Prepare input template more efficiently
            calc_type = (
                '\n'.join(formula_data['Calculation_types']) 
                if isinstance(formula_data['Calculation_types'], list) 
                else formula_data['Calculation_types']
            )
            
            template_data = {
                'title': self.project_signature,
                'extras': formula_data['extras'],
                'calc_type': calc_type,
                'chem_formula': formula_data['formula'],
                'tot': formula_data['qe_generation_template'],
                'features': formula_data['calculation_description'],
                'pseudo_dir': self.pseudopotentials.as_posix(),
                'out_dir': self.output_dir.as_posix()
            }
            
            input_prompt = qe_input_prompt_.format(**template_data)
            model_name = self.input_generation_model_hierarchy[len(self.input_generation_model_hierarchy) - self.num_change - 1]
            # Generate QE input
            generated_input = llm_in_and_out(input_prompt, model_name=model_name, lm_api=self.lm_api)
            if generated_input is None:
                self._update_status(WorkflowStatus.ERROR, 
                                    'Quantum Espresso input generation', 
                                    error="LLM returned None for input generation")
                return False

            # Update formula data
            formula_data['log_qe_gen_prompt'].append(input_prompt)
            formula_data['log_qe_input'].append(generated_input.qe_input.qe_input_gen)
            formula_data['generated_input'] = generated_input
            

            self._update_status(WorkflowStatus.SUCCESS, 
                                'Quantum Espresso input generation')
            return True

        except Exception as e:
            error_msg = f"Error during Quantum Espresso input generation: {str(e)}"
            print(f"Error: {error_msg}")
            self._update_status(WorkflowStatus.ERROR, 
                                'Quantum Espresso input generation', 
                                error=e)
            return False


    def qe_run(self) -> WorkflowResult:
        """
        Attempt to run a Quantum Espresso calculation.
            
        Returns:
            QERunResult containing the status and details of the calculation attempt
            
        Raises:
            ValueError: If required attributes are missing
            AttributeError: If index or formula data is not set
        """

        # Initialize result object

        self._update_status(WorkflowStatus.PENDING, 
                                'Quantum Espresso is initializing')
        # Validate required attributes
        if not hasattr(self, 'indx'):
            self._update_status(WorkflowStatus.ERROR, 
                                'Quantum Espresso calculation failed', 
                                error="No index set for QE run")
            return False
            
        indx = self.indx
        formula_data = self.formulas[indx]
        
        if 'generated_input' not in formula_data:
            self._update_status(WorkflowStatus.ERROR, 
                                'Quantum Espresso calculation failed', 
                                error="No QE input found for current formula")
            return False
        
        generated_input = formula_data['generated_input']
        
        try:
            
            # add main_dir to qe_settings
            self.qe_settings['main_dir'] = self.main_dir
            self.qe_settings['output_dir'] = self.output_dir
            # Run QE calculation
            output_qe_run = run_(intake=generated_input,
                                 qe_settings=self.qe_settings)
            # Handle QE run error
            if isinstance(output_qe_run, QeRunError):
                
                self.formulas[self.indx]['output_qe_run'] = output_qe_run
                self.formulas[self.indx]['error_msg'].append(output_qe_run.error_msg.error_msg)
                self._update_status(WorkflowStatus.ERROR, 
                                    'Quantum Espresso calculation failed')
                return False
            
            # Log successful run
            self._update_status(WorkflowStatus.SUCCESS, 
                                'Quantum Espresso calculation completed successfully')
            return True
            
        except Exception as e:
            # Handle unexpected errors
            self._update_status(WorkflowStatus.UNEXPECTED_ERROR, 
                                'Exception in Quantum Espresso calculation', 
                                error=e)
            return False
        
    async def qe_crash_handle(self):
        """ 
        Handle a crash in a Quantum Espresso calculation.
        
        This method:
        1. Updates the calculation status to RETRY
        2. Logs a retry message
        3. Retries the calculation with the same model
        4. Returns the result of the retry
        """

        self._update_status(WorkflowStatus.PENDING, 
                            'Finding a solution for the QE crash')
        
        print(f"\033[1;33mRetrying, {self.num_retries} retries left...\033[0m")
        model_name = self.input_generation_model_hierarchy[len(self.input_generation_model_hierarchy) - self.num_change - 1]
        output_qe_run = self.formulas[self.indx]['output_qe_run']

        # Handle QE crash and retry
        output_qe_crash_handle = await qe_crash_handle(output_qe_run, model_name=model_name, lm_api=self.lm_api)
        if isinstance(output_qe_crash_handle, QeRunErrorSolution):
            self._update_status(WorkflowStatus.SUCCESS, 
                                'Found a solution for the QE crash')
            print(output_qe_crash_handle.solution.solution)
            self.formulas[self.indx]['log_qe_solution'].append(output_qe_crash_handle.solution.solution)
            self.formulas[self.indx]['log_qe_input'].append(output_qe_crash_handle.qe_input.qe_input_gen)
            self.formulas[self.indx]['generated_input'] = output_qe_crash_handle
            return True
        else:
            self._update_status(WorkflowStatus.ERROR, 
                                'Failed to find a solution for the QE crash')
            return None


    def switch_model_if_needed(self) -> WorkflowStatus:
        """
        Controls retry attempts and model switching logic for QE calculations.
        
        The method follows this logic:
        1. First tries with current model up to max retries
        2. If still failing, switches to next model and resets retries
        3. Continues until all models are tried or calculation succeeds
        
        Returns:
            CalculationStatus: The next status for the calculation:
                - RETRY: When should retry with current or next model
                - MAX_RETRIES_EXCEEDED: When all retries and models are exhausted
        """
        # Try with current model if retries remain
        if self.num_retries > 0:
            self.num_retries -= 1
            print(f"\033[1;33mRetrying with current model, {self.num_retries} retries remaining\033[0m")
            self._update_status(WorkflowStatus.RETRY, 
                                'Retrying with current model')
            return WorkflowStatus.RETRY

        # Switch models if possible
        if self.num_change > 0:
            self.num_change -= 1
            current_model_idx = len(self.input_generation_model_hierarchy) - self.num_change - 1
            next_model = self.input_generation_model_hierarchy[current_model_idx]
            
            print(f"\033[1;31mSwitching to model: {next_model}\033[0m")
            self.num_retries = 3  # Reset retries for new model
            self._update_status(WorkflowStatus.SWITCH,
                                'Switching to next model')
            return WorkflowStatus.RETRY
        
        # All retries and models exhausted
        print(f"\033[1;31mQE Run Failed: All models and retries exhausted\033[0m")
        self._update_status(WorkflowStatus.MAX_RETRIES_EXCEEDED, 
                            'QE Run Failed: All models and retries exhausted')
        return WorkflowStatus.MAX_RETRIES_EXCEEDED

    