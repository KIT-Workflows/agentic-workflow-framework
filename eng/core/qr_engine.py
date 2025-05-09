import re, os, subprocess, ast, logging
from typing import Any, List, Optional, Union, Dict
from pathlib import Path
from typing import Union, Optional
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from ..core.qr_doc_engine import QrDocInstance
from ..assets.text_prompt import X__m, error_keywords_prompt, QE_GEN_PATTERN
from ..interfaces.llms_engine import quick_chat

class LLMkwarg(BaseModel):
    kwargs : dict[str, Any]

class QeMatch(BaseModel):
    qe_input_gen: str

class CrashErr(BaseModel):
    error_msg: str

class DoneMsg(BaseModel):
    qe_output: str

class SOLUTION(BaseModel):
    solution: str

class ErrorKw(BaseModel):
    error_message: str 
    variables: List[str] 
    explanation: str 

class QeInputGenerate(BaseModel):
    qe_input: QeMatch = Field(..., title="Quantum Espresso input file")

class QeRunError(BaseModel):
    qe_input: QeMatch = Field(..., title="Quantum Espresso input file")
    error_msg: CrashErr = Field(..., title="Error message")

class QeRunErrorSolution(BaseModel):
    qe_input: QeMatch = Field(..., title="Quantum Espresso input file")
    error_msg: CrashErr = Field(..., title="Error message")
    solution_prompt: str = Field(..., title="Solution prompt")
    solution: SOLUTION = Field(..., title="Solution for the error message")

class QeRunDone(BaseModel):
    qe_input: QeMatch = Field(..., title="Quantum Espresso input file")
    qe_output: DoneMsg = Field(..., title="Quantum Espresso output")

def set_process_group():
    os.setpgrp()  # Create new process group

def llm_in_and_out(input_prompt: str,
                   lm_api: str,
                   model_name = 'dbrx',
                   ) -> Optional[QeInputGenerate]:
    try:
        
        llm_out = quick_chat(
            query=input_prompt,
            lm_api=lm_api,
            model= model_name,
            extras={'temperature': 0.0, 'max_tokens': 5000}
        )
            
        matches = re.findall(QE_GEN_PATTERN, llm_out, re.DOTALL)[-1]
        if not is_valid_qe_input(matches):
            logger.error("Invalid QE input generated")
            return None
            
        return QeInputGenerate(
            qe_input=QeMatch(qe_input_gen=matches)
        )
    
    except Exception as e:
        logger.error(f"Failed to process LLM output: {e}")
        return None

def qe_output_parser(output_path: str,
                    f1_pat:str = '<=!',
                    f2_pat:str = '=$') -> str:
    
    with open(output_path) as f:
        strs = f.read()
    
    pattern = rf'(?{f1_pat})(.*?)(?{f2_pat})'
    try:
        out_msg_ = re.findall(pattern, strs, re.DOTALL)[0].strip()
        return out_msg_
    
    except Exception as e:
        print("Returnig the entire output")
        return strs


def parse_crash_error(crash_file: Path) -> str:
    """Parse error message from CRASH file."""
    try:
        crash_content = crash_file.read_text()
        error_matches = re.findall(r"%+\s+(.*?)%+", crash_content, re.DOTALL)
        if error_matches:
            return error_matches[-1].strip().split('\n')[-1].strip()
    except Exception as e:
        return f"Error parsing crash file: {str(e)}"
    return "Unknown error in CRASH file"

def run_(
    intake: Union[QeInputGenerate, QeRunErrorSolution],
    qe_settings: Dict 
    ) -> Optional[Union[QeRunError, QeRunDone]]:
    """
    Run Quantum Espresso calculation with improved error handling and resource management.
    
    Args:
        intake: Input configuration
        main_dir: Main directory for QE files
    
    Returns:
        QeRunError or QeRunDone object, or None if execution fails
    """
    # Input validation - check the class name instead of using isinstance
    intake_type = type(intake).__name__
    if intake_type not in ('QeInputGenerate', 'QeRunErrorSolution'):
        raise ValueError(f"Invalid intake type: {intake_type}")
    
    input_content = intake.qe_input.qe_input_gen
    # write file
    with open(f'{qe_settings["main_dir"]}/tmp0_X.in', 'w') as f:
        f.write(input_content)
    
    main_dir = Path(qe_settings['main_dir'])
    output_dir = qe_settings['output_dir']

    # Format the command components
    if not qe_settings['use_slurm']:
        print("Running locally")
        activate_cmd = qe_settings['activate_command'].format(
            CONDA_PATH=qe_settings['conda_path'],
            QE_ENV=qe_settings['qe_env'],
            MAIN_DIR=main_dir
        )
        qe_prefix = qe_settings['qe_prefix'].format(
            n_cores=qe_settings['n_cores']
        )
        qe_command = f"{activate_cmd} && {qe_prefix} pw.x < tmp0_X.in > tmp0_X.out"

        # Run QE command and wait for completion
        def set_process_group():
            os.setpgrp()  # Create new process group

        try:
            result = subprocess.run(
                qe_command,
                shell=True,
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
                preexec_fn=set_process_group
            )
            
        except subprocess.TimeoutExpired as e:
            print(f"Process timed out after {e.timeout} seconds")
            # Try to kill the process
            try:
                subprocess.run(['pkill', '-f', 'pw.x'], check=False)
            except FileNotFoundError:
                try:
                    subprocess.run(['killall', 'pw.x'], check=False)
                except Exception as kill_error:
                    print(f"Warning: Could not kill process: {kill_error}")

        except Exception as e:
            print(f"Error during execution: {e}")

        # Common error handling for both timeout and other exceptions
        if os.path.exists(main_dir / 'CRASH'):
            output_content = parse_crash_error(main_dir / 'CRASH')
            return QeRunError(
                qe_input=QeMatch.model_validate(intake.qe_input.model_dump()),
                error_msg=CrashErr(error_msg=output_content)
            )
        else:
            # No CRASH file means calculation might be okay
            output_file = main_dir / 'tmp0_X.out'
            output_content = qe_output_parser(str(output_file))
            return QeRunDone(
                qe_input=QeMatch.model_validate(intake.qe_input.model_dump()),
                qe_output=DoneMsg(qe_output=output_content)
            )
    elif qe_settings['use_slurm']:
        print("Running with slurm")
        # submit slurm job
        activate_cmd = qe_settings['activate_command'].format(
            CONDA_PATH=qe_settings['conda_path'],
            QE_ENV=qe_settings['qe_env'],
            MAIN_DIR=qe_settings['main_dir']
        )

        prefix_cmd = qe_settings['qe_prefix'].format(
            n_cores=qe_settings['n_cores']
        )
        
        output_file = main_dir / 'tmp0_X.out'  # Define output_file
        input_file = main_dir / 'tmp0_X.in' # Input file
        
        sbatch_command = qe_settings['sbatch_script_template'].format(
            name='qroissant_job',
            module_name=qe_settings['module_name'],
            input_path=input_file.as_posix(),
            output_path=output_file.as_posix(),
            n_cores=qe_settings['n_cores'],
            output_dir=output_dir.as_posix(),
            qe_prefix=prefix_cmd,
            activate_command=activate_cmd
        )
        
        # write the sbatch command to a file
        sbatch_file = main_dir / 'sbatch_command.sh'
        sbatch_file.write_text(sbatch_command)
        
        try:
            # Submit the job and get job ID
            result = subprocess.run(f'sbatch {sbatch_file}', 
                                    shell=True, 
                                    capture_output=True, 
                                    text=True,
                                    check=True)
            
            # Clean up the script file
            sbatch_file.unlink()
            
            if os.path.exists(main_dir / 'CRASH'):
                output_content = parse_crash_error(main_dir / 'CRASH')
                return QeRunError(
                    qe_input=QeMatch.model_validate(intake.qe_input.model_dump()),
                    error_msg=CrashErr(error_msg=output_content)
                )
            else:
                output_content = qe_output_parser(str(output_file))
                if result.stdout:
                    output_content += f"\nSlurm submission output:\n{result.stdout}"
                
                return QeRunDone(
                    qe_input=QeMatch.model_validate(intake.qe_input.model_dump()),
                    qe_output=DoneMsg(qe_output=output_content)
                )
                
        except subprocess.CalledProcessError as e:
            print(f"Slurm job submission failed: {e}")
            return None
        
    return QeRunDone(
            qe_input=QeMatch.model_validate(intake.qe_input.model_dump()),
            qe_output=DoneMsg(qe_output=output_content)
        )
 

def error_kw(error_msg:str, lm_api: str) -> ErrorKw:
    """This function extracts the error keywords from the error message."""
    x_ = quick_chat(query= error_keywords_prompt.format(error_message= error_msg),
                    lm_api=lm_api,
                    model='mistralai/mixtral-8x22b-instruct', 
                    extras={'temperature': 0.0, 'max_tokens': 1500}
    )

    try:
        x_1 = re.findall(r'```python(.*)```', x_, re.DOTALL)[-1]
        # find keywords and abbreviations
        error_kw_ = re.findall(r'error_keywords = (.*?)(?:$)', x_1, re.DOTALL)[0].strip()
        error_kw_ast = ast.literal_eval(error_kw_)
        x_2 = ErrorKw(error_message= error_kw_ast['error_message'],
                        variables= error_kw_ast['variables'],
                        explanation= error_kw_ast['explanation']
                    )
        return x_2
    
    except Exception as e:
        print(f"FAILED: {e}")
        return None


async def qe_crash_handle(
    qe_out: QeRunError,
    lm_api: str,
    model_name: str = 'mistralai/mixtral-8x22b-instruct',
) -> Optional[QeRunErrorSolution]:
    """
    Handle Quantum Espresso crash errors by analyzing the error message and generating a solution.
    
    Args:
        qe_out: QeRunError object containing error information
        model_name: Name of the model to use for LLM processing
        llm: Language Model Engine class
    
    Returns:
        QeRunErrorSolution object if successful, None otherwise
    """
    # Input validation
    qe_out_type = type(qe_out).__name__
    if qe_out_type not in ('QeRunError', 'QeRunErrorSolution'):
        logger.error(f"Invalid input: expected QeRunError, got {qe_out_type}")
        return None
    
    # Extract error keywords
    logger.info(f"Processing error: {qe_out.error_msg.error_msg}")
    ekw_ = error_kw(qe_out.error_msg.error_msg, lm_api=lm_api)
    
    if not isinstance(ekw_, ErrorKw):
        logger.error("Failed to extract error keywords")
        return None
    
    # Process error documentation
    try:
        err_doc_ = []
        doc_results, _, _ = await QrDocInstance.a_b_from_query(
            query=ekw_.variables,
            token_pattern_=r"(?u)[a-zA-Z0-9]{2,}|[0-9]+"
        )
        
        for doc in doc_results:
            err_doc_.extend(doc.flatten().tolist())
        
        # Prepare error message and format prompt
        error_msg_ = "\n".join([
            ekw_.error_message,
            ", ".join(ekw_.variables),
            ekw_.explanation
        ])
        
        X__m2 = X__m.format(
            input_file=qe_out.qe_input.qe_input_gen,
            error_msg=error_msg_,
            doc="\n".join(err_doc_)
        )
        
        # Process with LLM
        llm_out_crash = quick_chat(
            query=X__m2,
            lm_api=lm_api,
            model=model_name,
            extras={'temperature': 0.0, 'max_tokens': 15000}
        )
        
        print(llm_out_crash)
        # Extract and validate matches
        matches = re.findall(QE_GEN_PATTERN, llm_out_crash, re.DOTALL)[-1]
        if not is_valid_qe_input(matches):
            logger.error("Generated input file validation failed")
            return None
            
        # Create solution
        return QeRunErrorSolution(
            qe_input=QeMatch(qe_input_gen=matches),
            error_msg=CrashErr(error_msg=error_msg_),
            solution_prompt=X__m2,
            solution=SOLUTION(solution=llm_out_crash)
        )
        
    except Exception as e:
        logger.error(f"Error in qe_crash_handle: {str(e)}", exc_info=True)
        return None
    

def is_valid_qe_input(content: str) -> bool:
    """
    Validates if a Quantum Espresso input file contains all required sections.
    
    Args:
        content: String containing the QE input file content
        
    Returns:
        bool: True if all required sections are present, False otherwise
        
    Note:
        Required sections are:
        - Namelist cards: &control, &system, &electrons
        - Data cards: ATOMIC_SPECIES, ATOMIC_POSITIONS, K_POINTS
    """
    if not content or not isinstance(content, str):
        logger.error("Invalid input: content must be a non-empty string")
        return False

    # Define required sections with their regex patterns
    required_sections = {
        'namelists': {
            '&control': r'&control.*?/',
            '&system': r'&system.*?/',
            '&electrons': r'&electrons.*?/'
        },
        'cards': {
            'ATOMIC_SPECIES': r'ATOMIC_SPECIES\s*\n',
            'ATOMIC_POSITIONS': r'ATOMIC_POSITIONS\s*[\{\(]?.*?[\}\)]?\s*\n',
            'K_POINTS': r'K_POINTS\s*[\{\(]?.*?[\}\)]?\s*\n'
        }
    }

    try:
        # Compile regex patterns once (optimization)
        patterns = {
            name: re.compile(pattern, re.IGNORECASE | re.DOTALL)
            for section in required_sections.values()
            for name, pattern in section.items()
        }

        # Check all sections
        missing = []
        found = {name: bool(patterns[name].search(content))
                for name in patterns}

        # Collect missing sections
        for section_type, sections in required_sections.items():
            missing.extend(name for name in sections
                         if not found[name])

        if missing:
            logger.warning(f"Missing required sections: {', '.join(missing)}")
            return False

        return True

    except re.error as e:
        logger.error(f"Regex error while validating input: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error while validating input: {e}")
        return False


def is_valid_qe_input_old(content:str) -> bool:
    required_cards = {'&control', '&system', '&electrons'}
    found_cards = set()
    has_atomic_species = False
    has_atomic_positions = False
    has_k_points = False

    try:
        # Check for required cards
        for card in required_cards:
            if re.search(rf'{card}.*?/', content, re.IGNORECASE | re.DOTALL):
                found_cards.add(card)

        # Check for ATOMIC_SPECIES
        if re.search(r'ATOMIC_SPECIES', content, re.IGNORECASE):
            has_atomic_species = True

        # Check for ATOMIC_POSITIONS
        if re.search(r'ATOMIC_POSITIONS', content, re.IGNORECASE):
            has_atomic_positions = True

        # Check for K_POINTS
        if re.search(r'K_POINTS', content, re.IGNORECASE):
            has_k_points = True

        # Evaluate conditions
        all_cards_present = found_cards == required_cards
        all_sections_present = has_atomic_species and has_atomic_positions and has_k_points

        if all_cards_present and all_sections_present:
            return True
        else:
            missing = []
            if not all_cards_present:
                missing.extend(required_cards - found_cards)
            if not has_atomic_species:
                missing.append("ATOMIC_SPECIES")
            if not has_atomic_positions:
                missing.append("ATOMIC_POSITIONS")
            if not has_k_points:
                missing.append("K_POINTS")
                
            print(f"Missing: {missing}")
            return False
        
    except Exception as e:
        print(f"Error: {e}")
        return None
    
###############   
#def qe_crash_handle_old(qe_out:QeRunError,
#    model_name = 'dbrx',
#    llm = LMEngine) -> QeRunErrorSolution:
#
#    if not isinstance(qe_out, QeRunError):
#        print('Not a QE_run_error')
#        return None
#    
#    print(qe_out.error_msg.error_msg)
#    
#    ekw_ = error_kw(qe_out.error_msg.error_msg)
#    
#    if not isinstance(ekw_, ErrorKw):
#        print('No error keywords found')
#        return None
#    
#    print(ekw_)
#    err_doc_ , _, _ = QrDocInstance.a_b_from_query(query= ekw_.variables,
#                                     token_pattern_=r"(?u)[a-zA-Z0-9]{2,}|[0-9]+"
#                                    )
#    err_doc_ = []    
#    for i in err_doc_:
#        err_doc_.extend(i.flatten().tolist()) 
#
#    input_file_ = qe_out.qe_input.qe_input_gen
#    error_msg_ = ekw_.error_message + '\n' + ', '.join(ekw_.variables) + '\n' + ekw_.explanation
#    X__m2 = X__m.format( input_file= input_file_,
#                        error_msg= error_msg_,
#                        doc= '\n'.join(err_doc_))
#    
#    with llm(lm_api, model= model_dict[model_name],  kwarg=default_llm_kwarg_qe_gen) as jam:
#        
#        llm_out = jam(X__m2)[0]
#    
#    try:
#        matches = re.findall(QE_GEN_PATTERN, llm_out, re.DOTALL)[-1]
#        if is_valid_qe_input(matches):
#            return QeRunErrorSolution(
#                qe_input= QeMatch(qe_input_gen= matches),
#                error_msg= CrashErr(error_msg = error_msg_),
#                solution_prompt= X__m2,
#                solution= SOLUTION(solution = llm_out)
#            )    
#        else:
#            print('failed to generate a valid input file')
#            return None
#
#
#    except Exception as e:
#        print(f"FAILED: {e}")
#        return None
#        
#
#
#def llm_in_and_out_old(input_prompt:str,
#                   llm = LMEngine,
#                   model_name = 'dbrx',
#                   ) -> QeInputGenerate:
#
#    with llm(model= model_dict[model_name], lm_api= LM_API, kwarg=default_llm_kwarg_qe_gen) as jam:
#        llm_out = jam(input_prompt)[0]
#        
#    try:
#        matches = re.findall(QE_GEN_PATTERN, llm_out, re.DOTALL)[-1]
#        if not is_valid_qe_input(matches):
#            raise ValueError('invalid input')
#        else:
#            return QeInputGenerate(
#                qe_input= QeMatch(qe_input_gen= matches)
#            )
#    
#    except Exception as e:
#        print(f"FAILED: {e}")
#        return None
#
#
#def qe_run_old(intake:QeInputGenerate|QeRunErrorSolution,
#            override=True) -> QeRunError|QeRunDone:
#
#    import subprocess
#    input = intake.qe_input.qe_input_gen
#    if override:
#        with open(f'{MAIN_DIR}/tmp0_X.in', 'w') as f:
#            f.write(input)
#
#    QE_COMMAND = QE_COMMAND_ + f" pw.x < tmp0_X.in > tmp0_X.out"
#    # Combine activation and Quantum ESPRESSO commands
#    command = f"{ACTIVATE_COMMAND} && {QE_COMMAND}"
#    # Execute the command in the shell
#    try:
#        subprocess.run(command, shell=True)
#        if os.path.exists(f'{MAIN_DIR}/CRASH'):
#            with open(f'{MAIN_DIR}/CRASH') as file:
#                exi = file.read()
#            error_ = re.findall(r"%+\s+(.*?)%+",   exi, re.DOTALL)[-1].strip().split('\n')[-1].strip() 
#
#            return QeRunError(
#                qe_input= QeMatch.validate(intake.qe_input.dict()),
#                error_msg= CrashErr(error_msg = error_) ,
#            )
#        
#        else:
#
#            strs = qe_output_parser(f'{MAIN_DIR}/tmp0_X.out')
#            return QeRunDone(
#                qe_input= QeMatch.model_validate(intake.qe_input.model_dump()),
#                qe_output= DoneMsg(qe_output= strs),
#            )
#    except Exception as e:
#        print(e)
#        return None
#