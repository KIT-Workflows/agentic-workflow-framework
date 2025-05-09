from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import traceback
from pathlib import Path
from ..assets.graph_engine import save_results_json, visualize_log_timeline
from .workflow_backend import (
    WorkflowContext, 
    WorkflowController, 
    QroissantProject, 
    ProjectConfig,
    default_qe_settings,
    save_results_json,
)

from fastapi.middleware.cors import CORSMiddleware
import asyncio
from datetime import datetime
import queue
import logging
import sys
import io
import json
import uuid
from fastapi import BackgroundTasks
import re

app = FastAPI(title="Workflow API")

# Get the root directory (2 levels up from current file)
ROOT_DIR = Path(__file__).parent.parent.parent

# Mount static files with absolute path
app.mount("/static", StaticFiles(directory=str(ROOT_DIR / "static")), name="static")

# Setup Jinja2 templates with absolute path
templates = Jinja2Templates(directory=str(ROOT_DIR / "templates"))

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a queue to store log messages
log_queue = queue.Queue()

# Function to strip ANSI color codes
def strip_ansi_codes(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

# Custom stream handler to capture logs
class QueueHandler(logging.Handler):
    def emit(self, record):
        # Strip ANSI codes from the message
        clean_message = strip_ansi_codes(self.format(record))
        log_entry = {
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'message': clean_message,
            'level': record.levelname.lower()
        }
        # Filter out workflow-status polling messages
        if not clean_message.startswith('127.0.0.1') and 'workflow-status' not in clean_message:
            log_queue.put(log_entry)

# Set up logging for our app
logger = logging.getLogger('workflow_api')
queue_handler = QueueHandler()
queue_handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(queue_handler)

# Modified StreamToLogger that works with Uvicorn
class StreamToLogger:
    def __init__(self, level):
        self.level = level
        self.linebuf = ''
        self.logger = logging.getLogger('workflow_api')
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

    def write(self, buf):
        if buf.strip():  # Only log non-empty lines
            for line in buf.rstrip().splitlines():
                self.logger.log(self.level, line.rstrip())
        # Also write to original stdout/stderr
        if self.level == logging.INFO:
            self.original_stdout.write(buf)
        else:
            self.original_stderr.write(buf)
    
    def flush(self):
        if self.level == logging.INFO:
            self.original_stdout.flush()
        else:
            self.original_stderr.flush()

    def isatty(self):
        # Delegate to original stdout/stderr
        return (self.original_stdout.isatty() if self.level == logging.INFO 
                else self.original_stderr.isatty())

    def fileno(self):
        # Delegate to original stdout/stderr
        return (self.original_stdout.fileno() if self.level == logging.INFO 
                else self.original_stderr.fileno())

# Store original stdout/stderr
original_stdout = sys.stdout
original_stderr = sys.stderr

# Replace stdout and stderr with our logging versions
sys.stdout = StreamToLogger(logging.INFO)
sys.stderr = StreamToLogger(logging.ERROR)

# Add route for the HTML page
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("wf_app.html", {"request": request})

# Add a test route to verify API is working
@app.get("/ping")
async def ping():
    return {"status": "ok", "message": "API is running"}

class WorkflowRequest(BaseModel):
    calculation_prompt: Optional[str] = None
    workflow_index: Optional[int] = 0
    gen_model_hierarchy: List[str] = ['mistralai/mixtral-8x22b-instruct', 'meta-llama/llama-3.1-405b-instruct', 'anthropic/claude-3.5-sonnet']
    model_config: Dict[str, str] = {
        'parameter_evaluation': 'mistralai/mixtral-8x22b-instruct',
        'condition_finder': 'mistralai/mixtral-8x22b-instruct'
    }
    interface_agent_kwargs: Dict[str, Any] = {
        'model': 'mistral',
        'extras': {'max_tokens': 8000}
    }
    lm_api: str
    project_config: Optional[Dict[str, Any]] = None

# Add the SSE endpoint
@app.get("/logs")
async def stream_logs():
    async def event_generator():
        while True:
            try:
                if not log_queue.empty():
                    log_entry = log_queue.get_nowait()
                    yield f"data: {json.dumps(log_entry)}\n\n"
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"Error in event generator: {e}")
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

# Add a dictionary to track active workflows
active_workflows: Dict[str, WorkflowController] = {}

@app.post("/workflow/")
async def create_workflow(request: WorkflowRequest, background_tasks: BackgroundTasks):
    try:
        logger.info("Starting workflow processing")
        workflow_id = str(uuid.uuid4())  # Generate unique ID for this workflow
        
        if request.lm_api is None:
            logger.error("LM_API is not set")
            raise HTTPException(status_code=400, detail="LM_API is not set")

        # Set up default project configuration
        current_dir = Path.cwd()
        out_dir = current_dir / 'out_dir'
        pseudopotentials_dir = current_dir / 'new_pp'

        # Ensure directories exist
        out_dir.mkdir(exist_ok=True)
        pseudopotentials_dir.mkdir(exist_ok=True)

        # Update QE settings
        updated_qe_settings = default_qe_settings.copy()
        updated_qe_settings['n_cores'] = 8
        updated_qe_settings['use_slurm'] = False
        updated_qe_settings['qe_prefix'] = ''

        # Create base config
        base_config = {
            'name': 'test',
            'participants': ['test'],
            'metadata': {'test': 'test'},
            'main_dir': current_dir,
            'output_dir': out_dir,
            'pseudopotentials': pseudopotentials_dir,
            'project_signature': 'test',
            'num_retries_permodel': 3,
            'async_batch_size': 10,
            'MAXDocToken': 19000,
            'gen_model_hierarchy': request.gen_model_hierarchy,
            'model_config': request.model_config,
            'lm_api': request.lm_api,
            'qe_settings': updated_qe_settings
        }

        # Update with user-provided config if any
        if request.project_config:
            base_config.update(request.project_config)

        # Create ProjectConfig instance
        try:
            config = ProjectConfig(**base_config)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create ProjectConfig: {str(e)}")

        # Initialize workflow context and controller
        workflow_context = WorkflowContext(
            project=QroissantProject(config),
            workflow_index=request.workflow_index,
            calculation_prompt=request.calculation_prompt,
            lm_func_kwargs=request.interface_agent_kwargs
        )
        
        controller = WorkflowController(workflow_context)
        active_workflows[workflow_id] = controller  # Store the controller
        
        # Start workflow in background task
        background_tasks.add_task(run_workflow, controller, workflow_id)
        
        return {
            "status": "started",
            "workflow_id": workflow_id,
            "message": "Workflow started successfully"
        }

    except Exception as e:
        logger.error(f"Error in workflow: {str(e)}")
        traceback_str = traceback.format_exc()
        logger.error(traceback_str)
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "traceback": traceback_str
            }
        )

async def run_workflow(controller: WorkflowController, workflow_id: str):
    try:
        await controller.run_workflow()
        
        # Save results after workflow completion
        workflow_context = controller.context
        try:
            if workflow_context:
                logger.info(f"Saving results for workflow {workflow_id}")
                save_path_filename = save_results_json(
                    project=workflow_context.project,
                    fsm_state=controller.fsm.current_state_name,
                    workflow_index=workflow_id,
                    save_dir='wf_api_results'
                )
                save_path = save_path_filename.parent

                # Visualize results
                with open(save_path_filename, 'r') as f:
                    data = json.load(f)
                logger.info(f"Json saved to: {save_path_filename}")
                visualize_log_timeline(
                    data[0]['workflow_log'],
                    save_path=save_path / f'qe_results_{workflow_id}.png'
                )
                logger.info(f"Visualization saved to: {save_path / f'qe_results_{workflow_id}.png'}")

        except Exception as e:
            logger.error(f"Error occurred during saving/visualization: {str(e)}")
            traceback.print_exc()
            
    finally:
        # Clean up when workflow finishes or is stopped
        if workflow_id in active_workflows:
            del active_workflows[workflow_id]

@app.post("/stop-workflow/{workflow_id}")
async def stop_workflow(workflow_id: str):
    try:
        if workflow_id not in active_workflows:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        controller = active_workflows[workflow_id]
        logger.info(f"Stopping workflow: {workflow_id}")
        
        # Trigger graceful shutdown
        await controller.stop_workflow()
        
        return {
            "status": "success",
            "message": "Workflow stopped successfully"
        }
    except Exception as e:
        logger.error(f"Error stopping workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/workflow-status/{workflow_id}")
async def get_workflow_status(workflow_id: str):
    if workflow_id not in active_workflows:
        return {"status": "not_found"}
    
    controller = active_workflows[workflow_id]
    return {
        "status": "running" if controller.is_running else "stopped",
        "current_state": controller.fsm.current_state_name
    }

# Define the correct results directory
RESULTS_DIR = Path("wf_api_results")
RESULTS_DIR.mkdir(exist_ok=True)  # Ensure directory exists

@app.get("/results/{workflow_id}")
async def get_results(workflow_id: str):
    try:
        # Look for files containing the workflow_id
        possible_files = list(RESULTS_DIR.glob(f'*{workflow_id}*.json'))
        if not possible_files:
            raise HTTPException(status_code=404, detail="Results file not found")
        
        # Get the most recent file if multiple exist
        results_path = sorted(possible_files)[-1]
        logger.info(f"Found results file: {results_path}")
        
        with open(results_path) as f:
            return JSONResponse(content=json.load(f))
    except Exception as e:
        logger.error(f"Error getting results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/timeline/{workflow_id}")
async def get_timeline(workflow_id: str):
    try:
        # Look for files containing the workflow_id
        possible_files = list(RESULTS_DIR.glob(f'*{workflow_id}*.png'))
        if not possible_files:
            raise HTTPException(status_code=404, detail="Timeline plot not found")
        
        # Get the most recent file if multiple exist
        timeline_path = sorted(possible_files)[-1]
        logger.info(f"Found timeline file: {timeline_path}")
        
        return FileResponse(timeline_path)
    except Exception as e:
        logger.error(f"Error getting timeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add a debug endpoint to list matching files
@app.get("/debug/files/{workflow_id}")
async def list_matching_files(workflow_id: str):
    try:
        json_files = list(RESULTS_DIR.glob(f'*{workflow_id}*.json'))
        png_files = list(RESULTS_DIR.glob(f'*{workflow_id}*.png'))
        
        return {
            "json_files": [str(f.name) for f in json_files],
            "png_files": [str(f.name) for f in png_files],
            "workflow_id": workflow_id
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)