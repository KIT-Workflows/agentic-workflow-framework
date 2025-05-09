# Workflow API Technical Documentation

## Overview
The Workflow API is a FastAPI-based service that manages computational workflows with machine learning models. It provides endpoints for creating, monitoring, and managing workflow executions.

## Base URL
```
http://127.0.0.1:8001
```

## Authentication
Currently, the API does not require authentication but requires an LM API key in the workflow request.

## Endpoints

### Health Check
```
GET /ping
```
Verifies if the API is running.

**Response:**
```json
{
    "status": "ok",
    "message": "API is running"
}
```

### Create Workflow
```
POST /workflow/
```
Creates and starts a new workflow execution.

**Request Body:**
```json
{
    "calculation_prompt": string,
    "workflow_index": integer | null,
    "gen_model_hierarchy": [
        string,
        string,
        string
    ],
    "model_config": {
        "parameter_evaluation": string,
        "condition_finder": string
    },
    "interface_agent_kwargs": {
        "model": string,
        "extras": {
            "max_tokens": integer
        }
    },
    "lm_api": string,
    "project_config": object | null
}
```

**Response:**
```json
{
    "status": "started",
    "workflow_id": string,
    "message": "Workflow started successfully"
}
```

### Check Workflow Status
```
GET /workflow-status/{workflow_id}
```
Returns the current status of a workflow.

**Response:**
```json
{
    "status": "running" | "stopped" | "not_found",
    "current_state": string
}
```

### Stop Workflow
```
POST /stop-workflow/{workflow_id}
```
Stops a running workflow.

**Response:**
```json
{
    "status": "success",
    "message": "Workflow stopped successfully"
}
```

### Get Results
```
GET /results/{workflow_id}
```
Retrieves the results of a completed workflow.

### Get Timeline Visualization
```
GET /timeline/{workflow_id}
```
Returns a PNG visualization of the workflow timeline.

### Stream Logs
```
GET /logs
```
Provides real-time log streaming using Server-Sent Events (SSE).

## Error Handling
The API returns standard HTTP status codes:
- 200: Success
- 400: Bad Request
- 404: Not Found
- 500: Internal Server Error

Error responses include detailed messages and, when applicable, stack traces.

## Directory Structure
- `/wf_api_results`: Contains workflow results and visualizations
- `/static`: Static files
- `/templates`: HTML templates

## Models and Configuration
### Default Model Hierarchy
1. databricks/dbrx-instruct
2. meta-llama/llama-3.1-405b-instruct
3. anthropic/claude-3.5-sonnet

### Default Model Configuration
- parameter_evaluation: mistralai/mixtral-8x22b-instruct
- condition_finder: mistralai/mixtral-8x22b-instruct

## Logging
The API implements a custom logging system that:
- Captures both stdout and stderr
- Strips ANSI color codes
- Provides real-time log streaming
- Filters out unnecessary system messages

## Request Templates

### Project Configuration Template
The following template shows the structure of a project configuration object that can be included in workflow requests:

```json
{
    "project_config": {
        "name": string,
        "participants": string[],
        "metadata": {
            "project_type": string,
            "material": string,
            "purpose": string,
            [key: string]: any
        },
        "main_dir": string (path),
        "output_dir": string (path),
        "pseudopotentials": string (path),
        "project_signature": string,
        "num_retries_permodel": integer,
        "async_batch_size": integer,
        "MAXDocToken": integer,
        "gen_model_hierarchy": string[],
        "model_config": {
            "parameter_evaluation": string,
            "condition_finder": string
        },
        "qe_settings": {
            "use_slurm": boolean,
            "qe_env": string,
            "n_cores": integer,
            "conda_path": string (path),
            "qe_prefix": string,
            "module_name": string
        }
    }
}
```

**Example:**
```json
{
    "project_config": {
        "name": "CeO2_optimization",
        "participants": ["researcher1", "researcher2"],
        "metadata": {
            "project_type": "material_optimization",
            "material": "CeO2",
            "purpose": "band_structure_calculation"
        },
        "main_dir": "/path/to/your/project",
        "output_dir": "/path/to/your/project/output",
        "pseudopotentials": "/path/to/your/project/pseudo",
        "project_signature": "CeO2_opt_v1",
        "num_retries_permodel": 5,
        "async_batch_size": 10,
        "MAXDocToken": 19000,
        "gen_model_hierarchy": [
            "databricks/dbrx-instruct",
            "meta-llama/llama-3.1-405b-instruct",
            "anthropic/claude-3.5-sonnet"
        ],
        "model_config": {
            "parameter_evaluation": "mistralai/mixtral-8x22b-instruct",
            "condition_finder": "mistralai/mixtral-8x22b-instruct"
        },
        "qe_settings": {
            "use_slurm": false,
            "qe_env": "qe",
            "n_cores": 8,
            "conda_path": "/path/to/conda/etc/profile.d/conda.sh",
            "qe_prefix": "mpirun -n 8",
            "module_name": "qe"
        }
    }
}
```

**Field Descriptions:**
- `name`: Project identifier
- `participants`: List of project team members
- `metadata`: Custom metadata for project documentation
- `main_dir`: Main project directory path
- `output_dir`: Directory for output files
- `pseudopotentials`: Directory containing pseudopotential files
- `project_signature`: Unique project identifier
- `num_retries_permodel`: Maximum retry attempts per model
- `async_batch_size`: Batch size for async operations
- `MAXDocToken`: Maximum tokens for documentation
- `gen_model_hierarchy`: Ordered list of ML models to try
- `model_config`: Configuration for specific model tasks
- `qe_settings`: Quantum Espresso execution settings

**Notes:**
- All paths should be absolute paths or relative to the project root
- The `qe_settings` should match your computational environment
- Model names should be valid and accessible in your environment

## Usage Guide

### Getting Started
1. Ensure the API server is running (default: http://127.0.0.1:8001)
2. Verify connectivity using the health check endpoint
3. Prepare your workflow configuration
4. Submit your workflow request

### Basic Workflow Example
Here's a step-by-step guide to create and monitor a workflow:

1. **Check API Health**
```bash
curl http://127.0.0.1:8001/ping
```

2. **Create a New Workflow**
Basic workflow request:
```bash
curl -X POST http://127.0.0.1:8001/workflow/ \
  -H "Content-Type: application/json" \
  -d '{
    "calculation_prompt": "Calculate band structure for CeO2",
    "workflow_index": 0,
    "lm_api": "your_api_key_here",
    "gen_model_hierarchy": [
        "databricks/dbrx-instruct",
        "meta-llama/llama-3.1-405b-instruct",
        "anthropic/claude-3.5-sonnet"
    ],
    "model_config": {
        "parameter_evaluation": "mistralai/mixtral-8x22b-instruct",
        "condition_finder": "mistralai/mixtral-8x22b-instruct"
    }
}'
```

3. **Monitor Workflow Status**
```bash
curl http://127.0.0.1:8001/workflow-status/{workflow_id}
```

4. **Get Results When Complete**
```bash
curl http://127.0.0.1:8001/results/{workflow_id}
```

5. **View Timeline Visualization**
```bash
curl -O http://127.0.0.1:8001/timeline/{workflow_id}
```

### Advanced Usage with Project Configuration

#### Complete Workflow Request with Custom Project Configuration
```bash
curl -X POST http://127.0.0.1:8001/workflow/ \
  -H "Content-Type: application/json" \
  -d '{
    "calculation_prompt": "Calculate band structure for CeO2",
    "workflow_index": 0,
    "lm_api": "your_api_key_here",
    "gen_model_hierarchy": [
        "databricks/dbrx-instruct",
        "meta-llama/llama-3.1-405b-instruct",
        "anthropic/claude-3.5-sonnet"
    ],
    "model_config": {
        "parameter_evaluation": "mistralai/mixtral-8x22b-instruct",
        "condition_finder": "mistralai/mixtral-8x22b-instruct"
    },
    "interface_agent_kwargs": {
        "model": "mistral",
        "extras": {
            "max_tokens": 8000
        }
    },
    "project_config": {
        "name": "CeO2_optimization",
        "participants": ["researcher1", "researcher2"],
        "metadata": {
            "project_type": "material_optimization",
            "material": "CeO2",
            "purpose": "band_structure_calculation"
        },
        "main_dir": "/path/to/your/project",
        "output_dir": "/path/to/your/project/output",
        "pseudopotentials": "/path/to/your/project/pseudo",
        "project_signature": "CeO2_opt_v1",
        "num_retries_permodel": 5,
        "async_batch_size": 10,
        "MAXDocToken": 19000,
        "qe_settings": {
            "use_slurm": false,
            "qe_env": "qe",
            "n_cores": 8,
            "conda_path": "/path/to/conda/etc/profile.d/conda.sh",
            "qe_prefix": "mpirun -n 8",
            "module_name": "qe"
        }
    }
}'
```

### Real-time Monitoring

#### Stream Logs
To monitor logs in real-time using curl:
```bash
curl -N http://127.0.0.1:8001/logs
```

Or using JavaScript in a web browser:
```javascript
const eventSource = new EventSource('http://127.0.0.1:8001/logs');
eventSource.onmessage = function(event) {
    const log = JSON.parse(event.data);
    console.log(`${log.timestamp} [${log.level}]: ${log.message}`);
};
```

### Common Workflows

1. **Basic Material Calculation**
   - Use default configuration
   - Provide calculation prompt
   - Monitor results

2. **Advanced Material Optimization**
   - Custom project configuration
   - Multiple model hierarchy
   - Custom computational resources

3. **Batch Processing**
   - Configure async_batch_size
   - Monitor multiple workflows
   - Collect results

### Project Configuration Guide

#### Essential Fields
1. **Project Information**
   - `name`: Unique identifier for your project
   - `project_signature`: Version or signature
   - `participants`: Team members involved

2. **Directory Structure**
   - `main_dir`: Root directory for project
   - `output_dir`: Results storage
   - `pseudopotentials`: QE pseudopotential files

3. **Computational Settings**
   - `n_cores`: Number of CPU cores
   - `use_slurm`: Cluster submission
   - `qe_prefix`: MPI configuration

4. **Model Settings**
   - `gen_model_hierarchy`: Fallback model sequence
   - `model_config`: Task-specific models
   - `MAXDocToken`: Token limit for documentation

### Best Practices

1. **Directory Structure**
   ```
   project_root/
   ├── output/
   ├── pseudo/
   └── results/
   ```

2. **Error Handling**
   - Set appropriate `num_retries_permodel`
   - Monitor logs for failures
   - Check status regularly

3. **Resource Management**
   - Match `n_cores` to available resources
   - Configure `async_batch_size` appropriately
   - Monitor system resources

4. **Model Selection**
   - Order models by preference in hierarchy
   - Match models to task requirements
   - Consider token limits
