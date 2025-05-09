from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
import logging

# Configure logging first so it's available everywhere
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Add configuration class
class Settings:
    JSON_FILE_PATH = 'eng/assets/xqe_univ_kg_load_v1.json'
    DEBUG = True

settings = Settings()

# Improve JSON loading with better error handling
def load_json_data():
    try:
        with open(settings.JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                logger.error("JSON data must be a list")
                return []
            return data
    except FileNotFoundError:
        logger.error(f"JSON file not found at {settings.JSON_FILE_PATH}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error loading JSON: {str(e)}")
        return []

# Load json data
json_data = load_json_data()

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse(
        "index.html", 
        {"request": request}
    )

@app.get("/api/data")
async def get_data():
    if not json_data:
        raise HTTPException(
            status_code=503, 
            detail="Service temporarily unavailable - Data failed to load"
        )
    
    try:
        nodes = []
        edges = []
        seen_nodes = set()
        
        # First pass: collect all valid nodes
        for item in json_data:
            if not isinstance(item, dict):
                continue
                
            node_id = item.get("Parameter_Name") or item.get("Card_Name")
            if not node_id or not isinstance(node_id, str):
                continue
                
            if node_id in seen_nodes:
                logger.warning(f"Duplicate node found: {node_id}")
                continue
            
            seen_nodes.add(node_id)
            nodes.append({
                "id": node_id,
                "label": node_id,
                "group": item.get("Namelist", "") or item.get("Card_Name", ""),
                "title": (f"{item.get('Description', 'No description')}\n"
                         f"Type: {item.get('Value_Type', 'Not specified')}")
            })
        
        # Second pass: add edges only between existing nodes
        for item in json_data:
            if not isinstance(item, dict):
                continue
                
            source_id = item.get("Parameter_Name") or item.get("Card_Name")
            if not source_id or source_id not in seen_nodes:
                continue
                
            relationships = item.get("Relationships_Conditions_to_Other_Parameters_Cards", {})
            if not isinstance(relationships, dict):
                continue
                
            for target_id, description in relationships.items():
                # Only create edge if both source and target nodes exist
                if target_id and isinstance(target_id, str) and target_id in seen_nodes:
                    edges.append({
                        "id": f"edge-{source_id}-{target_id}",  # Add unique edge ID
                        "source": source_id,  # Use 'source' instead of 'from'
                        "target": target_id,  # Use 'target' instead of 'to'
                        "title": str(description) if description else "Related"
                    })
        
        logger.info(f"Successfully processed {len(nodes)} nodes and {len(edges)} edges")
        return {"nodes": nodes, "edges": edges}
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing data")

@app.get("/api/node/{node_id}")
async def get_node_details(node_id: str):
    if not node_id or not isinstance(node_id, str):
        raise HTTPException(status_code=400, detail="Valid node ID string is required")
        
    # First check for exact matches
    for item in json_data:
        if not isinstance(item, dict):
            continue
            
        if (item.get("Parameter_Name") == node_id or 
            item.get("Card_Name") == node_id):
            return item
    
    # Then check for case-insensitive matches
    node_id_lower = node_id.lower()
    for item in json_data:
        if not isinstance(item, dict):
            continue
            
        if (str(item.get("Parameter_Name", "")).lower() == node_id_lower or 
            str(item.get("Card_Name", "")).lower() == node_id_lower):
            return item
    
    # Handle namelist groups
    if node_id.startswith('&'):
        return {
            "Parameter_Name": node_id,
            "Type": "Namelist",
            "Description": "A namelist group that contains related parameters.",
            "Group": "Namelist"
        }
    
    raise HTTPException(
        status_code=404, 
        detail=f"Node '{node_id}' not found"
    )

def main():
    import uvicorn
    import socket
    
    # Log startup information
    logger.info("Starting FastAPI application")
    logger.info(f"Loaded {len(json_data)} items from JSON file")
    
    # Try ports starting from 8001 until we find an available one
    port = 8001
    while port < 8020:  # Limit port search to reasonable range
        try:
            # Test if port is available
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                break
        except OSError:
            logger.warning(f"Port {port} is in use, trying next port")
            port += 1
    else:
        logger.error("No available ports found in range 8001-8019")
        raise SystemExit(1)
    
    logger.info(f"Starting server on port {port}")
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=port,
        reload=True
    )

if __name__ == "__main__":
    main()
