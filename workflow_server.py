from eng.interfaces.workflow_api import app
import socket
import logging

logger = logging.getLogger(__name__)

def main():
    import uvicorn
    
    # Find available port
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

    # Run the server with the available port
    uvicorn.run(
        "workflow_server:app",
        host="127.0.0.1",
        port=port,
        reload=True
    )

if __name__ == "__main__":
    main() 