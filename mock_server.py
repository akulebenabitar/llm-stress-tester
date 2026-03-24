#!/usr/bin/env python3
"""
Simple mock LMStudio server for testing the stress tester script.
This mimics the OpenAI-compatible API endpoints that LMStudio exposes.
"""

import json
import time
import random
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import sys

class MockLMStudioHandler(BaseHTTPRequestHandler):
    """Handler for mock LMStudio API endpoints"""
    
    # Track request counts for rate limiting simulation
    request_count = 0
    request_times = []
    lock = threading.Lock()
    
    def do_GET(self):
        """Handle GET requests"""
        path = urlparse(self.path).path
        
        if path == '/v1/models':
            self.handle_models()
        else:
            self.send_error(404, "Endpoint not found")
    
    def do_POST(self):
        """Handle POST requests"""
        path = urlparse(self.path).path
        
        if path == '/v1/chat/completions':
            self.handle_chat_completions()
        elif path == '/v1/completions':
            self.handle_completions()
        else:
            self.send_error(404, "Endpoint not found")
    
    def handle_models(self):
        """Handle /v1/models endpoint"""
        response = {
            "data": [
                {
                    "id": "mock-llama-3-8b-instruct",
                    "object": "model",
                    "owned_by": "mock"
                },
                {
                    "id": "mock-mistral-7b-instruct",
                    "object": "model",
                    "owned_by": "mock"
                }
            ]
        }
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode('utf-8'))
    
    def handle_chat_completions(self):
        """Handle /v1/chat/completions endpoint"""
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        
        try:
            request_data = json.loads(post_data.decode('utf-8'))
        except:
            self.send_error(400, "Invalid JSON")
            return
        
        # Simple rate limiting simulation
        with self.lock:
            self.request_count += 1
            current_time = time.time()
            self.request_times.append(current_time)
            
            # Clean old times (keep last 60 seconds)
            self.request_times = [t for t in self.request_times if current_time - t < 60]
            
            # Simulate rate limiting after 100 requests in 60 seconds
            if len(self.request_times) > 100:
                self.send_response(429)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                error_response = {
                    "error": {
                        "message": "Rate limit exceeded",
                        "type": "rate_limit_error",
                        "code": 429
                    }
                }
                self.wfile.write(json.dumps(error_response).encode('utf-8'))
                return
        
        # Check for required fields
        if 'model' not in request_data:
            self.send_error(400, "Missing 'model' field")
            return
        
        if 'messages' not in request_data:
            self.send_error(400, "Missing 'messages' field")
            return
        
        # Simulate some processing time
        time.sleep(random.uniform(0.1, 0.5))
        
        # Generate a mock response
        response = {
            "id": f"chatcmpl-{random.randint(1000000, 9999999)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request_data['model'],
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "This is a mock response from the test server. Hello!"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 15,
                "total_tokens": 25
            }
        }
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode('utf-8'))
    
    def handle_completions(self):
        """Handle /v1/completions endpoint"""
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        
        try:
            request_data = json.loads(post_data.decode('utf-8'))
        except:
            self.send_error(400, "Invalid JSON")
            return
        
        # Simulate some processing time
        time.sleep(random.uniform(0.1, 0.3))
        
        response = {
            "id": f"cmpl-{random.randint(1000000, 9999999)}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": request_data.get('model', 'mock-model'),
            "choices": [
                {
                    "text": "This is a mock completion response.",
                    "index": 0,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 10,
                "total_tokens": 15
            }
        }
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode('utf-8'))
    
    def log_message(self, format, *args):
        """Override to reduce log noise"""
        pass


def run_mock_server(port=1234, host='localhost'):
    """Run the mock LMStudio server"""
    server_address = (host, port)
    httpd = HTTPServer(server_address, MockLMStudioHandler)
    
    print(f"Starting mock LMStudio server on http://{host}:{port}")
    print("Press Ctrl+C to stop the server")
    print("\nAvailable endpoints:")
    print("  GET  /v1/models")
    print("  POST /v1/chat/completions")
    print("  POST /v1/completions")
    print("\nMock models: mock-llama-3-8b-instruct, mock-mistral-7b-instruct")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down mock server...")
        httpd.shutdown()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Mock LMStudio server for testing")
    parser.add_argument('--port', '-p', type=int, default=1234, help='Port to run on (default: 1234)')
    parser.add_argument('--host', '-b', default='localhost', help='Host to bind to (default: localhost)')
    
    args = parser.parse_args()
    
    run_mock_server(port=args.port, host=args.host)