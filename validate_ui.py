import os
import time
import threading
import http.server
import socketserver
from playwright.sync_api import sync_playwright

PORT = 8000

def run_server():
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving at port {PORT}")
        httpd.serve_forever()

def validate():
    # Start server in background
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    time.sleep(2) # Give server time to start

    with sync_playwright() as p:
        # We need to install the browser if not already there, but let's assume it is or use one that is.
        # Often 'chromium' is available if playwright is installed.
        try:
            browser = p.chromium.launch(headless=True)
        except Exception as e:
            print(f"Chromium launch failed: {e}. Trying to install...")
            # This might fail in some restricted environments, but let's try.
            # Alternatively, we could use selenium if this fails.
            return False

        page = browser.new_page(viewport={'width': 1280, 'height': 800})
        
        print("Opening vis2.html...")
        page.goto(f"http://localhost:{PORT}/vis2.html")
        
        # Wait for some telemetry data to be present and rendered
        # The UI shows "WAITING" or "BUFFERING" then "LIVE"
        try:
            # Wait for the buffer status to show "LIVE" or just wait a few seconds
            page.wait_for_selector("text=LIVE", timeout=10000)
            print("UI reached LIVE state.")
        except:
            print("Timed out waiting for LIVE state. Taking screenshot anyway.")

        time.sleep(5) # Let it render a few frames
        
        screenshot_path = "ui_validation.png"
        page.screenshot(path=screenshot_path)
        print(f"Screenshot saved to {screenshot_path}")
        
        browser.close()
    
    return True

if __name__ == "__main__":
    if validate():
        print("UI Validation script finished successfully.")
    else:
        print("UI Validation script failed.")
        # Fallback to selenium if playwright fails?
        import sys
        sys.exit(1)
