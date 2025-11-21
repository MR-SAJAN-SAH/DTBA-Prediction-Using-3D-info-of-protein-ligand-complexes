#!/usr/bin/env python3
"""
Run script for the BioAffinity AI Web Application
"""

import os
import sys
from app import app, load_models

def main():
    # Check if model exists
    model_path = "checkpoints/best_model.pth"
    if not os.path.exists(model_path):
        print("❌ Error: Model file 'checkpoints/best_model.pth' not found!")
        print("Please make sure you have trained the model and the best_model.pth file is in the checkpoints directory.")
        sys.exit(1)
    
    # Load models
    print("🚀 Starting BioAffinity AI Web Application...")
    print("📦 Loading ensemble models...")
    
    try:
        load_models()
        print("✅ All models loaded successfully!")
        print("🌐 Starting web server on http://localhost:5000")
        print("📖 Open your browser and navigate to the above URL to use the application")
        print("⏹️  Press Ctrl+C to stop the server")
        
        # Run Flask app
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()