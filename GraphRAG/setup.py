#!/usr/bin/env python3
"""
Setup script for the Agentic RAG Platform
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        sys.exit(1)

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("❌ Python 3.9+ is required")
        sys.exit(1)
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")

def setup_environment():
    """Setup environment configuration"""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists():
        if env_example.exists():
            run_command("cp .env.example .env", "Creating .env file from template")
            print("⚠️  Please edit .env file with your API keys and configuration")
        else:
            print("❌ .env.example file not found")
            sys.exit(1)
    else:
        print("✅ .env file already exists")

def install_dependencies():
    """Install Python dependencies"""
    run_command("pip install -r requirements.txt", "Installing Python dependencies")

def create_directories():
    """Create necessary directories"""
    directories = [
        "data/documents",
        "data/vectors", 
        "data/feedback",
        "data/metrics",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Created necessary directories")

def check_redis():
    """Check if Redis is available"""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        print("✅ Redis is running and accessible")
        return True
    except Exception as e:
        print(f"⚠️  Redis not available: {e}")
        print("💡 Install and start Redis:")
        print("   Ubuntu/Debian: sudo apt install redis-server && sudo systemctl start redis")
        print("   macOS: brew install redis && brew services start redis")
        print("   Windows: Download from https://redis.io/download")
        return False

def validate_environment():
    """Validate environment configuration"""
    env_vars = [
        "OPENAI_API_KEY"
    ]
    
    missing_vars = []
    for var in env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("💡 Please set these variables in your .env file")
        return False
    
    print("✅ Environment variables validated")
    return True

def run_tests():
    """Run basic tests"""
    print("🧪 Running basic tests...")
    
    try:
        # Test imports
        from src.models.schemas import RAGState
        from src.graph.workflow import RAGWorkflow
        from src.ingestion.pipeline import DocumentIngestionPipeline
        print("✅ Core imports successful")
        
        # Test basic workflow creation
        workflow = RAGWorkflow()
        print("✅ Workflow initialization successful")
        
        return True
    except Exception as e:
        print(f"❌ Tests failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Setup Agentic RAG Platform")
    parser.add_argument("--skip-redis", action="store_true", help="Skip Redis check")
    parser.add_argument("--dev", action="store_true", help="Development setup")
    args = parser.parse_args()
    
    print("🚀 Setting up Agentic RAG Platform...")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Setup environment
    setup_environment()
    
    # Install dependencies
    install_dependencies()
    
    # Create directories
    create_directories()
    
    # Check Redis (unless skipped)
    if not args.skip_redis:
        check_redis()
    
    # Validate environment (only if not dev mode)
    if not args.dev:
        validate_environment()
    
    # Run basic tests
    run_tests()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit .env file with your API keys")
    print("2. Install and start Redis:")
    print("   - Ubuntu/Debian: sudo apt install redis-server && sudo systemctl start redis")
    print("   - macOS: brew install redis && brew services start redis")
    print("   - Windows: Download from https://redis.io/download")
    print("3. Start the application: python -m src.api.main")
    print("4. Visit http://localhost:8000/docs for API documentation")
    print("\nFor development mode with auto-reload:")
    print("python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000")

if __name__ == "__main__":
    main()
