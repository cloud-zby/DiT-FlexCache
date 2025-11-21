import sys
import yaml
import os

def main():
    try:
        import kvcache_pyo3
    except ImportError as e:
        print(f"Failed to import kvcache_pyo3: {e}")
        print("Make sure the PyO3 wheel is installed correctly.")
        sys.exit(1)

    os.environ["RUST_LOG"]="debug"
    
    # Load configuration
    try:
        with open("/gemini/space/jinyx5/repo/teleai_pipe/examples/flexcache/master.yaml", "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Failed to load config.yaml: {e}")
        sys.exit(1)

    print(f"Starting KV Master with config: {config}")

    # Run master with automatic lifecycle management
    try:
        result = kvcache_pyo3.run_master_blocking(config)
        if hasattr(result, 'error') and result.error() is not None:
            print(f"Failed to run master: {result.error()}")
            sys.exit(1)
        
        print("✅ Master completed successfully")
        
    except KeyboardInterrupt:
        print("\\n🛑 Master interrupted by user")
    except Exception as e:
        print(f"❌ Failed to run master: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()