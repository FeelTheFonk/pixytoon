import sys

def run_server():
    try:
        from sddj.server import main
        main()
    except KeyboardInterrupt:
        print("\n  [SDDj] Server stopped by user.", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"\n  [ERROR] FATAL CRASH: {e}", file=sys.stderr)
        print("  [INFO] Please check models and port availability.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    run_server()
