import sys
try:
    from retriever import table_lookup, get_table_registry
    print("retriever OK")
except Exception as e:
    print(f"RETRIEVER ERROR: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

try:
    from fastapi_app import app
    print("fastapi_app OK")
except Exception as e:
    print(f"FASTAPI ERROR: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

print("ALL IMPORTS OK")
