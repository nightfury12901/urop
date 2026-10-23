import sys
import os

# Ensure workspace root and smartpark_eval are in python path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
workspace_dir = os.path.abspath(os.path.join(root_dir, ".."))
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)
