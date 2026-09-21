import sys
import os

media_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
contracts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../packages/contracts/src"))

if media_dir not in sys.path:
    sys.path.insert(0, media_dir)
if contracts_dir not in sys.path:
    sys.path.insert(0, contracts_dir)
