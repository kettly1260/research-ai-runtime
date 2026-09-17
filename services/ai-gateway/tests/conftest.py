import sys
import os

gateway_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
contracts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../packages/contracts/src"))

if gateway_dir not in sys.path:
    sys.path.insert(0, gateway_dir)
if contracts_dir not in sys.path:
    sys.path.insert(0, contracts_dir)
