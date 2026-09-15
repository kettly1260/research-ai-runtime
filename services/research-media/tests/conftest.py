import sys
import os

media_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
gateway_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ai-gateway"))

if gateway_dir in sys.path:
    sys.path.remove(gateway_dir)
if media_dir not in sys.path:
    sys.path.insert(0, media_dir)

# Clear any cached 'app' modules from other services
for mod_name in list(sys.modules.keys()):
    if mod_name == "app" or mod_name.startswith("app."):
        del sys.modules[mod_name]
