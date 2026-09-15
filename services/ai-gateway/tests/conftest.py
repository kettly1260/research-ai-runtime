import sys
import os

gateway_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
media_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../research-media"))

if media_dir in sys.path:
    sys.path.remove(media_dir)
if gateway_dir not in sys.path:
    sys.path.insert(0, gateway_dir)

# Clear any cached 'app' modules from other services
for mod_name in list(sys.modules.keys()):
    if mod_name == "app" or mod_name.startswith("app."):
        del sys.modules[mod_name]
