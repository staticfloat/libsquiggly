import os, sys

# Loop over every directory beneath us that has a __init__.py inside of it
root_dir = os.path.dirname(__file__)
for module_dir in os.listdir(root_dir):
    if not os.path.isdir(os.path.join(root_dir, module_dir)) or not os.path.isfile(os.path.join(root_dir, module_dir, '__init__.py')):
        continue

    # Import this module into the current namespace
    module = __import__(module_dir, locals(), globals())

# Cleanup thse dangling bindings
del module_dir
del root_dir
del os
del sys