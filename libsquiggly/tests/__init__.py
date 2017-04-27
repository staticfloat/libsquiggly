# Create our "figures" directory
import os
if not os.path.isdir("figures"):
    os.mkdir("figures")

# Don't pop open windows
from matplotlib.pyplot import switch_backend
switch_backend('Agg')
