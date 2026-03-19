import numpy as np
import matplotlib.pyplot as plt
from scipy.special import genlaguerre
import sys
import os

# Add project root to sys.path to import paper_style
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from experiments.configs.paper_style import apply_paper_style, SINGLE_COLUMN_PT

def wigner_fock(n, q, p):
    """
    Wigner function for a Fock state |n>.
    W_n(q, p) = (1/pi) * (-1)^n * exp(-(q^2 + p^2)) * L_n(2 * (q^2 + p^2))
    """
    rho_sq = q**2 + p**2
    L_n = genlaguerre(n, 0)
    return (1.0 / np.pi) * ((-1)**n) * np.exp(-rho_sq) * L_n(2 * rho_sq)

# Apply paper style
# width = 1/4 * SINGLE_COLUMN_PT
# We use panel_aspect=1.0 for phase space plot (square)
figsize = apply_paper_style(width_pt=0.25 * SINGLE_COLUMN_PT, panel_aspect=1.0)

# Setup the grid
x = np.linspace(-5, 5, 400)
y = np.linspace(-5, 5, 400)
Q, P = np.meshgrid(x, y)

# Calculate Wigner function for n=5
n = 5
W = wigner_fock(n, Q, P)

# Plotting
fig, ax = plt.subplots(figsize=figsize)
# Use a high number of levels for smooth gradients
cp = ax.contourf(Q, P, W, levels=100, cmap='RdBu_r')

# Remove colorbar for 1/4 column plot as it might be too crowded, 
# or keep it if requested. The user didn't specify. 
# Usually at 1/4 column, we might skip labels or simplify.
# But I'll keep the basics.
ax.set_title(f'Fock |{n}>')
ax.set_xlabel('q')
ax.set_ylabel('p')

# Ensure aspect ratio is equal for phase space
ax.set_aspect('equal')

# Save as SVG
output_path = 'experiments/exampleplot/fock_5_wigner.svg'
plt.savefig(output_path, format='svg')
print(f"Successfully saved Wigner function plot with paper style to {output_path}")
plt.close()
