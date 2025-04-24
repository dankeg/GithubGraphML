from graph_tool.all import *
from typing import *

from matplotlib.patches import Wedge, ConnectionPatch
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np


abbrivation_map = {
    'AS': 'Assembly',
    'JS': 'JavaScript', 
    'PL': 'Pascal',
    'PL': 'Perl',
    'PY': 'Python',
    'RB': 'Ruby',
    'VB': 'VisualBasic',
}


def language_usage_distribution(language_usage: dict[Any, tuple[str]], output=None, start_radius=0.3, base_ring_width=0.8, label_threshold_angle=16, min_angle=0.8):
    # Count langauge combination usages
    combined_list = list(language_usage.values())
    data = Counter(combined_list)

    # Group by number of languages known
    layered_data = defaultdict(list)
    total = sum(data.values())

    for langs, count in data.items():
        layer = len(langs)
        label = '+'.join(sorted(langs)) + f': {count}'
        layered_data[layer].append((label, count))

    # Assign global angular spans
    segments = []
    for layer in sorted(layered_data.keys()):
        for label, count in layered_data[layer]:
            segments.append({
                'label': label,
                'count': count,
                'layer': layer
            })

    # --- Normalize angles: enforce minimum angle per wedge --- #
    # First, compute raw angles
    for seg in segments:
        seg['raw_angle'] = seg['count'] / total * 360

    # Apply minimum threshold
    fixed_angle_total = 0
    remaining_segments = []
    for seg in segments:
        if seg['raw_angle'] < min_angle:
            seg['final_angle'] = min_angle
            fixed_angle_total += min_angle
        else:
            remaining_segments.append(seg)

    # Redistribute remaining angle proportionally
    remaining_angle = 360 - fixed_angle_total
    remaining_total = sum(seg['count'] for seg in remaining_segments)
    for seg in remaining_segments:
        seg['final_angle'] = seg['count'] / remaining_total * remaining_angle

    # Assign theta1/theta2
    current_angle = 0
    for seg in segments:
        seg['theta1'] = current_angle
        seg['theta2'] = current_angle + seg['final_angle']
        current_angle += seg['final_angle']

    # --- Plot --- #
    _, ax = plt.subplots(figsize=(18, 18))
    ax.set_aspect('equal')
    colors = plt.cm.tab20.colors
    max_layer = max(layered_data.keys())

    # Compute log-scaled radii for each layer
    layer_radii = {layer: start_radius + np.log1p(layer - 1) * base_ring_width
                for layer in range(1, max_layer + 2)}  # +1 to get next outer edge

    for i, segment in enumerate(segments):
        layer = segment['layer']
        r0 = layer_radii[layer]
        r1 = layer_radii[layer + 1]

        wedge = Wedge(center=(0, 0),
                    r=r1,
                    theta1=segment['theta1'],
                    theta2=segment['theta2'],
                    width=r1 - r0,
                    facecolor=colors[i % len(colors)],
                    edgecolor='white')
        ax.add_patch(wedge)

        # Smart label placement
        theta_deg = (segment['theta1'] + segment['theta2']) / 2
        theta_rad = np.radians(theta_deg)
        mid_r = (r0 + r1) / 2
        x = mid_r * np.cos(theta_rad)
        y = mid_r * np.sin(theta_rad)
        angle_span = segment['theta2'] - segment['theta1']

        if angle_span < label_threshold_angle:
            # External label at chart edge
            label_radius = 2.1
            lx = label_radius * np.cos(theta_rad)
            ly = label_radius * np.sin(theta_rad)
            ha = 'left' if lx > 0 else 'right'

            ax.text(lx, ly, segment['label'], ha=ha, va='center', fontsize=5)
            line = ConnectionPatch(xyA=(lx, ly), coordsA=ax.transData,
                                xyB=(x, y), coordsB=ax.transData,
                                arrowstyle="-", color='gray', lw=0.5)
            ax.add_artist(line)
        else:
            # Internal label
            ax.text(x, y, segment['label'].replace(': ', ':\n'), ha='center', va='center', fontsize=7)

    # Abbriviations Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label=f"{abbr} = {full}",
                    markerfacecolor='gray', markersize=6)
        for abbr, full in abbrivation_map.items()]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.1),
                fontsize=7, frameon=False, ncol=2, title='Abbreviations')
    
    # Final layout
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.axis('off')
    plt.title('Hierarchical Language Chart\n(min wedge size = %.02f%%)' % (100 * (min_angle / 360)), fontsize=11)

    # Show output
    if output:
        plt.savefig(output, dpi=300, bbox_inches='tight')
    else:
        plt.show()
