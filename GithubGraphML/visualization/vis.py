from graph_tool.all import *


lang_color = {
    -1: '#A9A9A9',       # User - Grey
    0:  '#FF5733',       # Assembly - strong orange-red
    1:  '#F1C40F',       # JavaScript - vibrant yellow
    2:  '#8E44AD',       # Pascal - deep purple
    3:  '#1ABC9C',       # Perl - teal
    4:  '#3498DB',       # Python - blue
    5:  '#E74C3C',       # Ruby - ruby red
    6:  '#2ECC71',       # VisualBasic - green
}


def draw_bipartite(bipartite, **kwargs):
    pl_id_vp = bipartite.vp['programming_language_id']
    is_repo_vp = bipartite.vp['is_repository']
    coloring = pl_id_vp.t(lang_color.__getitem__, value_type='string')
    for v in bipartite.vertices(): 
        if not is_repo_vp[v]:
            coloring[v] = 'grey'
    args = {
        'output_size': (10192, 10192),
        'output': 'bipartite.png',
        'vertex_fill_color': coloring,
        **kwargs,
    }
    graph_draw(bipartite, **args)

def draw_combined(combined, **kwargs):
    coloring = combined.ep['programming_language_id'].t(lang_color.__getitem__, value_type='string')
    args = {
        'output_size': (10192, 10192),
        'output': 'combined.png',
        'edge_color': coloring,
        **kwargs,
    }
    graph_draw(combined, **args)
