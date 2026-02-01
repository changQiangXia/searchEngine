"""Web visualization components."""

from apps.web.visualizations.galaxy import generate_galaxy_plot
from apps.web.visualizations.attention import generate_attention_heatmap

__all__ = ["generate_galaxy_plot", "generate_attention_heatmap"]