"""TF–target panel registry (P3 of the revision plan).

Every axis/WP refers to panels by name through :func:`load_panel`. New
panels can be added by dropping a loader into :mod:`fragility.panels.loaders`
and registering it in :data:`PANEL_REGISTRY` below.
"""

from .registry import (
    PANEL_REGISTRY,
    PanelSpec,
    load_panel,
    list_panels,
)

__all__ = [
    "PANEL_REGISTRY",
    "PanelSpec",
    "load_panel",
    "list_panels",
]
