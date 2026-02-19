"""
Interactive Bloch Sphere visualization with Matplotlib widgets.

This module extends the Bloch sphere with interactive controls
for view angle, quantum states, and common gates.
"""

__all__ = ['BlochInteractive']

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from .bloch import Bloch
from .core.operators import sigmax, sigmay, sigmaz
from .core.states import basis
from .core.gates import hadamard_transform


class BlochInteractive(Bloch):
    """
    Interactive Bloch sphere with sliders and buttons.

    Extends the Bloch class with Matplotlib widgets for
    interactive exploration of qubit states and gates.

    Parameters
    ----------
    figsize : list, optional
        Figure size [width, height]. Default: [9, 8].

    Examples
    --------
    >>> b = BlochInteractive()
    >>> b.show()
    """

    def __init__(self, figsize=None, **kwargs):
        self._figsize = figsize or [9, 8]
        self._current_state = basis(2, 0)
        super().__init__(**kwargs)

    def show(self):
        """Launch the interactive Bloch sphere."""
        self.fig = plt.figure(figsize=self._figsize)

        # Main sphere
        self.axes = self.fig.add_axes(
            [0.1, 0.3, 0.8, 0.65], projection='3d'
        )

        # View angle sliders
        ax_azim = self.fig.add_axes([0.2, 0.2, 0.6, 0.03])
        ax_elev = self.fig.add_axes([0.2, 0.15, 0.6, 0.03])
        self._sl_azim = Slider(ax_azim, 'Azimuth', -180, 180, valinit=-60)
        self._sl_elev = Slider(ax_elev, 'Elevation', -90, 90, valinit=30)
        self._sl_azim.on_changed(self._on_view_change)
        self._sl_elev.on_changed(self._on_view_change)

        # State buttons
        states = {
            '|0⟩': basis(2, 0),
            '|1⟩': basis(2, 1),
            '|+⟩': (basis(2, 0) + basis(2, 1)).unit(),
            '|-⟩': (basis(2, 0) - basis(2, 1)).unit(),
            '|+i⟩': (basis(2, 0) + 1j * basis(2, 1)).unit(),
            '|-i⟩': (basis(2, 0) - 1j * basis(2, 1)).unit(),
        }
        self._state_buttons = []
        for i, (label, state) in enumerate(states.items()):
            ax = self.fig.add_axes([0.05 + i * 0.1, 0.08, 0.08, 0.04])
            btn = Button(ax, label)
            btn.on_clicked(lambda _, s=state: self._set_state(s))
            self._state_buttons.append(btn)

        # Gate buttons
        gates = {
            'X': sigmax(),
            'Y': sigmay(),
            'Z': sigmaz(),
            'H': hadamard_transform(),
        }
        self._gate_buttons = []
        for i, (label, gate) in enumerate(gates.items()):
            ax = self.fig.add_axes([0.15 + i * 0.1, 0.02, 0.08, 0.04])
            btn = Button(ax, label)
            btn.on_clicked(lambda _, g=gate: self._apply_gate(g))
            self._gate_buttons.append(btn)

        self._refresh()
        plt.show()

    def _set_state(self, state):
        """Set current state and refresh."""
        self._current_state = state
        self._refresh()

    def _apply_gate(self, gate):
        """Apply gate to current state and refresh."""
        self._current_state = (gate * self._current_state).unit()
        self._refresh()

    def _on_view_change(self, _):
        """Update view angles from sliders."""
        self.view = [self._sl_azim.val, self._sl_elev.val]
        self.axes.view_init(
            elev=self._sl_elev.val,
            azim=self._sl_azim.val
        )
        self.fig.canvas.draw_idle()

    def _refresh(self):
        """Clear and redraw the sphere."""
        self.clear()
        self.add_states(self._current_state)
        if self.fig is not None:
            self.make_sphere()
            if hasattr(self, '_sl_elev'):
                self.axes.view_init(
                    elev=self._sl_elev.val,
                    azim=self._sl_azim.val
                )
            self.fig.canvas.draw_idle()
