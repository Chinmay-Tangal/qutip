"""Tests for BlochInteractive."""

import pytest
import numpy as np
from qutip import basis, sigmax
from qutip.bloch_interactive import BlochInteractive


class TestBlochInteractive:
    """Test the interactive Bloch sphere."""

    def test_creation(self):
        """BlochInteractive can be created."""
        b = BlochInteractive()
        assert b._current_state == basis(2, 0)

    def test_set_state(self):
        """Setting state updates current_state."""
        b = BlochInteractive()
        new_state = basis(2, 1)
        b._set_state(new_state)
        assert b._current_state == new_state

    def test_apply_gate(self):
        """Applying X gate to |0⟩ gives |1⟩."""
        b = BlochInteractive()
        b._set_state(basis(2, 0))
        b._apply_gate(sigmax())
        fidelity = abs(b._current_state.overlap(basis(2, 1))) ** 2
        assert abs(fidelity - 1.0) < 1e-10