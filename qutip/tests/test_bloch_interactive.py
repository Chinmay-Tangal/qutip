"""Tests for BlochInteractive."""

import pytest
import numpy as np
from qutip import basis, sigmax, sigmay, sigmaz
from qutip.core.gates import hadamard_transform
from qutip.bloch_interactive import BlochInteractive


class TestBlochInteractive:
    """Test the interactive Bloch sphere."""

    def test_creation(self):
        """BlochInteractive can be created with default state |0>."""
        b = BlochInteractive()
        assert b._current_state == basis(2, 0)

    def test_custom_figsize(self):
        """Custom figure size is stored."""
        b = BlochInteractive(figsize=[12, 10])
        assert b._figsize == [12, 10]

    def test_default_figsize(self):
        """Default figure size is [9, 8]."""
        b = BlochInteractive()
        assert b._figsize == [9, 8]

    def test_set_state(self):
        """Setting state updates current_state."""
        b = BlochInteractive()
        new_state = basis(2, 1)
        b._set_state(new_state)
        assert b._current_state == new_state

    def test_set_state_plus(self):
        """Setting state to |+> works."""
        b = BlochInteractive()
        plus = (basis(2, 0) + basis(2, 1)).unit()
        b._set_state(plus)
        fidelity = abs(b._current_state.overlap(plus)) ** 2
        assert abs(fidelity - 1.0) < 1e-10

    def test_apply_x_gate(self):
        """Applying X gate to |0> gives |1>."""
        b = BlochInteractive()
        b._apply_gate(sigmax())
        fidelity = abs(b._current_state.overlap(basis(2, 1))) ** 2
        assert abs(fidelity - 1.0) < 1e-10

    def test_apply_z_gate(self):
        """Applying Z gate to |0> gives |0>."""
        b = BlochInteractive()
        b._apply_gate(sigmaz())
        fidelity = abs(b._current_state.overlap(basis(2, 0))) ** 2
        assert abs(fidelity - 1.0) < 1e-10

    def test_apply_h_gate(self):
        """Applying H gate to |0> gives |+>."""
        b = BlochInteractive()
        b._apply_gate(hadamard_transform())
        plus = (basis(2, 0) + basis(2, 1)).unit()
        fidelity = abs(b._current_state.overlap(plus)) ** 2
        assert abs(fidelity - 1.0) < 1e-10

    def test_apply_x_gate_twice(self):
        """Applying X gate twice returns to original state."""
        b = BlochInteractive()
        b._apply_gate(sigmax())
        b._apply_gate(sigmax())
        fidelity = abs(b._current_state.overlap(basis(2, 0))) ** 2
        assert abs(fidelity - 1.0) < 1e-10

    def test_inherits_bloch(self):
        """BlochInteractive is a subclass of Bloch."""
        from qutip import Bloch
        b = BlochInteractive()
        assert isinstance(b, Bloch)
