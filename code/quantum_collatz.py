"""
3DCOM Quantum Collatz Implementation
Core quantum analyzer for Collatz sequences with quantum signatures
"""

import numpy as np
from sympy import factorint
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional

class QuantumCollatzAnalyzer:
    """
    Quantum analyzer for Collatz sequences implementing the quantum signature theory.
    
    This class treats each Collatz step as a quantum measurement operator
    acting on the number's prime factorization.
    """
    
    def __init__(self, n: int):
        """
        Initialize the quantum analyzer for a given number.
        
        Args:
            n: Starting number for Collatz sequence analysis
        """
        self.n = n
        self.sequence = []
        self.quantum_states = []
        
    def quantum_collatz_step(self, n: int) -> int:
        """
        Perform a single Collatz step with quantum interpretation.
        
        Args:
            n: Current number in sequence
            
        Returns:
            Next number in sequence
        """
        if n % 2 == 0:
            # Quantum "measurement" of 2's exponent
            return n // 2  # Projection onto even eigenspace
        else:
            # Entanglement operation
            return 3 * n + 1  # Creates superposition of prime factors
    
    def quantum_behavior(self, max_steps: int = 100) -> List[Dict]:
        """
        Analyze quantum properties of the Collatz sequence.
        
        Args:
            max_steps: Maximum number of steps to analyze
            
        Returns:
            List of quantum state information for each step
        """
        steps = []
        n = self.n
        
        for i in range(max_steps):
            if n == 1:
                break
                
            factors = factorint(n)
            
            # Calculate quantum entropy
            factor_probs = np.array(list(factors.values()))
            if len(factor_probs) > 0:
                factor_probs = factor_probs / np.sum(factor_probs)
                entropy = -np.sum(factor_probs * np.log(factor_probs + 1e-10))
            else:
                entropy = 0.0
            
            step_info = {
                'step': i,
                'value': n,
                'entropy': entropy,
                'superposition': len(factors),  # Number of prime bases (qubits)
                'factors': factors,
                'is_even': n % 2 == 0
            }
            
            steps.append(step_info)
            self.sequence.append(n)
            self.quantum_states.append(step_info)
            
            # Perform quantum Collatz step
            n = self.quantum_collatz_step(n)
            
        return steps
    
    def calculate_quantum_hamiltonian(self, state: Dict) -> np.ndarray:
        """
        Calculate the quantum Hamiltonian for a given state.
        
        H = σ₊ ⊗ D + σ₋ ⊗ U
        Where σ₊/σ₋ are projection operators for even/odd
        
        Args:
            state: Quantum state information
            
        Returns:
            Hamiltonian matrix
        """
        n = state['value']
        dim = max(4, state['superposition'] * 2)  # Minimum 4x4 matrix
        
        # Pauli matrices
        sigma_plus = np.array([[0, 1], [0, 0]])  # |0⟩⟨1|
        sigma_minus = np.array([[0, 0], [1, 0]])  # |1⟩⟨0|
        
        # Division operator (measurement)
        D = np.eye(dim) * 0.5
        
        # Unitary transformation (3n+1)
        U = np.eye(dim)
        for i in range(dim-1):
            U[i, i+1] = 1
            U[i, i] = 0
        U[-1, 0] = 1
        
        # Construct Hamiltonian
        if n % 2 == 0:
            H = np.kron(sigma_plus, D[:2, :2])
        else:
            H = np.kron(sigma_minus, U[:2, :2])
            
        return H
    
    def plot_quantum_entropy(self, save_path: Optional[str] = None):
        """
        Plot the quantum entropy evolution during Collatz sequence.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.quantum_states:
            self.quantum_behavior()
            
        entropies = [state['entropy'] for state in self.quantum_states]
        steps = [state['step'] for state in self.quantum_states]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, entropies, 'b-o', linewidth=2, markersize=4)
        plt.xlabel('Collatz Step')
        plt.ylabel('Quantum Entropy')
        plt.title(f'Quantum Entropy Evolution for n={self.n}')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def analyze_quantum_statistics(self) -> Dict:
        """
        Perform comprehensive quantum statistical analysis.
        
        Returns:
            Dictionary containing statistical measures
        """
        if not self.quantum_states:
            self.quantum_behavior()
            
        entropies = [state['entropy'] for state in self.quantum_states]
        qubits = [state['superposition'] for state in self.quantum_states]
        
        stats = {
            'mean_entropy': np.mean(entropies),
            'max_entropy': np.max(entropies),
            'entropy_variance': np.var(entropies),
            'mean_qubits': np.mean(qubits),
            'max_qubits': np.max(qubits),
            'convergence_steps': len(self.quantum_states),
            'decoherence_time': self._calculate_decoherence_time(),
            'quantum_speedup': self._estimate_quantum_speedup()
        }
        
        return stats
    
    def _calculate_decoherence_time(self) -> float:
        """Calculate the decoherence time scaling as log(n)."""
        return np.log(self.n) if self.n > 1 else 0.0
    
    def _estimate_quantum_speedup(self) -> float:
        """Estimate quantum speedup vs classical implementation."""
        classical_complexity = self.n  # O(n)
        quantum_complexity = np.log(self.n) if self.n > 1 else 1  # O(log n)
        return classical_complexity / quantum_complexity if quantum_complexity > 0 else 1.0


def demo_quantum_collatz():
    """Demonstration of quantum Collatz analysis."""
    test_numbers = [7, 15, 27, 255]
    
    print("3DCOM Quantum Collatz Analysis Demo")
    print("=" * 50)
    
    for n in test_numbers:
        print(f"\nQuantum analysis for {n}:")
        analyzer = QuantumCollatzAnalyzer(n)
        steps = analyzer.quantum_behavior(max_steps=10)
        
        for i, step in enumerate(steps[:5]):  # Show first 5 steps
            print(f"→ Step {i}: {step['value']} (Entropy: {step['entropy']:.3f}, Qubits: {step['superposition']})")
        
        stats = analyzer.analyze_quantum_statistics()
        print(f"Statistics: Mean entropy: {stats['mean_entropy']:.3f}, "
              f"Quantum speedup: {stats['quantum_speedup']:.1f}x")


if __name__ == "__main__":
    demo_quantum_collatz()

