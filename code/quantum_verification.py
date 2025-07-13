"""
Quantum Verification Protocols
Comprehensive verification and validation for 3DCOM quantum implementations
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
from dataclasses import dataclass
from scipy.stats import kstest
import matplotlib.pyplot as plt

def hellinger_distance(p, q):
    """Calculate Hellinger distance between two probability distributions."""
    return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q))**2))


@dataclass
class VerificationResult:
    """Results from quantum verification."""
    test_name: str
    passed: bool
    fidelity: float
    hellinger_distance: float
    classical_time: float
    quantum_time: float
    speedup: float
    error_rate: float
    additional_metrics: Dict[str, Any]


class QuantumClassicalValidator:
    """
    Validates quantum implementations against classical references.
    """
    
    def __init__(self, tolerance: float = 0.05):
        """
        Initialize validator.
        
        Args:
            tolerance: Tolerance for quantum-classical differences
        """
        self.tolerance = tolerance
        self.test_results = []
    
    def validate_collatz_sequence(self, quantum_analyzer, classical_sequence: List[int]) -> VerificationResult:
        """
        Validate quantum Collatz analyzer against classical sequence.
        
        Args:
            quantum_analyzer: Quantum analyzer instance
            classical_sequence: Classical Collatz sequence
            
        Returns:
            Verification result
        """
        # Run quantum analysis
        quantum_start = self._get_time()
        quantum_steps = quantum_analyzer.quantum_behavior(len(classical_sequence))
        quantum_time = self._get_time() - quantum_start
        
        # Extract quantum sequence
        quantum_sequence = [step['value'] for step in quantum_steps]
        
        # Classical timing (simulated)
        classical_time = len(classical_sequence) * 1e-6  # 1μs per step
        
        # Calculate metrics
        fidelity = self._calculate_sequence_fidelity(quantum_sequence, classical_sequence)
        hellinger_dist = self._calculate_hellinger_distance(quantum_sequence, classical_sequence)
        speedup = classical_time / quantum_time if quantum_time > 0 else 1.0
        error_rate = 1.0 - fidelity
        
        # Determine if test passed
        passed = (fidelity >= (1.0 - self.tolerance) and 
                 hellinger_dist <= self.tolerance)
        
        result = VerificationResult(
            test_name="collatz_sequence_validation",
            passed=passed,
            fidelity=fidelity,
            hellinger_distance=hellinger_dist,
            classical_time=classical_time,
            quantum_time=quantum_time,
            speedup=speedup,
            error_rate=error_rate,
            additional_metrics={
                'sequence_length': len(classical_sequence),
                'quantum_entropy_mean': np.mean([s['entropy'] for s in quantum_steps]),
                'quantum_qubits_mean': np.mean([s['superposition'] for s in quantum_steps])
            }
        )
        
        self.test_results.append(result)
        return result
    
    def validate_3dcom_circuit(self, quantum_circuit, expected_properties: Dict) -> VerificationResult:
        """
        Validate 3DCOM quantum circuit properties.
        
        Args:
            quantum_circuit: Quantum circuit instance
            expected_properties: Expected circuit properties
            
        Returns:
            Verification result
        """
        # Analyze circuit properties
        actual_qubits = quantum_circuit.num_qubits
        actual_depth = quantum_circuit.depth()
        actual_gates = len(quantum_circuit.gates)
        
        expected_qubits = expected_properties.get('qubits', actual_qubits)
        expected_depth = expected_properties.get('depth', actual_depth)
        expected_gates = expected_properties.get('gates', actual_gates)
        
        # Calculate fidelity based on property matching
        qubit_fidelity = 1.0 - abs(actual_qubits - expected_qubits) / max(expected_qubits, 1)
        depth_fidelity = 1.0 - abs(actual_depth - expected_depth) / max(expected_depth, 1)
        gate_fidelity = 1.0 - abs(actual_gates - expected_gates) / max(expected_gates, 1)
        
        overall_fidelity = (qubit_fidelity + depth_fidelity + gate_fidelity) / 3
        
        # Hellinger distance based on gate distribution
        hellinger_dist = self._calculate_gate_distribution_distance(
            quantum_circuit.count_ops(), expected_properties.get('gate_distribution', {})
        )
        
        passed = (overall_fidelity >= (1.0 - self.tolerance) and 
                 hellinger_dist <= self.tolerance)
        
        result = VerificationResult(
            test_name="3dcom_circuit_validation",
            passed=passed,
            fidelity=overall_fidelity,
            hellinger_distance=hellinger_dist,
            classical_time=0.0,  # Not applicable
            quantum_time=0.0,   # Not applicable
            speedup=1.0,
            error_rate=1.0 - overall_fidelity,
            additional_metrics={
                'actual_qubits': actual_qubits,
                'actual_depth': actual_depth,
                'actual_gates': actual_gates,
                'expected_qubits': expected_qubits,
                'expected_depth': expected_depth,
                'expected_gates': expected_gates
            }
        )
        
        self.test_results.append(result)
        return result
    
    def _get_time(self) -> float:
        """Get current time in seconds."""
        import time
        return time.time()
    
    def _calculate_sequence_fidelity(self, seq1: List[int], seq2: List[int]) -> float:
        """Calculate fidelity between two sequences."""
        if not seq1 or not seq2:
            return 0.0
        
        min_len = min(len(seq1), len(seq2))
        matches = sum(1 for i in range(min_len) if seq1[i] == seq2[i])
        return matches / min_len
    
    def _calculate_hellinger_distance(self, seq1: List[int], seq2: List[int]) -> float:
        """Calculate Hellinger distance between sequence distributions."""
        if not seq1 or not seq2:
            return 1.0
        
        # Create probability distributions
        all_values = sorted(set(seq1 + seq2))
        
        dist1 = np.array([seq1.count(v) for v in all_values])
        dist2 = np.array([seq2.count(v) for v in all_values])
        
        dist1 = dist1 / np.sum(dist1)
        dist2 = dist2 / np.sum(dist2)
        
        return hellinger(dist1, dist2)
    
    def _calculate_gate_distribution_distance(self, actual_gates: Dict, expected_gates: Dict) -> float:
        """Calculate distance between gate distributions."""
        if not expected_gates:
            return 0.0
        
        all_gates = set(actual_gates.keys()) | set(expected_gates.keys())
        
        actual_dist = np.array([actual_gates.get(g, 0) for g in all_gates])
        expected_dist = np.array([expected_gates.get(g, 0) for g in all_gates])
        
        if np.sum(actual_dist) > 0:
            actual_dist = actual_dist / np.sum(actual_dist)
        if np.sum(expected_dist) > 0:
            expected_dist = expected_dist / np.sum(expected_dist)
        
        return hellinger(actual_dist, expected_dist)


class HardwareBenchmarkValidator:
    """
    Validates hardware optimization results and benchmarks.
    """
    
    def __init__(self):
        self.benchmark_results = {}
        self.platform_metrics = {
            'ibm_kyoto': {'max_qubits': 127, 'typical_fidelity': 0.85, 'gate_time': 476e-9},
            'rigetti_aspen': {'max_qubits': 80, 'typical_fidelity': 0.80, 'gate_time': 200e-9},
            'ionq_harmony': {'max_qubits': 11, 'typical_fidelity': 0.92, 'gate_time': 100e-9}
        }
    
    def benchmark_platform_optimization(self, platform: str, optimization_result) -> VerificationResult:
        """
        Benchmark optimization results for a specific platform.
        
        Args:
            platform: Platform name
            optimization_result: Optimization result to validate
            
        Returns:
            Verification result
        """
        platform_specs = self.platform_metrics.get(platform, {})
        
        # Validate optimization improvements
        depth_improvement = (optimization_result.original_depth - optimization_result.optimized_depth) / optimization_result.original_depth
        gate_improvement = (optimization_result.original_gates - optimization_result.optimized_gates) / optimization_result.original_gates
        
        # Expected improvements based on platform
        expected_improvements = {
            'ibm_kyoto': {'depth': 0.2, 'gates': 0.15, 'fidelity': 0.1},
            'rigetti_aspen': {'depth': 0.15, 'gates': 0.1, 'fidelity': 0.08},
            'ionq_harmony': {'depth': 0.3, 'gates': 0.2, 'fidelity': 0.12}
        }
        
        expected = expected_improvements.get(platform, {'depth': 0.1, 'gates': 0.1, 'fidelity': 0.05})
        
        # Calculate validation metrics
        depth_score = min(1.0, depth_improvement / expected['depth']) if expected['depth'] > 0 else 1.0
        gate_score = min(1.0, gate_improvement / expected['gates']) if expected['gates'] > 0 else 1.0
        fidelity_score = min(1.0, optimization_result.fidelity_improvement / expected['fidelity']) if expected['fidelity'] > 0 else 1.0
        
        overall_fidelity = (depth_score + gate_score + fidelity_score) / 3
        
        passed = overall_fidelity >= 0.8  # 80% of expected performance
        
        result = VerificationResult(
            test_name=f"{platform}_optimization_benchmark",
            passed=passed,
            fidelity=overall_fidelity,
            hellinger_distance=0.0,  # Not applicable
            classical_time=0.0,
            quantum_time=0.0,
            speedup=1.0,
            error_rate=1.0 - overall_fidelity,
            additional_metrics={
                'depth_improvement': depth_improvement,
                'gate_improvement': gate_improvement,
                'fidelity_improvement': optimization_result.fidelity_improvement,
                'depth_score': depth_score,
                'gate_score': gate_score,
                'fidelity_score': fidelity_score,
                'platform_specs': platform_specs
            }
        )
        
        self.benchmark_results[platform] = result
        return result
    
    def cross_platform_comparison(self, optimization_results: Dict) -> Dict[str, Any]:
        """
        Compare optimization results across platforms.
        
        Args:
            optimization_results: Dictionary of optimization results by platform
            
        Returns:
            Comparison analysis
        """
        if not optimization_results:
            return {}
        
        # Find best platform for each metric
        best_fidelity = max(optimization_results.keys(), 
                           key=lambda p: optimization_results[p].fidelity_improvement)
        best_depth = min(optimization_results.keys(), 
                        key=lambda p: optimization_results[p].optimized_depth)
        best_gates = min(optimization_results.keys(), 
                        key=lambda p: optimization_results[p].optimized_gates)
        
        # Calculate platform rankings
        platforms = list(optimization_results.keys())
        fidelity_ranking = sorted(platforms, 
                                key=lambda p: optimization_results[p].fidelity_improvement, 
                                reverse=True)
        depth_ranking = sorted(platforms, 
                             key=lambda p: optimization_results[p].optimized_depth)
        
        comparison = {
            'best_fidelity_platform': best_fidelity,
            'best_depth_platform': best_depth,
            'best_gates_platform': best_gates,
            'fidelity_ranking': fidelity_ranking,
            'depth_ranking': depth_ranking,
            'platform_scores': {
                platform: {
                    'fidelity_improvement': result.fidelity_improvement,
                    'depth_reduction': (result.original_depth - result.optimized_depth) / result.original_depth,
                    'gate_reduction': (result.original_gates - result.optimized_gates) / result.original_gates
                }
                for platform, result in optimization_results.items()
            }
        }
        
        return comparison


class ComprehensiveTestSuite:
    """
    Comprehensive test suite for 3DCOM quantum implementations.
    """
    
    def __init__(self):
        self.validator = QuantumClassicalValidator()
        self.benchmark_validator = HardwareBenchmarkValidator()
        self.test_cases = self._generate_test_cases()
    
    def _generate_test_cases(self) -> List[Dict]:
        """Generate comprehensive test cases."""
        return [
            {'name': 'small_numbers', 'numbers': [3, 7, 15], 'max_qubits': 4},
            {'name': 'medium_numbers', 'numbers': [27, 63], 'max_qubits': 6},
            {'name': 'large_numbers', 'numbers': [127, 255], 'max_qubits': 8},
            {'name': 'edge_cases', 'numbers': [1, 2, 4, 8, 16], 'max_qubits': 5},
            {'name': 'prime_numbers', 'numbers': [7, 11, 13, 17, 19], 'max_qubits': 5}
        ]
    
    def run_full_test_suite(self) -> Dict[str, List[VerificationResult]]:
        """
        Run the complete test suite.
        
        Returns:
            Dictionary of test results by category
        """
        results = {}
        
        print("Running Comprehensive 3DCOM Quantum Test Suite")
        print("=" * 60)
        
        for test_case in self.test_cases:
            print(f"\nRunning {test_case['name']} tests...")
            case_results = []
            
            for number in test_case['numbers']:
                # Test quantum Collatz analyzer
                result = self._test_quantum_collatz(number)
                case_results.append(result)
                
                # Test 3DCOM circuit
                result = self._test_3dcom_circuit(number, test_case['max_qubits'])
                case_results.append(result)
            
            results[test_case['name']] = case_results
            
            # Print summary for this test case
            passed_tests = sum(1 for r in case_results if r.passed)
            print(f"  {passed_tests}/{len(case_results)} tests passed")
        
        return results
    
    def _test_quantum_collatz(self, number: int) -> VerificationResult:
        """Test quantum Collatz analyzer for a specific number."""
        # Import here to avoid circular imports
        from ..src.quantum_collatz import QuantumCollatzAnalyzer
        
        # Generate classical reference
        classical_sequence = self._generate_classical_collatz(number)
        
        # Test quantum implementation
        quantum_analyzer = QuantumCollatzAnalyzer(number)
        
        return self.validator.validate_collatz_sequence(quantum_analyzer, classical_sequence)
    
    def _test_3dcom_circuit(self, number: int, max_qubits: int) -> VerificationResult:
        """Test 3DCOM quantum circuit for a specific number."""
        # Import here to avoid circular imports
        from ..src.quantum_3dcom import OptimizedQuantum3DCOM
        
        # Create quantum circuit
        q3dcom = OptimizedQuantum3DCOM(number)
        circuit = q3dcom.build_quantum_circuit()
        
        # Expected properties based on number analysis
        expected_properties = {
            'qubits': min(max_qubits, len(str(number)) + 2),  # Rough estimate
            'depth': 10,  # Expected depth
            'gates': 15,  # Expected gate count
            'gate_distribution': {'ry': 3, 'cx': 2}
        }
        
        return self.validator.validate_3dcom_circuit(circuit, expected_properties)
    
    def _generate_classical_collatz(self, n: int, max_steps: int = 100) -> List[int]:
        """Generate classical Collatz sequence."""
        sequence = []
        current = n
        
        for _ in range(max_steps):
            if current == 1:
                break
            
            sequence.append(current)
            
            if current % 2 == 0:
                current = current // 2
            else:
                current = 3 * current + 1
        
        return sequence
    
    def generate_test_report(self, results: Dict[str, List[VerificationResult]]) -> str:
        """
        Generate comprehensive test report.
        
        Args:
            results: Test results by category
            
        Returns:
            Formatted test report
        """
        report = []
        report.append("3DCOM Quantum Test Suite Report")
        report.append("=" * 50)
        report.append("")
        
        total_tests = 0
        total_passed = 0
        
        for category, test_results in results.items():
            report.append(f"Category: {category.upper()}")
            report.append("-" * 30)
            
            category_passed = 0
            for result in test_results:
                total_tests += 1
                if result.passed:
                    total_passed += 1
                    category_passed += 1
                
                status = "PASS" if result.passed else "FAIL"
                report.append(f"  {result.test_name}: {status}")
                report.append(f"    Fidelity: {result.fidelity:.3f}")
                report.append(f"    Error Rate: {result.error_rate:.3f}")
                if result.speedup > 1:
                    report.append(f"    Speedup: {result.speedup:.1f}x")
                report.append("")
            
            report.append(f"Category Summary: {category_passed}/{len(test_results)} passed")
            report.append("")
        
        report.append("OVERALL SUMMARY")
        report.append("-" * 20)
        report.append(f"Total Tests: {total_tests}")
        report.append(f"Passed: {total_passed}")
        report.append(f"Failed: {total_tests - total_passed}")
        report.append(f"Success Rate: {total_passed/total_tests:.1%}")
        
        return "\n".join(report)


def demo_verification():
    """Demonstration of verification capabilities."""
    print("3DCOM Quantum Verification Demo")
    print("=" * 40)
    
    # Run test suite
    test_suite = ComprehensiveTestSuite()
    results = test_suite.run_full_test_suite()
    
    # Generate and print report
    report = test_suite.generate_test_report(results)
    print("\n" + report)
    
    # Save report to file
    with open('/home/ubuntu/3dcom_quantum/verification/test_report.txt', 'w') as f:
        f.write(report)
    
    print(f"\nDetailed test report saved to: test_report.txt")


if __name__ == "__main__":
    demo_verification()

