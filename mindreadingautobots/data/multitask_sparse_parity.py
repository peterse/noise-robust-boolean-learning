#!/usr/bin/env python3
"""
Multitask Sparse Parity Dataset Generator

This script creates the "multitask sparse parity" dataset as described in the paper. 
The dataset consists of:
- Control bits (one-hot encoding of task number)
- Task bits
- Output bit (parity of a fixed subset of task bits determined by which control bit is active)
"""

import numpy as np
import pickle
import os
import sys

# Add the path to access sequence_generators
sys.path.append("../src")

from mindreadingautobots.sequence_generators import data_io


def generate_multitask_sparse_parity(n_tasks, n_bits, k, n_data, p_bitflip=0.0, seed=None):
    """
    Generate a multitask sparse parity dataset.
    
    Args:
        n_tasks: Number of subtasks (distinct versions of sparse parity)
        n_bits: Total length of task bits
        k: Size of the fixed subset for each parity calculation
        n_data: Number of data points to generate
        p_bitflip: Probability of flipping bits in the task bits (not control bits)
        seed: Random seed for reproducibility
        
    Returns:
        X: Array of shape (n_data, n_tasks + n_bits + 1) containing noiseless data:
           - First n_tasks bits are control bits (one-hot encoding of task)
           - Next n_bits are task bits
           - Last bit is the output (parity of relevant task bits)
        Z: Array of same shape as X but with noise in the task bits (if p_bitflip > 0)
        task_subsets: List of k indices for each task indicating which bits to use for parity
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize the dataset
    total_bits = n_tasks + n_bits + 1  # control bits + task bits + output bit
    X = np.zeros((n_data, total_bits), dtype=np.int32)
    
    # Generate random task subsets (each task uses a different subset of k indices)
    task_subsets = []
    for i in range(n_tasks):
        # Generate a random subset of k indices from the task bits
        subset = np.sort(np.random.choice(n_bits, k, replace=False))
        task_subsets.append(subset)
    
    # Generate data for each example
    for i in range(n_data):
        # Randomly select a task (which control bit to activate)
        active_task = np.random.randint(0, n_tasks)
        
        # Set the control bit (one-hot encoding)
        X[i, active_task] = 1
        
        # Generate random task bits
        task_bits = np.random.randint(0, 2, n_bits)
        X[i, n_tasks:n_tasks+n_bits] = task_bits
        
        # Compute the parity of the subset corresponding to the active task
        relevant_subset = task_subsets[active_task]
        relevant_bits = task_bits[relevant_subset]
        parity = np.sum(relevant_bits) % 2
        
        # Set the output bit
        X[i, -1] = parity
    
    # Apply noise to task bits if specified
    if p_bitflip > 0:
        # Create a copy of X
        Z = np.copy(X)
        
        # Generate noise mask for task bits only
        flips = np.random.binomial(1, p_bitflip, size=(n_data, n_bits))
        
        # Apply noise to task bits only
        Z[:, n_tasks:n_tasks+n_bits] = np.logical_xor(
            X[:, n_tasks:n_tasks+n_bits], 
            flips
        ).astype(np.int32)
        
        # Recompute output bit based on noisy task bits
        for i in range(n_data):
            active_task = np.argmax(Z[i, :n_tasks])
            relevant_subset = task_subsets[active_task]
            relevant_bits = Z[i, n_tasks:n_tasks+n_bits][relevant_subset]
            Z[i, -1] = np.sum(relevant_bits) % 2

        print(X, Z, task_subsets)            
        return X, Z, task_subsets

    else:
        return X, X, task_subsets



def verify_examples(X, task_subsets, n_tasks, num_examples=5):
    """Verify correctness of several examples in the dataset"""
    for i in range(min(num_examples, len(X))):
        example = X[i]
        
        # Determine which task is active
        active_task = np.argmax(example[:n_tasks])
        
        # Get the task bits
        task_bits = example[n_tasks:-1]
        
        # Get the subset for the active task
        relevant_subset = task_subsets[active_task]
        relevant_bits = task_bits[relevant_subset]
        
        # Compute expected parity
        expected_parity = np.sum(relevant_bits) % 2
        actual_parity = example[-1]
        
        # Display the example
        print(f"\nExample {i+1}:")
        print(f"Full string: {''.join([str(x) for x in example[:-1]])}")
        print(f"Control bits: {''.join([str(x) for x in example[:n_tasks]])}")
        print(f"Task bits: {''.join([str(x) for x in task_bits])}")
        print(f"Active task: {active_task+1}")
        print(f"Relevant subset indices: {relevant_subset}")
        print(f"Relevant bits: {relevant_bits}")
        print(f"Expected answer: {expected_parity}")
        print(f"Actual answer: {actual_parity}")
        print(f"Correct: {expected_parity == actual_parity}")


def main():
    # Dataset parameters - small values for testing
    n_tasks = 2  # Number of tasks
    task_bits_length = 8  # Length of the task bits portion
    k = 3  # Size of subset for parity calculation
    n_train = 10  # Number of training examples
    n_val = 4  # Number of validation examples
    p_bitflip = 0.1  # Probability of flipping bits in task bits (noise level)
    seed = 42  # Random seed

    # Generate the dataset
    print(f"Generating multitask_sparse_parity with {n_tasks} tasks, {task_bits_length} task bits, k={k}, noise={p_bitflip}")
    X, Z, task_subsets = generate_multitask_sparse_parity(
        n_tasks=n_tasks,
        n_bits=task_bits_length,
        k=k,
        n_data=n_train + n_val,
        p_bitflip=p_bitflip,
        seed=seed
    )

    # Print the task subsets (which bits are used for each task)
    for i, subset in enumerate(task_subsets):
        print(f"Task {i+1} uses bits {subset} for parity calculation")

    # Split into train and validation sets
    X_train = X[:n_train]
    X_val = X[n_train:]
    Z_train = Z[:n_train]
    Z_val = Z[n_train:]

    # Verify some examples
    print("\nVerifying noiseless training examples:")
    verify_examples(X_train, task_subsets, n_tasks, 2)
    
    if p_bitflip > 0:
        print("\nVerifying noisy training examples:")
        verify_examples(Z_train, task_subsets, n_tasks, 2)
    
    # Compare original and noisy examples
    if p_bitflip > 0:
        print("\nComparing original and noisy examples:")
        for i in range(2):  # Show 2 examples
            print(f"Example {i+1}:")
            print(f"Original task bits: {X_train[i, n_tasks:-1]}")
            print(f"Noisy task bits: {Z_train[i, n_tasks:-1]}")
            print(f"Bits flipped: {np.sum(X_train[i, n_tasks:-1] != Z_train[i, n_tasks:-1])}")
            print(f"Original answer: {X_train[i, -1]}")
            print(f"Noisy answer: {Z_train[i, -1]}\n")

    # Save the datasets
    os.makedirs('data/multitask_sparse_parity', exist_ok=True)
    data_io.save_numpy_as_dict(X_train, 'data/multitask_sparse_parity/noiseless_train.pkl')
    data_io.save_numpy_as_dict(X_val, 'data/multitask_sparse_parity/noiseless_val.pkl')
    if p_bitflip > 0:
        data_io.save_numpy_as_dict(Z_train, 'data/multitask_sparse_parity/train.pkl')
        data_io.save_numpy_as_dict(Z_val, 'data/multitask_sparse_parity/val.pkl')

    # Print saved data
    print("\nPrinting saved data:")
    for filename in ['noiseless_train.pkl', 'noiseless_val.pkl', 'train.pkl', 'val.pkl']:
        filepath = f'data/multitask_sparse_parity/{filename}'
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                print(f"\nContents of {filename}:")
                print(f"Number of examples: {len(data['line'])}")
                print("First example:")
                print(f"Line: {data['line'][0]}")
                print(f"Label: {data['label'][0]}")


if __name__ == "__main__":
    main()