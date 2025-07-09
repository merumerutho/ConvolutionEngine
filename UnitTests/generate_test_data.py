import numpy as np
from scipy import signal
import argparse

def generate_white_noise(num_samples):
    return np.random.randn(num_samples).astype(np.float32)

def generate_sine_wave(num_samples, frequency, sample_rate):
    t = np.linspace(0., 1., num_samples)
    return np.sin(2. * np.pi * frequency * t).astype(np.float32)

def generate_input_signal(num_samples, sample_rate):
    noise = generate_white_noise(num_samples)
    sine1 = generate_sine_wave(num_samples, 440.0, sample_rate)
    sine2 = generate_sine_wave(num_samples, 880.0, sample_rate)
    return (noise * 0.1 + sine1 * 0.45 + sine2 * 0.45)

def generate_dense_ir(num_taps):
    return np.random.randn(num_taps).astype(np.float32)

def generate_sparse_ir(num_taps, ir_length):
    positions = np.random.randint(0, ir_length, num_taps, dtype=np.uint64)
    values = np.random.randn(num_taps).astype(np.float32)
    ir = np.zeros(ir_length, dtype=np.float32)
    for pos, val in zip(positions, values):
        ir[pos] += val
    return ir, positions, values

def generate_velvet_ir(num_pos_taps, num_neg_taps, ir_length):
    pos_positions = np.random.randint(0, ir_length, num_pos_taps, dtype=np.uint64)
    neg_positions = np.random.randint(0, ir_length, num_neg_taps, dtype=np.uint64)
    ir = np.zeros(ir_length, dtype=np.float32)
    for pos in pos_positions:
        ir[pos] += 1.0
    for pos in neg_positions:
        ir[pos] -= 1.0
    return ir, pos_positions, neg_positions

def write_cpp_header(f, name, data, dtype="float"):
    f.write(f"const {dtype} {name}[] = {{")
    f.write(", ".join(map(str, data)))
    f.write("};\n")
    f.write(f"const size_t {name}_size = {len(data)};\n\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", default="src/generated_test_data.h")
    args = parser.parse_args()

    np.random.seed(42)
    sample_rate = 48000
    input_signal_length = 4096
    dense_ir_length = 256
    sparse_ir_length = 256
    velvet_ir_length = 256
    num_sparse_taps = 100
    num_velvet_pos_taps = 50
    num_velvet_neg_taps = 50

    input_signal = generate_input_signal(input_signal_length, sample_rate)
    dense_ir = generate_dense_ir(dense_ir_length)
    sparse_ir, sparse_positions, sparse_values = generate_sparse_ir(num_sparse_taps, sparse_ir_length)
    velvet_ir, velvet_pos_positions, velvet_neg_positions = generate_velvet_ir(num_velvet_pos_taps, num_velvet_neg_taps, velvet_ir_length)

    expected_output_dense = signal.convolve(input_signal, dense_ir, mode='full')
    expected_output_sparse = signal.convolve(input_signal, sparse_ir, mode='full')
    expected_output_velvet = signal.convolve(input_signal, velvet_ir, mode='full')

    with open(args.output_file, "w") as f:
        f.write("#pragma once\n\n")
        f.write("#include <cstddef>\n\n")

        write_cpp_header(f, "input_signal", input_signal)
        write_cpp_header(f, "dense_ir", dense_ir)
        write_cpp_header(f, "sparse_ir_positions", sparse_positions, dtype="size_t")
        write_cpp_header(f, "sparse_ir_values", sparse_values)
        write_cpp_header(f, "velvet_ir_pos_positions", velvet_pos_positions, dtype="size_t")
        write_cpp_header(f, "velvet_ir_neg_positions", velvet_neg_positions, dtype="size_t")
        write_cpp_header(f, "expected_output_dense", expected_output_dense)
        write_cpp_header(f, "expected_output_sparse", expected_output_sparse)
        write_cpp_header(f, "expected_output_velvet", expected_output_velvet)

if __name__ == "__main__":
    main()