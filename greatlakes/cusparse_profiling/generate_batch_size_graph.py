import matplotlib.pyplot as plt
import numpy as np

# Data from
# SAE_testing/cusparse_profiling/36947915_search_batch_size.out
batch_sizes = np.array([16, 32, 64, 128, 256])
dense_decoder = np.array([6.199264049530029, 12.333727836608887, 24.547199249267578, 50.87948989868164, 101.69142150878906])
sparse_decoder = np.array([9.088000297546387, 18.12272071838379, 37.61052703857422, 77.64073944091797, 155.8804473876953])

# ------------------ FIT LINEAR PROJECTION ------------------
# Linear fit on batch size vs decode time
dense_fit = np.polyfit(batch_sizes, dense_decoder, 1)   # y = m*x + b
sparse_fit = np.polyfit(batch_sizes, sparse_decoder, 1)

# Create batch sizes for projection (up to 1024, beyond memory limit)
batch_proj = np.linspace(16, 1024, 100)

dense_proj = np.polyval(dense_fit, batch_proj)
sparse_proj = np.polyval(sparse_fit, batch_proj)

# ------------------ PLOT ------------------
plt.figure(figsize=(8,5))
plt.plot(batch_sizes, dense_decoder, 'o', label="Dense Decoder (measured)")
plt.plot(batch_sizes, sparse_decoder, 'o', label="Sparse Decoder (measured)")

plt.plot(batch_proj, dense_proj, '--', label="Dense Decoder (projected)")
plt.plot(batch_proj, sparse_proj, '--', label="Sparse Decoder (projected)")

plt.xlabel("Batch Size")
plt.ylabel("Decode Time (ms)")
plt.title("Decode Time vs Batch Size for Two Decoders")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("decode_vs_batch_size.png")