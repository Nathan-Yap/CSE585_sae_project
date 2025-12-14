import matplotlib.pyplot as plt
import numpy as np

# Data from
# SAE_testing/cusparse_profiling/36948311_search_sparsity.out
sparsity = [
    99.5020, 99.5274, 99.5519, 99.5772, 99.6019,
    99.6260, 99.6501, 99.6758, 99.7014, 99.7258,
    99.7507, 99.7756, 99.8005, 99.8257, 99.8503,
    99.8751, 99.9001, 99.9252, 99.9499, 99.9750
]

normal_decode = [32.36675262451172,0.574944019317627,0.5131840109825134,0.5066559910774231,0.5030720233917236,0.5044800043106079,0.5014079809188843,0.5041599869728088,0.5029439926147461,0.5030720233917236,0.5024319887161255,0.5026879906654358,0.502240002155304,0.5035200119018555,0.5017279982566833,0.523967981338501,0.5021119713783264,0.5015040040016174,0.5021439790725708,0.5027199983596802]

sparse_decode = [71.85001373291016,1.3281279802322388,0.9533439874649048,0.9205759763717651,0.9041919708251953,0.8868160247802734,0.8704320192337036,0.8509119749069214,0.8294399976730347,0.8171200156211853,0.8038719892501831,0.7833920121192932,0.7567359805107117,0.7434560060501099,0.7372480034828186,0.7167680263519287,0.7239360213279724,0.7178559899330139,0.6880959868431091,0.6778879761695862]

sparsity = np.array(sparsity[2:])
dense_decoder = np.array(normal_decode[2:])
sparse_decoder = np.array(sparse_decode[2:])

# ------------------ FIT LINEAR PROJECTION ------------------
# Linear fit on batch size vs decode time
dense_fit = np.polyfit(sparsity, dense_decoder, 1)   # y = m*x + b
sparse_fit = np.polyfit(sparsity, sparse_decoder, 1)

# Create batch sizes for projection (up to 1024, beyond memory limit)
batch_proj = np.linspace(99.25, 100, 100)

dense_proj = np.polyval(dense_fit, batch_proj)
sparse_proj = np.polyval(sparse_fit, batch_proj)

m_d, b_d = dense_fit
m_s, b_s = sparse_fit

# Solve for intersection: m_d*x + b_d = m_s*x + b_s
x_intersect = (b_s - b_d) / (m_d - m_s)
y_intersect = m_d * x_intersect + b_d

print("Intersection sparsity:", x_intersect)
print("Intersection decode time:", y_intersect)

# ------------------ PLOT ------------------
plt.figure(figsize=(8,5))
plt.plot(sparsity, dense_decoder, 'o', label="Dense Decoder (measured)")
plt.plot(sparsity, sparse_decoder, 'o', label="Sparse Decoder (measured)")

plt.plot(batch_proj, dense_proj, '--', label="Dense Decoder (projected)")
plt.plot(batch_proj, sparse_proj, '--', label="Sparse Decoder (projected)")

# plt.plot(
#     x_intersect, y_intersect,
#     'ro', markersize=8, label=f"Intersection\n({x_intersect:.4f}%, {y_intersect:.2f} ms)"
# )
# plt.annotate(
#     f"({x_intersect:.4f}%, {y_intersect:.2f} ms)",
#     xy=(x_intersect, y_intersect),
#     xytext=(x_intersect + 0.02, y_intersect + 5),
#     arrowprops=dict(arrowstyle="->", lw=1)
# )


plt.xlabel("Sparsity (%)")
plt.ylabel("Decode Time (ms)")
plt.title("Decode Time vs Latent Sparsity for Decoder (batch size 1)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("sparsity_to_decode_time.png")