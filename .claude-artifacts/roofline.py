import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── A100 hardware ceilings ────────────────────────────────────────────────────
DRAM_PEAK   = 2000    # HBM2e ~2 TB/s
L2_PEAK     = 12288   # ~12 TB/s
L1_PEAK     = 33792   # ~33 TB/s
INT32_PEAK  = 19500   # A100 INT32 ~19.5 TOPS

# ── Measured values (dense_500MiB, A100, BS=256, IPT=22) ─────────────────────
# Bytes: 524288015 uint8 in + 524288015 uint16 out = 1572864045
total_bytes = 524_288_015 + 524_288_015 * 2

# bw_ceiling_xpose: warp-transpose load+store, no scan, no DFA
# NCU: 113199884 instrs (avg of 2 runs), 1.51 ms
xpose_instrs  = 113_199_884
xpose_runtime = 1.51e-3
AI_xpose  = xpose_instrs / total_bytes
perf_xpose = xpose_instrs / xpose_runtime / 1e9

# p1_add: warp-transpose + decoupled lookback scan (AddOp, no DFA)
# NCU: avg(278044378, 277926042) = 277985210 instrs, avg(2.76, 2.75) ms
add_instrs   = (278_044_378 + 277_926_042) // 2
add_runtime  = (2.76e-3 + 2.75e-3) / 2
AI_add  = add_instrs / total_bytes
perf_add = add_instrs / add_runtime / 1e9

# p1_transpose: warp-transpose + lookback + DFA composition
# NCU: avg(448732237, 448864301) = 448798269 instrs, avg(3.23, 3.24) ms
p1_instrs   = (448_732_237 + 448_864_301) // 2
p1_runtime  = (3.23e-3 + 3.24e-3) / 2
AI_p1  = p1_instrs / total_bytes
perf_p1 = p1_instrs / p1_runtime / 1e9

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#0f1117')
ax.set_facecolor('#0f1117')

AI_range = np.logspace(-3, 3, 1000)

def roof(bw, peak_ops, ai):
    return np.minimum(bw * ai, peak_ops)

c_dram = '#4fc3f7'
c_l2   = '#81c784'
c_l1   = '#ffb74d'
c_int  = '#ef5350'

ax.plot(AI_range, roof(DRAM_PEAK,  INT32_PEAK, AI_range), color=c_dram, lw=1.5, ls='--', label='DRAM roof (2 TB/s)')
ax.plot(AI_range, roof(L2_PEAK,    INT32_PEAK, AI_range), color=c_l2,   lw=1.5, ls='--', label='L2 roof (12 TB/s)')
ax.plot(AI_range, roof(L1_PEAK,    INT32_PEAK, AI_range), color=c_l1,   lw=1.5, ls='--', label='L1 roof (33 TB/s)')
ax.axhline(INT32_PEAK, color=c_int, lw=1.5, ls=':', label='INT32 compute roof (19.5 TOPS)')

# bw_ceiling_xpose
ax.scatter([AI_xpose], [perf_xpose], color='#aaaaaa', s=70, zorder=5, marker='s')
ax.annotate(f'bw_ceiling_xpose\n(xpose only, no scan)\nAI={AI_xpose:.3f}  {perf_xpose:.0f} GOp/s',
            xy=(AI_xpose, perf_xpose),
            xytext=(AI_xpose * 0.15, perf_xpose * 1.8),
            color='#aaaaaa', fontsize=8,
            arrowprops=dict(arrowstyle='->', color='#aaaaaa', lw=1),
            ha='right')

# p1_add
ax.scatter([AI_add], [perf_add], color='#ce93d8', s=70, zorder=5, marker='^')
ax.annotate(f'p1_add\n(xpose + lookback scan)\nAI={AI_add:.3f}  {perf_add:.0f} GOp/s',
            xy=(AI_add, perf_add),
            xytext=(AI_add * 5.0, perf_add * 1.5),
            color='#ce93d8', fontsize=8,
            arrowprops=dict(arrowstyle='->', color='#ce93d8', lw=1),
            ha='left')

# p1_transpose
ax.scatter([AI_p1], [perf_p1], color='white', s=80, zorder=5)
ax.annotate(f'p1_transpose\n(xpose + lookback + DFA)\nAI={AI_p1:.3f}  {perf_p1:.0f} GOp/s',
            xy=(AI_p1, perf_p1),
            xytext=(AI_p1 * 5.0, perf_p1 * 0.4),
            color='white', fontsize=8,
            arrowprops=dict(arrowstyle='->', color='white', lw=1),
            ha='left')

# Ridge point
ridge_dram = INT32_PEAK / DRAM_PEAK
ax.axvline(ridge_dram, color=c_dram, lw=0.6, ls=':', alpha=0.5)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1e-2, 1e2)
ax.set_ylim(1, 1e5)
ax.set_xlabel('Arithmetic Intensity (ops / byte)', color='white', fontsize=11)
ax.set_ylabel('Performance (GOp/s)', color='white', fontsize=11)
ax.set_title('Roofline — A100 (integer) · p1 kernel family', color='white', fontsize=12, pad=10)

ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('#444')

ax.grid(True, which='both', color='#333', lw=0.5, ls='-')
ax.legend(fontsize=8, facecolor='#1e2230', edgecolor='#555', labelcolor='white',
          loc='upper left')

plt.tight_layout()
plt.savefig('/home/due/git/github.com/LiljeDue/lexer-experiments/.claude-artifacts/roofline.png',
            dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print('saved')
