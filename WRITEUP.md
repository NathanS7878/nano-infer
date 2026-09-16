# The ceiling that wasn't: predicting kernel speedups by counting bytes, and the time it was wrong by 10×

Three of the four CUDA kernels in this project hit a performance ceiling I
calculated before writing a line of CUDA. The fourth missed it by a factor of
ten. Finding out why taught me more than the three that worked.

## Counting bytes

Every kernel here replaces a sequence of PyTorch ops. PyTorch executes those
eagerly, so each one makes a full round trip through VRAM: read the inputs, write
the result, and the next op reads it straight back. A fused kernel does the whole
sequence in registers and touches memory once.

For a memory-bound op, that makes the speedup predictable. Count the bytes
PyTorch moves, count the bytes you have to move, divide.

RoPE — the rotary position embedding — is a good example. PyTorch runs
`x * cos + rotate_half(x) * sin` as five kernels. Per element of `x`:

| step | traffic |
|---|---|
| `-x2` | 2E |
| `cat((-x2, x1))` | 4E |
| `x * cos` | 4E |
| `rotated * sin` | 4E |
| `t1 + t2` | 6E |
| **PyTorch total** | **20E** |
| **compulsory minimum** (read x once, write once) | **4E** |

Predicted ceiling: 5.00×. **Measured: 5.03×.** SwiGLU predicted 1.67× and
measured 1.65×. Both within one percent, from arithmetic done on paper.

The interesting part of RoPE isn't the number, it's *where* the bytes went. Six
of PyTorch's twenty are `rotate_half`, an operation that computes nothing at all.
It exists only to rearrange an operand into the layout the next elementwise
kernel expects. A fused kernel replaces it with an index offset. **The most
profitable thing to fuse is usually not the expensive math — it's the plumbing
wrapped around it.**

So by the third kernel I trusted the method.

## The fourth kernel

Kernel 4 is decode attention with online softmax: one query token against the
whole cached KV, never materializing the attention matrix. It's the kernel that
matters, because Phase 2 measured decode running at 26.5 GB/s — **5.9% of this
card's 448 GB/s peak**. Kernels 1–3 couldn't move that; at decode they operate on
tensors of a few tens of kilobytes. This one streams the entire KV cache.

I counted the bytes. Per sequence, per layer, with L cached positions and grouped
query attention (14 query heads, 2 KV heads):

| step | traffic |
|---|---|
| `cache.gather` | 1024L |
| `repeat_kv` — materializes a 7× copy of K and V | **4096L** |
| `q @ kᵀ` | 3612L |
| mask / softmax / cast round trips | 392L |
| `probs @ v` | 3612L |
| **PyTorch total** | **~12736L** |
| **ours** — read K once, read V once | **512L** |

Predicted ceiling: **~24×**.

Measured: **2.48×.**

## Being wrong usefully

A miss that large means the model is broken, not that the kernel needs tuning. So
before touching anything I wrote down what could explain it.

The obvious suspect was my own design. I map one thread block to each
(sequence, query head) pair. With GQA, seven query heads share a single KV head —
so seven blocks each stream the same K and V independently. If those reads were
all reaching DRAM, I'd be moving seven times the compulsory traffic and the
ceiling would evaporate.

That's a testable claim, and the test is a controlled experiment: **hold the KV
pool completely fixed and vary only how many query heads share it.** Compulsory
traffic stays constant by construction; only the duplication changes.

| query heads | thread blocks | time |
|---|---|---|
| 2 | 64 | 281.8 µs |
| 4 | 128 | 282.4 µs |
| 8 | 256 | 292.0 µs |
| 14 | 448 | 510.2 µs |

From 2 to 8 query heads the work quadruples and **the wall time does not move**.

That single row of numbers kills two hypotheses at once. It isn't bandwidth
saturation — bandwidth-bound code gets slower when you ask it to move more. And
it isn't the duplicated reads, because quadrupling them cost 4%. Flat time
against rising work is the signature of something else entirely: a **latency
bound**. The kernel was spending its life waiting, and there was so much idle
capacity that four times the work fit in the same wall clock.

Once you see it that way the cause is obvious in hindsight. Online softmax is
inherently sequential across tiles — each tile needs the running maximum from the
one before it, then rescales everything accumulated so far. Each tile also cost
me several `__syncthreads()` barriers. And at 128 threads per block there were
only four warps available to hide the memory latency between those barriers. The
kernel wasn't bandwidth-starved; it was *dependency*-starved.

## The fix follows from the diagnosis

If latency is the problem, the lever is warps per SM and tiles per sequence —
both of which are set by one number, the block size. Sweeping it:

| case | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|
| batch 32, L=512 | 264.0 | 268.3 | 217.1 | **199.1** | 232.9 µs |
| batch 32, L=2048 | 970.8 | 936.6 | 749.5 | **668.7** | 749.0 µs |
| batch 1, L=1024 | 278.0 | 185.7 | 129.6 | 93.1 | **51.6** µs |

Batch 1 wants the widest block available and gains **3.6×** from it — exactly
what the diagnosis predicts, because with only 14 blocks in the entire grid there
is nothing else resident on the SM and every bit of latency hiding has to come
from inside the block.

Overall: **7.9% → 11.1% of peak, 2.48× instead of 2.25×.** A real improvement
that I would not have found by staring at the code, because the code looks fine.
It looks like a correct online softmax, which it is.

## What I actually learned

**A byte-counted ceiling only binds when the kernel is genuinely bandwidth-bound.**
That sounds obvious written down. It was not obvious while three kernels in a row
confirmed the method and taught me to trust it. The prediction is not a law; it's
a hypothesis that also happens to be a *diagnostic* — when a kernel misses its
byte-counted ceiling badly, that gap is telling you which bound you're actually
against, and it's worth more than the speedup would have been.

The same pattern showed up twice more:

- The **RoPE kernel beat** its 5.00× ceiling, hitting 5.40× at the largest size.
  Beating a ceiling also means the model is wrong. Decomposing it: 5.00× of
  traffic reduction times a 1.08× efficiency gap, because at 117 MB tensors
  PyTorch's chain drops to 81.1% of peak while ours holds 87.6%. Byte counting
  predicts the *floor* of a fusion win, not a bound on it.
- The **INT4 quantized matmul measured exactly the same speed as INT8**, despite
  holding half the bits. If either were bandwidth-bound, INT4 would be roughly
  twice as fast. Identical timings, plus a ~33 µs floor that appeared at every
  matrix size including one 8× smaller than another, said both were launch-bound.
  The 2× and 4× ceilings I'd computed were never in play, and quoting them as
  achieved would have been wrong.

Three kernels bandwidth-bound, one latency-bound, one launch-bound — and the byte
count is only the right predictor for the first group. **The number worth
reporting isn't "N× faster than PyTorch." It's percentage of peak bandwidth,
because that one tells you how much room is left, and whether you're even in the
right regime to be asking.**

### Postscript: the metric was also wrong

The paragraph above used to end here, with decode attention at 11.1% of peak and
two named-but-unimplemented fixes. Implementing one of them showed that the
number itself had been misread.

11.1% was computed against *compulsory* bytes — each KV element counted once.
But the kernel launched one block per (sequence, query head), and with GQA 14q/2kv
seven blocks each read the same KV rows. So the question "is it near the peak?"
has two different answers depending on which bytes you count, and I had only ever
computed one of them.

The experiment that settles it holds block count and per-block work exactly
constant and varies only how many query heads share a KV head:

| kv heads | n_rep | distinct | issued | µs | issued GB/s | % of peak |
|---|---|---|---|---|---|---|
| 14 | 1 | 235 MB | 235 MB | 1199.9 | 195.8 | 43.7% |
| 2 | 7 | 33.6 MB | 235 MB | 729.3 | 322.1 | **71.9%** |
| 1 | 14 | 16.8 MB | 235 MB | 694.2 | 338.4 | 75.5% |

Wall time flattens once n_rep ≥ 7: the duplicate reads *are* absorbed by cache,
so DRAM was never the limit. But a cache hit still costs a load instruction and
an issue slot, and against the traffic it actually asks for the kernel was at
**72% of peak, not 11%.** It was never leaving 89% of the card unused. It was a
load path near its ceiling, carrying seven times more traffic than the algorithm
requires.

That reframes the fix completely. "Go faster" was never available. "Ask for less"
was. One block per *KV* head, each K and V element read once into a register and
fed to all seven query heads sharing it: **2.60× on the kernel, 6.89× against the
Phase 2 paged path, 11.1% → 30.3% of compulsory peak**, at numerically identical
error against the fp64 ground truth.

So the lesson survives in a sharper form. Percentage of peak is still the number
worth reporting — but a percentage is a ratio, and **the numerator has to be the
bytes the hardware actually moved, not the bytes your algorithm deserved to
move.** The gap between those two is not noise. It *was* the optimization.

And the sting in the tail: the refit did not reach end-to-end throughput. A
2.6× faster kernel moved `generate_paged` by 0.99×, because the decode step took
~20 ms whether the batch was 1 or 32 and the context 33 tokens or 1025 — ~3,200
aten dispatches per step, only 169 of them matmuls. The loop was host-bound.
Which is Gotcha #4 of this project, recurring one level up: *profile before
optimizing, and profile the thing you are actually waiting on.*

I first drew the wrong lesson from that — that the Phase 3 kernels were
optimizing a part of the step nobody was waiting on. The same engine, with all
kernels off versus on, is 2.2–2.5× on decode. What had actually been flat was
kernel 4 versus 4b, which launch the *same number* of kernels. In a host-bound
loop, a fusion pays through the launches it deletes, not the GPU time it saves:
RMSNorm and SwiGLU each turned a chain of PyTorch ops into one launch; 4b turned
one launch into one launch.

Capturing the decode step as a CUDA graph removed the host from the loop: decode
5.3× faster at batch 32, and a batch-1 step now streams its 988 MB of weights at
60% of peak bandwidth rather than 11%. Only then did step time start growing with
context, and only then did the attention kernel become something the engine was
actually waiting on — kernels on versus off under graphs is 3.15× at a 512-token
context.

---

*Every number above is reproducible: `python -m bench.kernel_attention`,
`python -m bench.kernel_rope`, `python -m bench.quant_speed`. Hardware, method,
and the full set of measurements are in [SUMMARY.md](SUMMARY.md); the scaling
experiment and block-size sweep are logged in [PROGRESS.md](PROGRESS.md).*
