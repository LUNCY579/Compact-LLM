# TGSA retrofit conversion-damage report

Measured **before any training**, on development inputs only.

**Architecture:** hidden=1024 heads=16 kv=8 head_dim=128 rope_theta=1e+06  
**Converted layers:** [0, 2, 4]  
**Conditioning:** ['q']

## Headline

Conversion is **exactly** function-preserving in exact arithmetic: the
retrofit reduces algebraically to native attention while its conditioning
projections are zero. What remains is floating-point ordering, measured below.

| dtype | seq | KL median | KL max | top-1 agree | top-5 agree | |logit diff| max | norm ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| `float64` | 64 | 0.000e+00 | 0.000e+00 | 1.0000 | 1.0000 | 0.000e+00 | 1.000000 |
| `float32` | 64 | 0.000e+00 | 0.000e+00 | 1.0000 | 1.0000 | 0.000e+00 | 1.000000 |
| `float32` | 256 | 0.000e+00 | 0.000e+00 | 1.0000 | 1.0000 | 0.000e+00 | 1.000000 |
| `bfloat16` | 64 | 0.000e+00 | 0.000e+00 | 1.0000 | 1.0000 | 0.000e+00 | 1.000000 |
| `bfloat16` | 256 | 0.000e+00 | 0.000e+00 | 1.0000 | 1.0000 | 0.000e+00 | 1.000000 |

## Reading this

A float64 row with all-zero drift is the algebraic statement: nothing was
approximated. Non-zero rows at lower precision are the cost of computing the
same quantity in a different summation order, not a loss of pretrained
function. They should be compared against the drift between two runs of the
*native* model at that dtype, which is the floor.

No pass threshold is asserted. The full distribution is given so a reader can
judge, rather than being handed a cutoff invented after seeing the numbers.
