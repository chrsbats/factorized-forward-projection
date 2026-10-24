# Future directions: convolutional and attention architectures

Forward Projection can be instantiated with convolutional and attention blocks; the
original paper demonstrates convolutional networks and a small vision transformer.
Our own sequence experiments were **adaptations**, and their negative results should
be read at that scope.

## What we observed

We adapted FP to a causal convolutional residual network and to transformer-style
language models. Two measured issues dominated those runs:

1. **Target composition.** A local target constructed from a layer's own inputs is
   useful when the required label information is already linearly available in those
   inputs. In our causal-convolution probes, a block's label target was informative
   when the answer was directly visible in that block's tapped inputs, and close to
   uninformative when the answer required information that an earlier block would
   have had to create. Greedy one-pass fitting therefore gave the earlier block no
   incentive to expose the features the later block needed. This is a property of the
   target construction, not a proof that attention or convolution cannot be fitted
   this way.

2. **Scale alignment.** In the same adaptation, the predeclared residual-update scale
   was small enough that the fitted signal was effectively suppressed: at 5% of the
   incoming stream RMS, a convolution block learned nothing despite informative
   targets; at 20× that scale it learned every directly reachable task and beat the
   controls. The working scale was outside the original sweep grid, so the main run
   never explored it. This is an optimization/scale interaction, not an architectural
   impossibility.

For transformers, the difficulty is analogous: an intermediate target must be
predictable from the current representation while remaining useful after attention,
normalization and residual composition. Random label codes per token and greedy local
targets did not supply that pressure in our tests. Attention also introduces several
coupled parameter groups (Q/K/V, projections), so a single fitted local map is not a
drop-in replacement for a transformer block.

## What this does and does not establish

- It does **not** establish that FP is incompatible with CNNs or transformers.
- It does establish that naive local targets plus greedy one-pass fitting can fail to
  create compositional structure, and that scale choices can silently suppress the
  fitted signal.
- The original paper's demonstrated results are per-sample classification tasks,
  including convolutional and vision-transformer models; multi-hop autoregressive
  prediction is a different setting.

## Promising next experiments

- **Composition-pressure targets** that reward an earlier block for exposing features
  a later block needs, rather than only reconstructing its own local code.
- **Iterative or multi-pass FP**, refitting earlier layers after later targets are
  available, instead of a single greedy pass.
- **Architecture-aware target spaces** for attention: targets on value/projection
  outputs or on residual-stream increments, with normalization handled explicitly.
- **Scale/initialization sweeps** that include the regime where update RMS is
  comparable to the incoming stream, declared before the run.
- **Held-out target-quality diagnostics** that measure how much of a layer's target is
  predictable from its own inputs, so failures can be attributed to the target rather
  than the optimizer.
