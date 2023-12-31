# media-particles

Seeded particle and beam fields for display RGBA8 frames.

- Every dot is a pure function of `(index, seed, seconds)` — no state carries between frames, so any frame can be evaluated on its own, out of order, skipping frames or in parallel, and reproduce the same pixels.
- The seed is the user's, unlike the frame-derived grain seed in `media.post`; the two random contracts stay deliberately separate.
- Sizes and spreads are authored relative to the short side (inherited from the post-processing radius contract); the pixel measures come from the frame itself.
- Storage is `media.image`'s `Frame`; this crate only stamps additive soft dots. Effect-stack wiring and serialization live in `media.post`.
