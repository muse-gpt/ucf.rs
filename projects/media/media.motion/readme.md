# media.motion

Animation evaluation for the media family.

- **Identity** (`NodeId`): one stable name per animatable thing, so a revision,
  a lock or a comment can point at the same object across edits.
- **Tracks** (`Key`, `Track`): one property, its keys, its easing and its
  extrapolation. Evaluation is pure: the same frame always yields the same
  value, which is what lets a preview and an export share one code path.
- **Time** (`TimeWindow`, `LoopMode`): explicit mapping from project frames to
  the local frames a track is authored in.
- **Motion** (`Animated`, `Motion`, `Follow`): the authoring shape (optional
  tracks) and its evaluated result (plain numbers) for one node.

Nothing here knows about scenes, items or pixels; `media.scene` owns the IR and
`media.render` owns drawing.
