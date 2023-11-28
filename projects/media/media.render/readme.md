# media.render

帧编排：把 `Project` 逐帧画出来、把音频按秒切块混进编码器。

- 世界按全片最大 zoom 超采样光栅化，推近时不会糊。
- 与 `media.codec` 组合成一条从 JSON 到 MP4 的完整链路。
