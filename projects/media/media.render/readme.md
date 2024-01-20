# media.render

帧编排，两个独立组合入口，各按 feature 门控：

- `scene`（默认）`render_project`：把 `Project` 逐帧画出来、把音频按秒切块混进编码器。
  - 世界按全片最大 zoom 超采样光栅化，推近时不会糊。
  - 与 `media.codec` 组合成一条从 JSON 到 MP4 的完整链路。
- `image-pipeline` `process_image`：无 scene 的图像管线 decode→处理→encode（显影配方或效果栈 + 时刻），输出 PNG，依赖图只含 codec、post、image、types，不拉 scene、text、audio。照片与特效宿主调用这一入口，绑定侧零编排。
