# media.text

文字排版与字形光栅：字体解析、测量、换行、逐字入场、代码行高亮。

- 字体字节由调用方注入（`Fonts::from_bytes`），本仓不存任何字体资产，也从不探测系统字体路径；系统字体发现只存在于 `media.render` 的 scene 影片入口（宿主 IO 边界）。
- 分层是 feature 合同：core（恒有）= `from_bytes`/`measure`/`wrap`，纯度量，不拉 raster/scene/像素存储，供 `media.layout` 这类叶子消费；`paint` = `draw_line`（经 `media.raster` 混合字形位图）；`blocks` = `draw_block`/`draw_staggered`/`draw_code_line`（面向 scene IR 的展示助手）。默认 `blocks`，workspace 入口默认关，消费者按层显式开启。
- 字形缺失时逐字回退，等宽字体里的中文不会变豆腐块。
