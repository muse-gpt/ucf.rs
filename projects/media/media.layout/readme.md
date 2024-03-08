# media.layout

面板文档：页面/面板/标注 spec、grid 网格、文字盒自动尺寸、气泡几何与编号总览计划。

- 语义逐值吸收自 OpenComic 核心（Apache-2.0，© OpenComic Authors）：文字盒 pad=0.6×size、行高=1.25×size、盒宽=最宽行+2×pad、基线=顶+0.82×size；Caption 左上锚定、Balloon 中心锚定+逐行水平居中；气泡圆角=min(box_h/3, 12) 的切角八边形描边与尾巴射线求交（底边半宽 7.0）；总览 gutter 固定 16px、cell=round(page×scale).max(1)、页码标签高 cell_h/8、Bold、色 (20,20,24,235)。
- 本 crate 不画像素、不写文件：计划（`DrawOp`/`SheetPlan`）由 `media.render` 的 `layout-export` 组合入口经 `media.raster`/`media.text`/`media.codec` 落地。
- 排版复用 `media.text`（`TextMeasure` 缝），不建第二套文档排版引擎；不做系统字体发现，字体字节由宿主显式注入。
- 输入合同住在 Rust：绑定层默认值全部上移为 spec 常量；非法颜色、未知 `fit` 拼写、越界字号/边宽/总览参数是显式 `invalid_input`（有意修正旧绑定的静默回落）；缺图/坏图保持整页失败（`ImageStore` 解析错误即传播）。
- 只依赖 `media.types`、`geometry`、`media.image`、`media.text`（core 层）与 serde。
