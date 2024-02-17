# media.voxel

稀疏体素：`VoxelGrid`（`BTreeMap` 整数格，负坐标天然支持）、露出面提取 `faces`、显式着色策略 `Shading` 与等距预览计划 `preview`，以及宿主输入合同 `VoxelSpec`。

- 确定性是合同：格按 `BTreeMap` 序遍历、六向按 +Y/-Y/+X/-X/+Z/-Z 固定顺序出面、预览按面深度稳定排序。同一份面序列同时喂预览计划与 OBJ 编码，预览与导出永不打架；禁止改成 `HashMap` 或并行遍历。
- 剔除只看占位不看颜色：alpha=0 的体素仍遮住相邻面（源实现语义，测试锁定）。
- 亮度不住在网格里：六个固定阴影常量改为命名预设 `Shading::FixedSixDir`（+Y 1.0、-Y 0.45、+X 0.72、-X 0.62、+Z 0.58、-Z 0.66），它是显示预设而非物理材质，预览构建时显式传入；预览 alpha 强制 255 是文档化的预览合同。
- 本 crate 不画像素、不写文件：多边形填充由 `media.render` 的 `voxel-export` 组合入口驱动 `media.raster`，OBJ 编码由 `media.codec` 的 `obj` feature 单向消费 `Face` 序列。
- 输入合同住在 Rust：缺省 960x640、背景 `#0a0e1a`；非法 hex 颜色是显式 `invalid_input` 错误（吸收自 OpenVoxel 绑定时的有意修正——旧绑定静默回退浅灰）。
- 只依赖 `media.types`、`geometry` 与 serde。
