# media.image

像素与浮点图像存储：解码位图 `Bitmap`、渲染目标 `Frame`、布局 `Fit` 与线性 HDR `Framebuffer`，以及它们之间的格式转换。

- 只拥有存储、格式与转换，不拥有绘制语义（归 `media.raster`）与后处理语义（归 `media.post`）。
- 不依赖 scene、raster、post、codec，任何消费者都能单独取用。
