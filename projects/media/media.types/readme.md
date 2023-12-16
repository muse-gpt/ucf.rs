# media.types

媒体库共用的值：`Error`、确定性 `hash`、项目默认音频采样率。

- 不依赖其它媒体 crate，也不依赖设备。
- 帧是时间的纯函数，所以随机只能来自 `hash(frame seed)`，不能用全局 RNG。
