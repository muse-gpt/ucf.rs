# media.types

媒体库共用的值：`Error`、确定性 `hash`、有理时间基 `TimeBase`、项目默认音频采样率。

- 不依赖其它媒体 crate，也不依赖设备。
- 帧是时间的纯函数，所以随机只能来自 `hash(frame seed)`，不能用全局 RNG。
- 时间只有一个事实来源：`TimeBase` 以精确有理数 fps 推导 frame、秒与采样点，禁止三份可互相矛盾的裸值并存。
