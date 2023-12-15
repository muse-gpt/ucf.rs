//! HDR 帧缓冲与显示后处理。
//!
//! CPU 参考实现与 UCF 便携程序并行提供，二者通过 parity 测试互相约束；
//! 程序构建与 CPU 解释执行基于 `hardware-program`，颜色使用
//! `geometry-types` 的 `LinearRgb`，`Framebuffer` 存储归 `media.image`。
//! 本 crate 只拥有处理语义与其执行，不包含任何领域物理。

mod bloom_scale;
mod box_blur;
mod bright_extract;
mod post;
mod post_ucf;
mod tone_map_batch;

pub use bloom_scale::{UcfDownsample2xRgbBuilder, UcfUpsampleRgbBuilder};
pub use box_blur::UcfBoxBlur3x3Builder;
pub use bright_extract::HdrBrightExtractBuilder;
pub use hardware_program::{post_bloom_extract, tone_map_aces};
pub use post::{
    aces_filmic, apply_post_process, apply_tone_map, bloom_pass, composite_bloom_from_bright,
    extract_bright_pass, to_rgba8, tone_map_rgb, DEFAULT_EXPOSURE,
};
pub use post_ucf::{
    apply_tone_map_ucf, bloom_pass_ucf, extract_bright_pass_ucf, DEFAULT_BLOOM_INTENSITY,
    DEFAULT_BLOOM_THRESHOLD,
};
pub use tone_map_batch::tone_map_aces_batch;
