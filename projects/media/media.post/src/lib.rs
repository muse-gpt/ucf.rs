//! HDR 帧缓冲与显示后处理。
//!
//! CPU 参考实现与 UCF 便携程序并行提供，二者通过 parity 测试互相约束；
//! 程序构建与 CPU 解释执行基于 `hardware-program`，颜色使用
//! `geometry-types` 的 `LinearRgb`，`Framebuffer` 存储归 `media.image`。
//! 本 crate 只拥有处理语义与其执行，不包含任何领域物理。
//!
//! 输入域、传递函数、alpha、边界采样、模糊半径与 tone mapping 时机的正式合同
//! 成文于 [`contract`]，每个算子以 [`contract::InputDomain`] 显式标注自己消费
//! 线性 HDR 还是显示 RGBA8；显示空间可分离盒模糊的唯一实现见 [`display_blur`]。
//! 显示空间算子（[`display_ops`]）、有序栈合同（[`effect`]）、中立 JSON 栈
//! 声明（[`stack_spec`]）与通用参数组合（[`presets`]）共同构成效果栈；
//! 粒子与光束的生成求值归 `media.particles`，本 crate 只做栈适配。

mod bloom_scale;
mod box_blur;
mod bright_extract;
mod contract;
mod display_blur;
mod display_ops;
mod effect;
mod generative;
mod post;
mod post_ucf;
mod presets;
mod stack_spec;
mod tone_map_batch;

pub use bloom_scale::{UcfDownsample2xRgbBuilder, UcfUpsampleRgbBuilder};
pub use box_blur::UcfBoxBlur3x3Builder;
pub use bright_extract::HdrBrightExtractBuilder;
pub use contract::{InputDomain, relative_radius, short_side};
pub use display_blur::display_box_blur;
pub use display_ops::{Bloom, Blur, Chroma, Displace, Grain, Sweep};
pub use effect::{Effect, EffectContext, EffectStack};
pub use generative::{BeamsEffect, ParticlesEffect};
pub use hardware_program::{post_bloom_extract, tone_map_aces};
pub use post::{
    aces_filmic, apply_post_process, apply_tone_map, bloom_pass, composite_bloom_from_bright,
    extract_bright_pass, to_rgba8, tone_map_rgb, DEFAULT_BLOOM_INTENSITY,
    DEFAULT_BLOOM_THRESHOLD, DEFAULT_EXPOSURE,
};
pub use post_ucf::{apply_tone_map_ucf, bloom_pass_ucf, extract_bright_pass_ucf};
pub use presets::{Preset, PRESETS, preset, preset_names, volumetric};
pub use stack_spec::{EffectSpec, StackSpec};
pub use tone_map_batch::tone_map_aces_batch;
