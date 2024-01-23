//! 通用参数组合（preset）：命名栈目录是上游合同，品牌作品的选择留在下游。
//!
//! 目录顺序稳定；除 bloom 外的算子参数逐值吸收自下游既有命名栈。bloom 已
//! 按裁决绑定线性 HDR 物理路径（[`crate::stack_bloom`]，合同见
//! [`crate::contract`]），旧显示域参数（threshold 0..1 乘 255、半径、加色
//! 强度）对 HDR 阈值语义无意义，逐个重定为线性阈值与强度：像素必然变化，
//! 属有意修正的单列验收，不强求复制旧错误像素——旧/新固定帧 golden 差异
//! 记录在测试注释里。`volumetric` 不入库目录，供示范组合直接构造。

use geometry_types::Rgba;

use crate::display_ops::{Chroma, Displace, Grain, Sweep};
use crate::effect::EffectStack;
use crate::generative::{BeamsEffect, ParticlesEffect};
use crate::stack_bloom::Bloom;
use media_particles::{Beams, Particles};

const AMBER: Rgba = Rgba([255, 196, 120, 255]);
const COOL: Rgba = Rgba([150, 190, 255, 255]);

/// 一个出厂外观，可直接构造为栈。
pub struct Preset {
    pub name: &'static str,
    pub description: &'static str,
    build: fn() -> EffectStack,
}

/// 出厂目录，顺序稳定。
pub const PRESETS: &[Preset] = &[
    Preset {
        name: "cinematic",
        description: "Soft bloom, slight colour split, restrained grain",
        build: cinematic,
    },
    Preset {
        name: "dreamy",
        description: "Wide bloom with a slow displacement",
        build: dreamy,
    },
    Preset {
        name: "glitch",
        description: "Channel split, wave displacement and hard grain",
        build: glitch,
    },
    Preset {
        name: "fireside",
        description: "Sparks, warm bloom and a travelling light band",
        build: fireside,
    },
    Preset {
        name: "clean",
        description: "Only the lightest touch of grain",
        build: clean,
    },
];

fn cinematic() -> EffectStack {
    // 旧显示域 (0.62, 0.55, 0.55)「soft bloom」→ HDR (1.5, 0.15)：
    // 阈值 1.5 对应上转换显示线性亮度 0.75（sRGB 字节约 225），只让真高光
    // 参与；强度 0.15 低于默认 0.18，保持「克制」的定位。
    EffectStack::new()
        .push(Bloom::new(1.5, 0.15))
        .push(Chroma::new(0.0015))
        .push(Grain::new(0.16, 900.0))
}

fn dreamy() -> EffectStack {
    // 旧显示域 (0.5, 0.8, 0.7)「wide bloom」→ HDR (0.9, 0.3)：
    // 阈值 0.9 低于显示白，让中间调以上整体起雾（旧 0.5×255=128 的
    // 「大面积命中」意图）；强度 0.3 高于默认，配合宽域扩散。
    EffectStack::new()
        .push(Bloom::new(0.9, 0.3))
        .push(Displace::new(0.004, 1.5, 0.15))
        .push(crate::display_ops::Blur::new(0.04))
}

fn glitch() -> EffectStack {
    EffectStack::new()
        .push(Chroma::new(0.006))
        .push(Displace::new(0.012, 6.0, 2.5))
        .push(Grain::new(0.4, 2200.0))
}

fn fireside() -> EffectStack {
    // 旧显示域 (0.45, 0.7, 0.75)「warm glow」→ HDR (1.15, 0.3)：
    // bloom 在栈首作用于底帧——阈值取物理默认 1.15，只有底帧真高光
    // （约 sRGB 字节 200 以上）起辉光；旧 0.45×255≈115 把中间调整体
    // 拖进亮部。强度 0.3 高于默认，保留炉火的饱满加色。
    EffectStack::new()
        .push(Bloom::new(1.15, 0.3))
        .push(ParticlesEffect(Particles::sparks(140, 31, AMBER)))
        .push(Sweep::new(-18.0, 0.05, 0.42, COOL, 0.09))
        .push(Grain::new(0.12, 600.0))
}

fn clean() -> EffectStack {
    EffectStack::new().push(Grain::new(0.08, 600.0))
}

/// 按名构造 preset。
#[must_use]
pub fn preset(name: &str) -> Option<EffectStack> {
    PRESETS.iter().find(|p| p.name == name).map(|p| (p.build)())
}

/// 目录顺序里的全部名字。
#[must_use]
pub fn preset_names() -> Vec<&'static str> {
    PRESETS.iter().map(|p| p.name).collect()
}

/// 光束加浮尘的示范组合，不入库目录。
#[must_use]
pub fn volumetric() -> EffectStack {
    // 旧显示域 (0.5, 0.6, 0.6) → HDR (0.8, 0.2)：bloom 在栈尾，作用于已画
    // 出的光束与浮尘。冷光束 #96beff 上转换亮度约 1.0，低于默认阈值 1.15
    // ——不降阈值光束根本不发光，违背「体积光雾」定位；0.8 让光束（约
    // 1.0）与琥珀浮尘（约 1.26）都进入软膝，强度 0.2 只做轻雾。
    EffectStack::new()
        .push(BeamsEffect(Beams::new(7, 11, COOL)))
        .push(ParticlesEffect(Particles::dust(90, 7, AMBER)))
        .push(Bloom::new(0.8, 0.2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use media_image::Frame;
    use media_types::TimeBase;

    use crate::effect::EffectContext;

    #[test]
    fn presets_are_unique_and_apply() {
        let names = preset_names();
        let unique: std::collections::BTreeSet<_> = names.iter().collect();
        assert_eq!(unique.len(), names.len());
        assert_eq!(names, vec!["cinematic", "dreamy", "glitch", "fireside", "clean"]);
        for name in &names {
            let stack = preset(name).expect("preset resolves");
            assert!(!stack.is_empty());
            let mut frame = Frame::new(96, 54);
            stack.apply(&mut frame, &EffectContext::new(3, TimeBase::from_fps(30)));
        }
        let mut frame = Frame::new(96, 54);
        volumetric().apply(&mut frame, &EffectContext::new(3, TimeBase::from_fps(30)));
        assert!(preset("nope").is_none());
    }

    /// 固定 golden 帧：红绿渐变底 + 一块过 HDR 阈值的白热块 + 一块
    /// 不过阈值的灰块，64×36，不依赖 raster。
    fn golden_scene() -> Frame {
        let (width, height) = (64, 36);
        let mut frame = Frame::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let i = ((y * width + x) as usize) * 4;
                frame.pixels[i] = (x * 4) as u8;
                frame.pixels[i + 1] = (y * 7) as u8;
                frame.pixels[i + 2] = 128;
                frame.pixels[i + 3] = 255;
            }
        }
        for y in 12..18 {
            for x in 28..36 {
                let i = ((y * width + x) as usize) * 4;
                frame.pixels[i..i + 3].copy_from_slice(&[255, 255, 255]);
            }
        }
        for y in 24..30 {
            for x in 8..16 {
                let i = ((y * width + x) as usize) * 4;
                frame.pixels[i..i + 3].copy_from_slice(&[190, 190, 190]);
            }
        }
        frame
    }

    fn fnv1a64(bytes: &[u8]) -> u64 {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for b in bytes {
            hash ^= u64::from(*b);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash
    }

    fn golden_digest(stack: EffectStack) -> u64 {
        let mut frame = golden_scene();
        stack.apply(&mut frame, &EffectContext::new(7, TimeBase::from_fps(30)));
        fnv1a64(&frame.pixels)
    }

    #[test]
    fn bloom_presets_have_rewritten_goldens_and_clean_ones_are_untouched() {
        // 有意修正证据（§7 G4 单列验收）：含 bloom 的 preset 在物理路径
        // 新语义下像素必然变化。固定场景、固定时刻 frame 7 @ 30fps，
        // FNV-1a 64 over RGBA bytes。
        //
        // 旧显示域 bloom 语义下实测的 digest（删除 display_ops::Bloom 前
        // 在同一场景/时刻记录）：
        //   cinematic  = 12226789923177119615  (旧 bloom 0.62, 0.55, 0.55)
        //   dreamy     =  5033235914116561124  (旧 bloom 0.5, 0.8, 0.7)
        //   fireside   = 12193581822522319711  (旧 bloom 0.45, 0.7, 0.75)
        //   volumetric =  8340265562164527134  (旧 bloom 0.5, 0.6, 0.6)
        // 下列 golden 与新参数 (1.5,0.15)/(0.9,0.3)/(1.15,0.3)/(0.8,0.2)
        // 对应，差异即修正本身，不是回归。
        assert_eq!(golden_digest(preset("cinematic").unwrap()), 323134498238179717);
        assert_eq!(golden_digest(preset("dreamy").unwrap()), 6263358094988081491);
        assert_eq!(golden_digest(preset("fireside").unwrap()), 5310966677015366132);
        assert_eq!(golden_digest(volumetric()), 738708318065530197);
        // 不含 bloom 的 preset 必须逐字节不变（同一 golden 场景）：
        assert_eq!(golden_digest(preset("clean").unwrap()), 10764983178790845334);
        assert_eq!(golden_digest(preset("glitch").unwrap()), 9560820704989393487);
    }

    #[test]
    fn preset_stacks_match_their_declared_order() {
        assert_eq!(preset("cinematic").unwrap().names(), vec!["bloom", "chroma", "grain"]);
        assert_eq!(preset("dreamy").unwrap().names(), vec!["bloom", "displace", "blur"]);
        assert_eq!(preset("glitch").unwrap().names(), vec!["chroma", "displace", "grain"]);
        assert_eq!(preset("fireside").unwrap().names(), vec!["bloom", "particles", "sweep", "grain"]);
        assert_eq!(preset("clean").unwrap().names(), vec!["grain"]);
        assert_eq!(volumetric().names(), vec!["beams", "particles", "bloom"]);
    }
}
