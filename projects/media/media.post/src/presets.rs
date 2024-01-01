//! 通用参数组合（preset）：命名栈目录是上游合同，品牌作品的选择留在下游。
//!
//! 目录顺序稳定；每个 preset 的算子参数逐值吸收自下游既有命名栈，显示域
//! `Bloom` 语义下像素行为与原效果栈一致。`volumetric` 不入库目录，供示范
//! 组合直接构造。

use geometry_types::Rgba;

use crate::display_ops::{Bloom, Chroma, Displace, Grain, Sweep};
use crate::effect::EffectStack;
use crate::generative::{BeamsEffect, ParticlesEffect};
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
    EffectStack::new()
        .push(Bloom::new(0.62, 0.55, 0.55))
        .push(Chroma::new(0.0015))
        .push(Grain::new(0.16, 900.0))
}

fn dreamy() -> EffectStack {
    EffectStack::new()
        .push(Bloom::new(0.5, 0.8, 0.7))
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
    EffectStack::new()
        .push(Bloom::new(0.45, 0.7, 0.75))
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
    EffectStack::new()
        .push(BeamsEffect(Beams::new(7, 11, COOL)))
        .push(ParticlesEffect(Particles::dust(90, 7, AMBER)))
        .push(Bloom::new(0.5, 0.6, 0.6))
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
