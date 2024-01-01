//! 效果栈的中立声明合同：spec → 校验/规范化 → 执行 → 输出。
//!
//! 从零设计（上游此前没有栈的序列化合同）。形状遵循仓内既有 serde 约定：
//! 结构体 `rename_all = "camelCase"`，tagged enum `tag = "kind"` 且
//! `rename_all = "kebab-case"`。栈有序（前到后应用），空栈合法。
//!
//! 校验即拒绝：参数必须有限且落在算子合同域内，越界返回显式
//! [`media_types::Error::invalid_input`]，不做静默回落。钳制语义仍住在算子
//! 本身（直接 Rust 作者面逐值保持原行为）；声明面把同一参数域当作输入合同
//! 强制执行，绑定层只做宿主值到 [`StackSpec`] 的映射。

use geometry_types::Rgba;
use serde::Deserialize;

use media_types::Error;

use crate::display_ops::{Bloom, Blur, Chroma, Displace, Grain, Sweep};
use crate::effect::{Effect, EffectStack};
use crate::generative::{BeamsEffect, ParticlesEffect};
use media_particles::{Beams, Particles};

/// 一份完整的效果栈声明。
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StackSpec {
    /// 应用顺序即列表顺序；缺省与空列表都是合法的恒等栈。
    #[serde(default)]
    pub effects: Vec<EffectSpec>,
}

/// 栈中的一个算子声明。
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum EffectSpec {
    Blur { amount: f32 },
    Bloom {
        threshold: f32,
        radius: f32,
        strength: f32,
    },
    Chroma { amount: f32 },
    Displace {
        amount: f32,
        frequency: f32,
        speed: f32,
    },
    Sweep {
        angle: f32,
        speed: f32,
        width: f32,
        color: String,
        strength: f32,
    },
    Grain { strength: f32, density: f32 },
    Particles {
        count: u32,
        seed: u32,
        color: String,
        size: f32,
        speed: f32,
        drift: f32,
        twinkle: f32,
    },
    Beams {
        count: u32,
        seed: u32,
        color: String,
        #[serde(default)]
        origin: Option<[f32; 2]>,
        #[serde(default)]
        strength: Option<f32>,
        #[serde(default)]
        sway: Option<f32>,
    },
}

impl StackSpec {
    /// 解析并校验一份栈 JSON。
    pub fn from_json(json: &str) -> Result<Self, Error> {
        serde_json::from_str(json).map_err(|e| Error::invalid_input(format!("stack json: {e}")))
    }

    /// 校验并规范化为可执行栈。
    pub fn compile(&self) -> Result<EffectStack, Error> {
        let mut stack = EffectStack::new();
        for spec in &self.effects {
            stack = stack.push_boxed(spec.to_effect()?);
        }
        Ok(stack)
    }
}

impl EffectSpec {
    /// 校验参数域并构造算子。
    pub fn to_effect(&self) -> Result<Box<dyn Effect>, Error> {
        match self {
            Self::Blur { amount } => {
                let amount = in_range("blur", "amount", *amount, 0.0, 1.0)?;
                Ok(Box::new(Blur::new(amount)))
            }
            Self::Bloom { threshold, radius, strength } => {
                let threshold = in_range("bloom", "threshold", *threshold, 0.0, 1.0)?;
                let radius = in_range("bloom", "radius", *radius, 0.0, 1.0)?;
                let strength = in_range("bloom", "strength", *strength, 0.0, 4.0)?;
                Ok(Box::new(Bloom::new(threshold, radius, strength)))
            }
            Self::Chroma { amount } => {
                let amount = in_range("chroma", "amount", *amount, 0.0, 0.1)?;
                Ok(Box::new(Chroma::new(amount)))
            }
            Self::Displace { amount, frequency, speed } => {
                let amount = in_range("displace", "amount", *amount, 0.0, 0.05)?;
                let frequency = finite("displace", "frequency", *frequency)?;
                let speed = finite("displace", "speed", *speed)?;
                Ok(Box::new(Displace::new(amount, frequency, speed)))
            }
            Self::Sweep { angle, speed, width, color, strength } => {
                let angle = finite("sweep", "angle", *angle)?;
                let speed = finite("sweep", "speed", *speed)?;
                let width = in_range("sweep", "width", *width, 0.005, 1.0)?;
                let color = parse_color("sweep", color)?;
                let strength = in_range("sweep", "strength", *strength, 0.0, 4.0)?;
                Ok(Box::new(Sweep::new(angle, speed, width, color, strength)))
            }
            Self::Grain { strength, density } => {
                let strength = in_range("grain", "strength", *strength, 0.0, 1.0)?;
                let density = in_range("grain", "density", *density, 0.0, f32::INFINITY)?;
                Ok(Box::new(Grain::new(strength, density)))
            }
            Self::Particles { count, seed, color, size, speed, drift, twinkle } => {
                let color = parse_color("particles", color)?;
                let size = in_range("particles", "size", *size, 0.0005, 0.05)?;
                let speed = finite("particles", "speed", *speed)?;
                let drift = finite("particles", "drift", *drift)?;
                let twinkle = in_range("particles", "twinkle", *twinkle, 0.0, 1.0)?;
                Ok(Box::new(ParticlesEffect(Particles {
                    count: *count,
                    seed: *seed,
                    color,
                    size,
                    speed,
                    drift,
                    twinkle,
                })))
            }
            Self::Beams { count, seed, color, origin, strength, sway } => {
                let color = parse_color("beams", color)?;
                let origin = match origin {
                    Some([x, y]) => [
                        in_range("beams", "origin.x", *x, 0.0, 1.0)?,
                        in_range("beams", "origin.y", *y, 0.0, 1.0)?,
                    ],
                    None => [0.5, 0.28],
                };
                let strength = match strength {
                    Some(v) => finite("beams", "strength", *v)?,
                    None => 0.35,
                };
                let sway = match sway {
                    Some(v) => finite("beams", "sway", *v)?,
                    None => 6.0,
                };
                Ok(Box::new(BeamsEffect(Beams {
                    count: *count,
                    seed: *seed,
                    color,
                    origin,
                    strength,
                    sway,
                })))
            }
        }
    }
}

/// 有限且落在 `min..=max` 内；NaN 与无穷一律拒绝。
fn in_range(op: &str, param: &str, value: f32, min: f32, max: f32) -> Result<f32, Error> {
    if value.is_finite() && value >= min && value <= max {
        Ok(value)
    } else {
        Err(Error::invalid_input(format!(
            "{op}.{param} must be within {min}..={max}, got {value}"
        )))
    }
}

/// 只要求有限。
fn finite(op: &str, param: &str, value: f32) -> Result<f32, Error> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(Error::invalid_input(format!("{op}.{param} must be finite, got {value}")))
    }
}

/// `#rgb`、`#rrggbb`、`#rrggbbaa`。
fn parse_color(op: &str, input: &str) -> Result<Rgba, Error> {
    geometry_types::parse(input).ok_or_else(|| {
        Error::invalid_input(format!("{op}.color: not a hex color: {input}"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use media_image::Frame;
    use media_types::TimeBase;

    use crate::effect::EffectContext;

    fn scene() -> Frame {
        let mut frame = Frame::new(96, 54);
        for chunk in frame.pixels.chunks_exact_mut(4) {
            chunk[0] = 90;
            chunk[1] = 60;
            chunk[2] = 40;
            chunk[3] = 255;
        }
        frame
    }

    #[test]
    fn an_empty_stack_is_legal_and_identity() {
        let spec: StackSpec = serde_json::from_str(r#"{"effects": []}"#).unwrap();
        let stack = spec.compile().unwrap();
        assert!(stack.is_empty());
        let mut frame = scene();
        let before = frame.pixels.clone();
        stack.apply(&mut frame, &EffectContext::new(0, TimeBase::from_fps(30)));
        assert_eq!(frame.pixels, before);

        let omitted: StackSpec = serde_json::from_str("{}").unwrap();
        assert!(omitted.compile().unwrap().is_empty());
    }

    #[test]
    fn a_full_stack_round_trips_through_json() {
        // The fireside preset expressed as the neutral spec shape.
        let json = r##"{
            "effects": [
                { "kind": "bloom", "threshold": 0.45, "radius": 0.7, "strength": 0.75 },
                { "kind": "particles", "count": 140, "seed": 31,
                  "color": "#ffc478", "size": 0.0025, "speed": -0.18, "drift": 0.04, "twinkle": 0.85 },
                { "kind": "sweep", "angle": -18.0, "speed": 0.05, "width": 0.42,
                  "color": "#96beff", "strength": 0.09 },
                { "kind": "grain", "strength": 0.12, "density": 600.0 },
                { "kind": "beams", "count": 7, "seed": 11, "color": "#96beff" }
            ]
        }"##;
        let stack = StackSpec::from_json(json).unwrap().compile().unwrap();
        assert_eq!(stack.names(), vec!["bloom", "particles", "sweep", "grain", "beams"]);
        let mut frame = scene();
        stack.apply(&mut frame, &EffectContext::new(4, TimeBase::from_fps(30)));
    }

    #[test]
    fn out_of_range_parameters_error_explicitly() {
        for json in [
            r#"{"effects":[{"kind":"blur","amount":1.5}]}"#,
            r#"{"effects":[{"kind":"bloom","threshold":0.5,"radius":0.5,"strength":9.0}]}"#,
            r#"{"effects":[{"kind":"chroma","amount":0.5}]}"#,
            r#"{"effects":[{"kind":"displace","amount":0.5,"frequency":1.0,"speed":1.0}]}"#,
            r##"{"effects":[{"kind":"sweep","angle":0.0,"speed":1.0,"width":0.0,"color":"#fff","strength":1.0}]}"##,
            r#"{"effects":[{"kind":"grain","strength":-0.5,"density":100.0}]}"#,
            r##"{"effects":[{"kind":"particles","count":10,"seed":1,"color":"#fff","size":0.5,"speed":1.0,"drift":0.0,"twinkle":0.5}]}"##,
            r##"{"effects":[{"kind":"beams","count":1,"seed":1,"color":"#fff","origin":[2.0,0.5]}]}"##,
        ] {
            let error = StackSpec::from_json(json).unwrap().compile().err().unwrap();
            assert_eq!(error.code(), "invalid_input", "{json}");
        }
    }

    #[test]
    fn non_finite_and_malformed_inputs_are_rejected() {
        // JSON literals above cannot carry NaN/Infinity, so build the spec.
        let spec = StackSpec { effects: vec![EffectSpec::Grain { strength: f32::NAN, density: 100.0 }] };
        assert_eq!(spec.compile().err().unwrap().code(), "invalid_input");

        let error = StackSpec::from_json(r#"{"effects":[{"kind":"grain","strength":0.1,"density":100,"extra":true}]}"#);
        // Unknown fields are tolerated by serde defaults; unknown kinds are not.
        assert!(error.is_ok());
        let error = StackSpec::from_json(r#"{"effects":[{"kind":"sharpen","amount":1}]}"#).err().unwrap();
        assert_eq!(error.code(), "invalid_input");
        let error = StackSpec::from_json(r#"{"effects":[{"kind":"sweep","angle":0,"speed":1,"width":0.5,"color":"red","strength":1}]}"#).unwrap().compile().err().unwrap();
        assert_eq!(error.code(), "invalid_input");
    }

    #[test]
    fn compiled_stacks_reproduce_moments_out_of_order_and_skipped() {
        // Determinism of the whole declarative path: time-varying operators
        // (particles, sweep, grain) are pure per moment.
        let json = r##"{
            "effects": [
                { "kind": "particles", "count": 60, "seed": 31, "color": "#ffc478",
                  "size": 0.0025, "speed": -0.18, "drift": 0.04, "twinkle": 0.85 },
                { "kind": "sweep", "angle": -18.0, "speed": 0.05, "width": 0.42,
                  "color": "#96beff", "strength": 0.09 },
                { "kind": "grain", "strength": 0.12, "density": 600.0 }
            ]
        }"##;
        let stack = StackSpec::from_json(json).unwrap().compile().unwrap();
        let render = |frame_index: u32| {
            let mut frame = scene();
            stack.apply(&mut frame, &EffectContext::new(frame_index, TimeBase::from_fps(30)));
            frame.pixels
        };
        let sequential: Vec<_> = [0u32, 15, 30].iter().map(|f| render(*f)).collect();
        let mut shuffled: Vec<Vec<u8>> = vec![Vec::new(); 3];
        for (index, frame_index) in [(2, 30), (0, 0), (1, 15)] {
            shuffled[index] = render(frame_index);
        }
        for (want, got) in sequential.iter().zip(shuffled.iter()) {
            assert_eq!(want, got);
        }
        assert_ne!(sequential[0], sequential[2], "the moment actually moves the frame");
    }

    #[test]
    fn generative_wrappers_declare_domain_and_stable_names() {
        let stack = StackSpec {
            effects: vec![
                EffectSpec::Particles { count: 5, seed: 1, color: "#fff".into(), size: 0.003, speed: 0.1, drift: 0.0, twinkle: 0.5 },
                EffectSpec::Beams { count: 2, seed: 1, color: "#fff".into(), origin: None, strength: None, sway: None },
            ],
        }
        .compile()
        .unwrap();
        assert_eq!(stack.names(), vec!["particles", "beams"]);
        // Defaults land on the absorbed stage values.
        let beams = Beams::new(0, 0, Rgba([0, 0, 0, 255]));
        assert_eq!(beams.origin, [0.5, 0.28]);
        assert_eq!((beams.strength, beams.sway), (0.35, 6.0));
    }
}
