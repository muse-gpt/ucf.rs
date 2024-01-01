//! 生成式算子接入效果栈：粒子与光束的求值归 `media.particles`，本模块只做
//! 栈合同适配。
//!
//! 适配方向是单向的——`media.post` 依赖 `media.particles`，叶子 crate 不认识
//! 栈合同；`seconds` 由 [`EffectContext::seconds`] 从单一有理时间基派生后传入。

use media_image::Frame;
use media_particles::{Beams, Particles};

use crate::contract::InputDomain;
use crate::effect::{Effect, EffectContext};

/// 把 [`Particles`] 适配为栈算子，稳定名 `particles`。
pub struct ParticlesEffect(pub Particles);

impl Effect for ParticlesEffect {
    fn name(&self) -> &'static str {
        "particles"
    }

    fn input_domain(&self) -> InputDomain {
        InputDomain::DisplayRgba8
    }

    fn apply(&self, frame: &mut Frame, ctx: &EffectContext) {
        self.0.apply(frame, ctx.seconds());
    }
}

/// 把 [`Beams`] 适配为栈算子，稳定名 `beams`。
pub struct BeamsEffect(pub Beams);

impl Effect for BeamsEffect {
    fn name(&self) -> &'static str {
        "beams"
    }

    fn input_domain(&self) -> InputDomain {
        InputDomain::DisplayRgba8
    }

    fn apply(&self, frame: &mut Frame, ctx: &EffectContext) {
        self.0.apply(frame, ctx.seconds());
    }
}
