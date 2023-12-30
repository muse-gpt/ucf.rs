//! 效果合同与有序栈：每帧事实 + 无状态算子 + 前到后应用。
//!
//! 吸收自下游 AE 风格效果栈的通用合同（不含具体算子）：算子是 `(index, seed,
//! time)` 的纯函数，跨帧不携带状态，因此任意帧可独立求值、乱序、跳帧与并行取帧
//! 结果一致。时间只有一份：[`EffectContext`] 携带帧号与 [`media_types::TimeBase`]，
//! 秒数由其推导，不再有 frame/fps/seconds 三份可互相矛盾的值；画面尺寸同样只
//! 信 [`media_image::Frame`] 本身，上下文不再重复携带宽高。

use media_image::Frame;
use media_types::TimeBase;

use crate::contract::InputDomain;

/// 每帧提供给算子的事实：帧号与有理时间基。
#[derive(Clone, Copy, Debug)]
pub struct EffectContext {
    /// 序列内 0 基帧号。
    pub frame: u32,
    /// 单一时间基，秒数由它推导。
    pub time_base: TimeBase,
}

impl EffectContext {
    #[must_use]
    pub fn new(frame: u32, time_base: TimeBase) -> Self {
        Self { frame, time_base }
    }

    /// 序列起点到本帧的秒数（f32，供时变算子做相位）。
    #[must_use]
    pub fn seconds(&self) -> f32 {
        self.time_base.to_seconds(self.frame) as f32
    }
}

/// 一个效果算子。实现不得跨帧携带状态；必须声明自己消费的输入域。
pub trait Effect: Send + Sync {
    /// 稳定标识，供栈诊断与序列化合同使用。
    fn name(&self) -> &'static str;

    /// 该算子消费的缓冲语义域（见 [`crate::contract`]）。
    fn input_domain(&self) -> InputDomain;

    /// 就地应用。
    fn apply(&self, frame: &mut Frame, ctx: &EffectContext);
}

/// 有序效果列表，前到后依次应用。空栈合法，是恒等直通。
#[derive(Default)]
pub struct EffectStack {
    effects: Vec<Box<dyn Effect>>,
}

impl EffectStack {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn push(mut self, effect: impl Effect + 'static) -> Self {
        self.effects.push(Box::new(effect));
        self
    }

    /// 追加已装箱的算子，供序列化合同与生成式包装使用。
    #[must_use]
    pub fn push_boxed(mut self, effect: Box<dyn Effect>) -> Self {
        self.effects.push(effect);
        self
    }

    /// 按应用顺序列出算子名。
    #[must_use]
    pub fn names(&self) -> Vec<&'static str> {
        self.effects.iter().map(|effect| effect.name()).collect()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.effects.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.effects.is_empty()
    }

    /// 依次运行每个算子一次。
    pub fn apply(&self, frame: &mut Frame, ctx: &EffectContext) {
        for effect in &self.effects {
            effect.apply(frame, ctx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SetChannel {
        name: &'static str,
        value: u8,
    }

    impl Effect for SetChannel {
        fn name(&self) -> &'static str {
            self.name
        }
        fn input_domain(&self) -> InputDomain {
            InputDomain::DisplayRgba8
        }
        fn apply(&self, frame: &mut Frame, _ctx: &EffectContext) {
            for chunk in frame.pixels.chunks_exact_mut(4) {
                chunk[0] = self.value;
            }
        }
    }

    #[test]
    fn an_empty_stack_leaves_the_frame_alone() {
        let mut frame = Frame::new(4, 4);
        frame.pixels[0] = 77;
        EffectStack::new().apply(&mut frame, &EffectContext::new(0, TimeBase::from_fps(30)));
        assert_eq!(frame.pixels[0], 77);
        assert!(EffectStack::new().names().is_empty());
    }

    #[test]
    fn the_stack_applies_front_to_back() {
        let ctx = EffectContext::new(0, TimeBase::from_fps(30));
        let mut frame = Frame::new(2, 2);
        EffectStack::new()
            .push(SetChannel { name: "first", value: 10 })
            .push(SetChannel { name: "second", value: 20 })
            .apply(&mut frame, &ctx);
        assert_eq!(frame.pixels[0], 20, "later effect wins");

        let mut frame = Frame::new(2, 2);
        EffectStack::new()
            .push(SetChannel { name: "second", value: 20 })
            .push(SetChannel { name: "first", value: 10 })
            .apply(&mut frame, &ctx);
        assert_eq!(frame.pixels[0], 10, "order is observable");
    }

    #[test]
    fn seconds_derive_from_the_single_time_base() {
        let base = TimeBase::rational(30_000, 1_001).expect("ntsc");
        let ctx = EffectContext::new(30, base);
        assert!((ctx.seconds() - 30.0 * 1001.0 / 30_000.0).abs() < 1e-6);
        assert_eq!(EffectContext::new(15, TimeBase::from_fps(30)).seconds(), 0.5);
    }
}
