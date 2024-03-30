//! 声道全极点滤波器：由共振峰 (频率, 带宽) 构造级联 2 极点 IIR。
//!
//! GOLF（2024）的声道建模选择：把每个共振峰视为一对共轭极点，级联成
//! 2·n 阶全极点滤波器；先乘多项式再一次高阶滤波（比逐节串行快）。
//! 极点半径 r = e^{−π·bw/fs}、角 θ = 2π·f/fs 保证稳定（|p|<1），
//! 带宽即共振峰 Q——这是 LPC 声道的物理参数化，不是黑盒系数。
//!
//! 与旧「并联带通求和」的本质区别：并联近似丢掉了共振峰间相互作用
//! （formant interaction）与正确的相位关系；全极点级联是线性预测
//! 声道模型的标准形式。

/// 2·n 阶全极点滤波器（n 个共振峰）。系数 `a[k]` 对应 y[t] = x[t] + Σ −a[k]·y[t−k]
/// 的标准 IIR 形式（a[0]=1 省略存储）。
pub struct AllPole {
    /// 反馈系数 a[1..=order]。
    coefficients: Vec<f32>,
    state: Vec<f32>,
    order: usize,
}

impl AllPole {
    /// 由共振峰列表构造：每个 (frequency, bandwidth) 一对共轭极点。
    #[must_use]
    pub fn from_formants(formants: &[(f32, f32)], sample_rate: f32) -> Self {
        let (coefficients, order) = coefficients_for(formants, sample_rate);
        Self { coefficients, state: vec![0.0; order], order }
    }

    /// 就地更换共振峰：只换系数，保留滤波器状态。
    ///
    /// 这是「连续发声」合同的关键：换元音时若重建滤波器（状态归零），
    /// 音节接缝会出现空档或阶跃。系数连续更新时滤波器状态仍是同一条
    /// 声道的记忆，过渡只是共振峰位置在移动。阶数必须与构造时一致
    /// （同一组共振峰个数），否则 `debug_assert` 失败并保持旧系数。
    pub fn set_formants(&mut self, formants: &[(f32, f32)], sample_rate: f32) {
        let (coefficients, order) = coefficients_for(formants, sample_rate);
        debug_assert_eq!(order, self.order, "the tract order is fixed for one song");
        if order == self.order {
            self.coefficients = coefficients;
        }
    }

    /// 单样本滤波：y = x − Σ a[k]·y[t−k]。
    pub fn process(&mut self, input: f32) -> f32 {
        let mut output = input;
        for (index, coefficient) in self.coefficients.iter().enumerate() {
            output -= coefficient * self.state[index];
        }
        for i in (1..self.order).rev() {
            self.state[i] = self.state[i - 1];
        }
        if self.order > 0 {
            self.state[0] = output;
        }
        output
    }

    /// 声道归一化增益：把该组共振峰的**峰值**响应归一到 1。
    ///
    /// `H(e^{jω}) = 1 / A(e^{jω})`，峰值响应即 `1 / min|A|`；要让它等于 1，
    /// 输出应乘 `min|A|`（不是它的倒数——那会把 Q 倍增益再放大一次）。
    /// 用 DC 归一（旧实现）会把共振峰峰值放大到 Q 倍，输出幅度随元音
    /// 剧烈变化（实测相差一个数量级以上），只能靠整曲峰值归一掩盖。
    /// 峰值归一让 LF 源（每周期单位能量）的输出幅度与元音无关，响度
    /// 合同才能是固定增益 × `volume`。在 `0..π` 上取 1024 点数值求
    /// `|A(e^{jω})|` 的最小值，确定性、无平台差异。
    #[must_use]
    pub fn peak_gain(&self, _sample_rate: f32) -> f32 {
        let mut minimum = f32::INFINITY;
        for step in 0..=1024 {
            let omega = std::f32::consts::PI * step as f32 / 1024.0;
            let (mut re, mut im) = (1.0_f32, 0.0_f32);
            for (index, coefficient) in self.coefficients.iter().enumerate() {
                let phase = -omega * (index + 1) as f32;
                re += coefficient * phase.cos();
                im += coefficient * phase.sin();
            }
            minimum = minimum.min((re * re + im * im).sqrt());
        }
        minimum.max(1e-9)
    }

    /// 应用峰值归一后的增益。
    #[must_use]
    pub fn order(&self) -> usize {
        self.order
    }
}

/// 由共振峰多项式展开出全极点反馈系数（`a[1..=order]`）。
fn coefficients_for(formants: &[(f32, f32)], sample_rate: f32) -> (Vec<f32>, usize) {
    let mut polynomial = vec![1.0_f32];
    for &(frequency, bandwidth) in formants {
        let r = (-std::f32::consts::PI * bandwidth / sample_rate).exp();
        let theta = 2.0 * std::f32::consts::PI * frequency / sample_rate;
        // 1 − 2r cosθ z⁻¹ + r² z⁻²
        let section = [1.0, -2.0 * r * theta.cos(), r * r];
        let mut next = vec![0.0_f32; polynomial.len() + 2];
        for (i, a) in polynomial.iter().enumerate() {
            for (j, b) in section.iter().enumerate() {
                next[i + j] += a * b;
            }
        }
        polynomial = next;
    }
    let order = polynomial.len() - 1;
    // 归一 a[0]=1。
    let head = polynomial[0].max(1e-9);
    let coefficients = polynomial[1..].iter().map(|v| v / head).collect();
    (coefficients, order)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_stable_and_resonates_at_formant() {
        let fs = 44_100.0_f32;
        let mut filter = AllPole::from_formants(&[(1000.0, 80.0)], fs);
        // 单位脉冲激励后能量有限（稳定）。
        let mut peak = 0.0_f32;
        let mut tail_energy = 0.0_f32;
        for n in 0..fs as usize {
            let y = filter.process(if n == 0 { 1.0 } else { 0.0 });
            assert!(y.is_finite());
            peak = peak.max(y.abs());
            if n > fs as usize / 2 {
                tail_energy += y * y;
            }
        }
        assert!(peak > 0.1, "impulse response too weak");
        assert!(tail_energy < peak * peak, "ringing must decay");
    }

    #[test]
    fn changing_formants_keeps_the_filter_state() {
        // 换元音不得重建滤波器：状态连续，输出在接缝处没有空档。
        let fs = 44_100.0_f32;
        let mut filter = AllPole::from_formants(&[(800.0, 90.0), (1200.0, 90.0), (2550.0, 90.0)], fs);
        let mut warm = 0.0_f32;
        for n in 0..2000 {
            let source = (2.0 * std::f32::consts::PI * 220.0 * n as f32 / fs).sin();
            warm += filter.process(source);
        }
        assert!(warm.is_finite());
        filter.set_formants(&[(300.0, 90.0), (2300.0, 90.0), (3000.0, 90.0)], fs);
        assert_eq!(filter.order(), 6, "the order is fixed for one song");
        let first = filter.process(0.0);
        assert!(first.is_finite() && first.abs() > 0.0, "state must survive the vowel change");
        let gain = filter.peak_gain(fs);
        assert!(gain > 0.0 && gain.is_finite(), "peak normalisation must stay finite");
    }

    #[test]
    fn cascade_order_is_two_per_formant() {
        let filter = AllPole::from_formants(&[(600.0, 80.0), (1200.0, 90.0), (2500.0, 120.0)], 44_100.0);
        assert_eq!(filter.order(), 6);
    }

    #[test]
    fn sinusoid_at_formant_is_amplified_vs_off_formant() {
        let fs = 44_100.0_f32;
        let measure = |freq: f32| -> f32 {
            let mut filter = AllPole::from_formants(&[(1000.0, 80.0)], fs);
            let mut max = 0.0_f32;
            for n in 0..(fs as usize) {
                let t = n as f32 / fs;
                let y = filter.process((2.0 * std::f32::consts::PI * freq * t).sin());
                if n > fs as usize / 2 {
                    max = max.max(y.abs());
                }
            }
            max
        };
        assert!(measure(1000.0) > measure(300.0), "formant must amplify");
    }
}
