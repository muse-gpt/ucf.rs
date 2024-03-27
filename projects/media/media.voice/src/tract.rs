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
        Self { coefficients, state: vec![0.0; order], order }
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

    /// 增益（DC 归一）：使 0 Hz 响应为 1，共振峰间相对幅度由极点位置决定。
    #[must_use]
    pub fn dc_gain(&self) -> f32 {
        let sum: f32 = self.coefficients.iter().sum();
        1.0 / (1.0 - sum).abs().max(1e-6)
    }

    #[must_use]
    pub fn order(&self) -> usize {
        self.order
    }
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
