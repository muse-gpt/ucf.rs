//! 后处理合同：输入域、传递函数、alpha、边界采样、模糊半径与 tone mapping 时机。
//!
//! 本模块是全仓后处理语义的唯一事实源；`media.post` 的每个公开算子都必须声明
//! 自己的 [`InputDomain`]，混合两个域调用算子属于调用方错误，算子内部不做任何
//! 隐式转换。
//!
//! # 输入域
//!
//! - [`InputDomain::LinearHdr`]：`media_image::Framebuffer` 携带的线性 HDR 辐射
//!   （`geometry_types::LinearRgb`，f64 通道，允许 >1 能量）。这是 blur、bloom 与
//!   tone mapping 等物理算子的唯一正式路径。
//! - [`InputDomain::DisplayRgba8`]：`media_image::Frame` 的 RGBA8 显示字节。
//!   色差、位移、扫光、颗粒与显影配方是显示空间艺术效果，按合同在此域内原样
//!   操作字节，不冒充线性物理运算，也不被强行 HDR 化。
//!
//! # 传递函数
//!
//! 线性域直接在光值上运算。显示域把 8 位字节当作作者给定的显示编码值直接参与
//! 算术，算子不施加也不撤销任何 EOTF/OETF；需要进入线性域的工作由调用方在显式
//! 边界完成转换后再交进来。
//!
//! # Alpha
//!
//! RGBA8 帧一律按 straight（非预乘）alpha 解释。显示空间算子只读 RGB、只写 RGB，
//! alpha 字节原样直通；唯一例外是显示域 bloom 的亮部暂存帧内部把 alpha 置 255，
//! 该暂存帧从不写回目标 alpha。线性 HDR 帧缓冲没有 alpha 通道，编码输出恒为 255。
//!
//! # 边界采样
//!
//! 所有采样（模糊、色差、位移、重采样）使用 clamp-to-edge：越界坐标钳到最近边缘
//! 像素。不引入 wrap 或 mirror 模式。
//!
//! # 模糊半径单位
//!
//! 作者面半径是相对短边的比例（0..1），像素半径由 [`relative_radius`] 截断得到：
//! `radius = (amount × short_side × coefficient) as i32`。系数是每个算子数值合同
//! 的一部分：显示域 blur 为 0.06、显示域 bloom 暂存为 0.08、显影配方为 0.03。
//! 线性 HDR 金字塔的模糊不在此列——它是固定 3×3 盒，半径合同由金字塔层级表达。
//!
//! # Tone mapping
//!
//! ACES filmic 只在显式输出边界执行一次（[`crate::apply_tone_map`]，或
//! [`crate::to_rgba8`] 对 `LinearHdr` 编码的自动单次映射）。bloom 金字塔全程在
//! 线性域加回、绝不截断；显示空间算子从不执行 tone mapping。
//!
//! # CPU 与 UCF 双路径
//!
//! parity 合同只覆盖线性 HDR 路径：tone map 与亮部提取逐通道 1e-9，多尺度 bloom
//! 因程序中间量为 f32 放宽到 1e-5。显示空间算子本轮只有 CPU 实现，没有 UCF 程序；
//! 硬件消费者不得对它们宣称硬件执行，需要时显式返回 unsupported。

use media_image::Frame;

/// 算子的输入域标注：每个公开算子必须声明自己消费哪一种缓冲语义。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputDomain {
    /// 线性 HDR f64 辐射（`Framebuffer`），物理算子路径。
    LinearHdr,
    /// 显示空间 RGBA8 字节（`Frame`），艺术效果与显影配方路径。
    DisplayRgba8,
}

/// 短边像素数（至少 1），相对半径合同的基准。
#[must_use]
pub fn short_side(frame: &Frame) -> f32 {
    frame.width.min(frame.height).max(1) as f32
}

/// 把相对短边的作者半径换算为像素半径（向零截断）。
///
/// `amount` 必须由调用算子先按自己的参数域钳制；`coefficient` 是该算子的合同
/// 系数（见模块文档「模糊半径单位」）。
#[must_use]
pub fn relative_radius(amount: f32, frame: &Frame, coefficient: f32) -> i32 {
    (amount * short_side(frame) * coefficient) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_side_floor_and_radius_truncation() {
        let frame = Frame::new(100, 40);
        assert_eq!(short_side(&frame), 40.0);
        assert_eq!(relative_radius(0.5, &frame, 0.06), 1);
        assert_eq!(relative_radius(1.0, &frame, 0.03), 1);
        assert_eq!(relative_radius(0.0, &frame, 0.06), 0);
        let degenerate = Frame::new(0, 0);
        assert_eq!(short_side(&degenerate), 1.0);
    }
}
