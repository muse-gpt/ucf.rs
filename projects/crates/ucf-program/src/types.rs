use serde::{Deserialize, Serialize};

/// 标量类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScalarTy {
    Bool,
    I32,
    U32,
    F32,
    F64,
}

/// 程序类型（L0/L1 子集）。
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Ty {
    Scalar(ScalarTy),
    Vec2,
    Vec3,
    Vec4,
    Array {
        elem: Box<Ty>,
        len: u32,
    },
    Struct {
        name: String,
        fields: Vec<(String, Ty)>,
    },
}

impl Ty {
    pub fn f32() -> Self {
        Self::Scalar(ScalarTy::F32)
    }

    pub fn f64() -> Self {
        Self::Scalar(ScalarTy::F64)
    }

    pub fn vec3() -> Self {
        Self::Vec3
    }
}
