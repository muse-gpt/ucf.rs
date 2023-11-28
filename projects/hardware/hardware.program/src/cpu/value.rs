use std::fmt;

/// 运行时值（F32 语义为主，Vec3 用 f64 累加）。
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Bool(bool),
    U32(u32),
    F32(f32),
    F64(f64),
    Vec3([f64; 3]),
}

impl Value {
    pub fn as_f64(&self) -> Result<f64, String> {
        match self {
            Self::F32(v) => Ok(*v as f64),
            Self::F64(v) => Ok(*v),
            Self::U32(v) => Ok(*v as f64),
            Self::Bool(v) => Ok(if *v { 1.0 } else { 0.0 }),
            Self::Vec3(_) => Err("vec3".into()),
        }
    }

    pub fn as_bool(&self) -> Result<bool, String> {
        match self {
            Self::Bool(v) => Ok(*v),
            Self::F32(v) => Ok(*v != 0.0),
            Self::U32(v) => Ok(*v != 0),
            _ => Err("not bool".into()),
        }
    }

    pub fn as_vec3(&self) -> Result<[f64; 3], String> {
        match self {
            Self::Vec3(v) => Ok(*v),
            _ => Err("not vec3".into()),
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Bool(_) => "bool",
            Self::U32(_) => "u32",
            Self::F32(_) => "f32",
            Self::F64(_) => "f64",
            Self::Vec3(_) => "vec3",
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bool(v) => write!(f, "{}", v),
            Self::U32(v) => write!(f, "{}", v),
            Self::F32(v) => write!(f, "{}", v),
            Self::F64(v) => write!(f, "{}", v),
            Self::Vec3([x, y, z]) => write!(f, "vec3({}, {}, {})", x, y, z),
        }
    }
}
