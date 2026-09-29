use serde::{Deserialize, Serialize};

/// UCF internal shader program. Backends never consume HLSL or CUDA C source.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShaderProgram {
    pub entry: String,
    pub ops: Vec<ShaderOp>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ShaderOp {
    /// Parallel fill: out[i] = value for i in 0..count
    IotaFill { value: f32 },
    /// Naive row-major matmul: C[m,n] = A[m,k] * B[k,n]
    MatMul,
    /// Host/device copy without a kernel body
    MemCopy,
    /// RGBA8 spatial denoise: horizontal RGB mean with left neighbor, A passthrough
    Rgba8Denoise,
}

impl ShaderProgram {
    pub fn dispatch_fill(entry: impl Into<String>, value: f32) -> Self {
        Self {
            entry: entry.into(),
            ops: vec![ShaderOp::IotaFill { value }],
        }
    }

    pub fn matmul(entry: impl Into<String>) -> Self {
        Self {
            entry: entry.into(),
            ops: vec![ShaderOp::MatMul],
        }
    }

    pub fn mem_copy(entry: impl Into<String>) -> Self {
        Self {
            entry: entry.into(),
            ops: vec![ShaderOp::MemCopy],
        }
    }

    pub fn rgba8_denoise(entry: impl Into<String>) -> Self {
        Self {
            entry: entry.into(),
            ops: vec![ShaderOp::Rgba8Denoise],
        }
    }
}
