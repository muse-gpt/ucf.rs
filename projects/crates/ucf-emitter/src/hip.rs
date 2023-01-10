use ucf_types::{ShaderOp, ShaderProgram};

/// Emit HIP C++ source for `hiprtc` (UTF-8, NUL-terminated).
///
/// Authors never write kernels; the source lives only inside this emitter.
pub fn emit_hip(program: &ShaderProgram) -> Result<Vec<u8>, String> {
    for op in &program.ops {
        match op {
            ShaderOp::MemCopy => continue,
            ShaderOp::IotaFill { value } => {
                let text = hip_fill(&program.entry, *value);
                let mut bytes = text.into_bytes();
                bytes.push(0);
                return Ok(bytes);
            }
            ShaderOp::MatMul => {
                let text = hip_matmul(&program.entry);
                let mut bytes = text.into_bytes();
                bytes.push(0);
                return Ok(bytes);
            }
        }
    }
    Err("program has no emittable HIP ops".into())
}

fn hip_fill(entry: &str, value: f32) -> String {
    format!(
        r#"
extern "C" __global__ void {entry}(float* out, unsigned int count) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    out[i] = {value:?};
}}
"#
    )
}

fn hip_matmul(entry: &str) -> String {
    format!(
        r#"
extern "C" __global__ void {entry}(
    const float* a,
    const float* b,
    float* out,
    unsigned int m,
    unsigned int n,
    unsigned int k
) {{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * n) return;
    unsigned int row = idx / n;
    unsigned int col = idx % n;
    float acc = 0.0f;
    for (unsigned int i = 0; i < k; ++i) {{
        acc += a[row * k + i] * b[i * n + col];
    }}
    out[idx] = acc;
}}
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ucf_types::ShaderProgram;

    #[test]
    fn emit_fill_and_matmul_hip_source() {
        let fill = emit_hip(&ShaderProgram::dispatch_fill("ucf_fill", 1.0)).expect("fill");
        assert!(fill.ends_with(&[0]));
        let text = std::str::from_utf8(&fill[..fill.len() - 1]).unwrap();
        assert!(text.contains("__global__"));
        assert!(text.contains("ucf_fill"));
        let matmul = emit_hip(&ShaderProgram::matmul("ucf_matmul")).expect("matmul");
        assert!(matmul.ends_with(&[0]));
    }
}
