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
            ShaderOp::Rgba8Denoise => {
                let text = hip_rgba8_denoise(&program.entry);
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

fn hip_rgba8_denoise(entry: &str) -> String {
    format!(
        r#"
extern "C" __global__ void {entry}(
    const unsigned char* src,
    unsigned char* out,
    unsigned int width,
    unsigned int height
) {{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;
    unsigned int x = idx % width;
    unsigned int left = (x == 0) ? idx : (idx - 1);
    unsigned int base = idx * 4;
    unsigned int lbase = left * 4;
    out[base + 0] = (unsigned char)(((unsigned int)src[base + 0] + (unsigned int)src[lbase + 0]) >> 1);
    out[base + 1] = (unsigned char)(((unsigned int)src[base + 1] + (unsigned int)src[lbase + 1]) >> 1);
    out[base + 2] = (unsigned char)(((unsigned int)src[base + 2] + (unsigned int)src[lbase + 2]) >> 1);
    out[base + 3] = src[base + 3];
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
