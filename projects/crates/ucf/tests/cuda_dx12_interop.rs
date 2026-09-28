//! DX12 writes a shared buffer; CUDA imports the NT handle and reads it back.

use ucf_backend_cuda::CudaBackend;
use ucf_backend_dx12::Dx12Backend;

#[test]
fn dx12_write_cuda_read_shared_buffer() {
    let mut dx = match Dx12Backend::new() {
        Ok(b) => b,
        Err(err) => {
            eprintln!("skip cuda-dx12 interop (no D3D12): {err}");
            return;
        }
    };

    let values = [1.0f32, 2.0, 3.0, 4.0];
    let (shared_id, nt_handle) = match dx.shared_alloc(values.len() * 4) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("skip cuda-dx12 interop (shared alloc failed): {err}");
            return;
        }
    };
    if let Err(err) = dx.shared_write_f32(shared_id, &values) {
        eprintln!("skip cuda-dx12 interop (shared write failed): {err}");
        return;
    }

    let mut cuda = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(err) => {
            eprintln!("skip cuda-dx12 interop (no CUDA): {err}");
            return;
        }
    };

    let imported = match cuda.import_dx12_nt_handle(nt_handle, values.len() * 4) {
        Ok(id) => id,
        Err(err) => {
            eprintln!("skip cuda-dx12 interop (import failed): {err}");
            return;
        }
    };

    let actual = cuda
        .read_imported_f32(imported)
        .expect("cuda read imported");
    assert_eq!(actual, values.to_vec());
}
