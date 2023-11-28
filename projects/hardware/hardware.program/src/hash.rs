use crate::program::Program;

/// 稳定程序指纹（JSON + FNV-1a 64）。
pub fn program_hash(program: &Program) -> u64 {
    let json = serde_json::to_string(program).expect("program serializes");
    fnv1a64(json.as_bytes())
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x00000100000001B3;
    let mut hash = OFFSET;
    for b in bytes {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}
