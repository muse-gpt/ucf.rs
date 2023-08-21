/**
 * 将 `ucf-wasm` release 产物拷到本包根 `ucf_wasm_bg.wasm`。
 */
import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const pkgRoot = resolve(here, "..");
const repoRoot = resolve(pkgRoot, "../../../..");
const src = join(repoRoot, "target/wasm32-unknown-unknown/release/ucf_wasm.wasm");
const dest = join(pkgRoot, "ucf_wasm_bg.wasm");

if (!existsSync(src)) {
    console.error(
        `未找到 ${src}\n请先：rustup target add wasm32-unknown-unknown\n然后：cargo build -p ucf-wasm --target wasm32-unknown-unknown --release`,
    );
    process.exit(1);
}

mkdirSync(pkgRoot, { recursive: true });
copyFileSync(src, dest);
console.log(`已拷贝 → ${dest}`);
