// Build script: compile the CUDA kernel to PTX when the `cuda` feature is
// enabled. The PTX file is dropped into OUT_DIR so the runtime can pick it
// up via `include_bytes!`.
//
// We intentionally compile against SM_61 (Pascal — GTX 1060 / 1070 / 1080)
// as the lowest target, then list newer architectures so a single binary
// JIT-loads the best variant on any supported card. PTX is forward
// compatible: an SM_61 PTX module will JIT on Volta/Turing/Ampere/Ada at
// the cost of one-time JIT compilation per host process.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/cuda/spectral.cu");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUDA");
    // Environment knobs that affect whether nvcc can locate its host
    // compiler. If the user fixes one of these between builds we want to
    // re-run nvcc rather than reusing the stale stub PTX.
    println!("cargo:rerun-if-env-changed=PATH");
    println!("cargo:rerun-if-env-changed=CUDAHOSTCXX");
    println!("cargo:rerun-if-env-changed=VCINSTALLDIR");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    #[cfg(feature = "cuda")]
    cuda_build::build_kernel();
}

#[cfg(feature = "cuda")]
mod cuda_build {
    use std::path::PathBuf;
    use std::process::Command;

    pub fn build_kernel() {
        let cuda_root = find_cuda_helper::find_cuda_root();
        let cuda_paths: Vec<PathBuf> = cuda_root.into_iter().collect();
        let nvcc = locate_nvcc(&cuda_paths);

        let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
        let out_dir = PathBuf::from(out_dir);

        // Spectral kernel powers the alignment prefilter — the only
        // remaining CUDA kernel after GPU seeding was reverted (see
        // src/cuda/mod.rs doc and docs/GPU_SEEDING_PLAN.md for the
        // post-mortem on the abandoned mph_lookup / bucket_scan kernels).
        compile_one(
            &nvcc,
            &PathBuf::from("src/cuda/spectral.cu"),
            &out_dir.join("spectral.ptx"),
            "KIRA_SPECTRAL_PTX",
        );
    }

    fn compile_one(
        nvcc: &PathBuf,
        src: &PathBuf,
        ptx_path: &PathBuf,
        env_var: &str,
    ) {
        let mut cmd = Command::new(nvcc);
        cmd.args([
            "-ptx",
            "-O3",
            "-use_fast_math",
            "--gpu-architecture=compute_61",
            "-std=c++14",
        ]);
        if cfg!(target_os = "windows") {
            if let Some(ccbin) = locate_msvc_cl_exe() {
                cmd.arg("-ccbin").arg(ccbin);
            }
            if std::env::var_os("KIRA_NO_UNSUPPORTED_COMPILER").is_none() {
                cmd.arg("-allow-unsupported-compiler");
            }
        } else {
            cmd.args(["-Xcompiler", "-fPIC"]);
        }
        let output = cmd.arg(src).arg("-o").arg(ptx_path).output();

        match output {
            Ok(o) if o.status.success() => {
                println!("cargo:rustc-env={}={}", env_var, ptx_path.display());
                eprintln!(
                    "kira-ls-aligner: compiled CUDA kernel {} → {}",
                    src.display(),
                    ptx_path.display()
                );
            }
            Ok(o) => {
                let stdout = String::from_utf8_lossy(&o.stdout);
                let stderr = String::from_utf8_lossy(&o.stderr);
                emit_stub_ptx_and_warn(
                    ptx_path,
                    env_var,
                    &format!(
                        "nvcc failed (exit {:?}) compiling {}",
                        o.status.code(),
                        src.display()
                    ),
                );
                for line in stdout.lines().chain(stderr.lines()) {
                    let line = line.trim_end();
                    if !line.is_empty() {
                        println!("cargo:warning=  nvcc: {line}");
                    }
                }
                let hint = if cfg!(target_os = "windows") {
                    "If nvcc complains about cl.exe: run cargo from an x64 Native \
                     Tools Command Prompt for VS, OR pass --ccbin to nvcc, OR set \
                     CUDAHOSTCXX env var to the cl.exe full path."
                } else {
                    "Ensure a host C++ compiler (gcc/clang) matching your CUDA \
                     toolkit's supported versions is installed and on PATH."
                };
                println!("cargo:warning=  hint: {hint}");
            }
            Err(e) => {
                emit_stub_ptx_and_warn(
                    ptx_path,
                    env_var,
                    &format!(
                        "Failed to invoke nvcc at {}: {}. \
                         Install the CUDA toolkit (>= 11.0) and ensure nvcc is on PATH \
                         or set CUDA_PATH / CUDA_HOME env vars.",
                        nvcc.display(),
                        e
                    ),
                );
            }
        }
    }

    /// Fallback: when nvcc / a host compiler is unavailable, write a
    /// PTX file that the runtime will fail to load — but the Rust code
    /// still compiles. We emit a `cargo:warning=` so the failure is
    /// visible in build output, *without* aborting the build entirely.
    /// This lets developers `cargo check --features cuda` to validate the
    /// Rust side on machines that don't ship MSVC + CUDA.
    fn emit_stub_ptx_and_warn(ptx_path: &std::path::Path, env_var: &str, message: &str) {
        let stub = "// stub PTX generated because nvcc was unavailable.\n\
                    // Runtime will fail with `CudaError::Driver` if loaded.\n";
        let _ = std::fs::write(ptx_path, stub);
        println!(
            "cargo:warning=kira-ls-aligner CUDA build: {} \
             Generated a stub PTX so the Rust side still compiles; \
             the binary will refuse to run --gpu-server at runtime.",
            message
        );
        println!("cargo:rustc-env={}={}", env_var, ptx_path.display());
        // Mirror the legacy stub flag for backwards compatibility; the
        // mph_lookup path also gets its own flag.
        let stub_flag = format!("{}_STUB", env_var);
        println!("cargo:rustc-env={}=1", stub_flag);
    }

    /// Find the directory containing MSVC `cl.exe`, suitable for passing as
    /// `nvcc -ccbin <dir>`. nvcc on Windows requires a host compiler and
    /// only finds it via PATH by default — but cargo subprocesses don't
    /// always inherit the same PATH the user's shell does, especially when
    /// the VS environment was activated by a parent shell that cargo
    /// doesn't pass through. We probe four sources in order:
    ///
    ///   1. `CUDAHOSTCXX` env var (nvcc's official escape hatch)
    ///   2. `cl.exe` discoverable on PATH
    ///   3. `VCINSTALLDIR` env var pointing at a VC toolset
    ///   4. `vswhere.exe` -> latest VS install -> VC\Tools\MSVC\<ver>\bin\Hostx64\x64
    ///
    /// Returns `None` if none of these find a cl.exe. In that case we leave
    /// the `-ccbin` arg off and let nvcc emit its own (unfortunately
    /// unhelpful) error.
    #[cfg(target_os = "windows")]
    fn locate_msvc_cl_exe() -> Option<PathBuf> {
        // 1. Honour the user-set escape hatch.
        if let Ok(p) = std::env::var("CUDAHOSTCXX") {
            let pb = PathBuf::from(&p);
            if pb.is_file() {
                if let Some(parent) = pb.parent() {
                    return Some(parent.to_path_buf());
                }
            }
        }
        // 2. cl.exe on the inherited PATH.
        if let Ok(p) = which("cl.exe") {
            if let Some(parent) = p.parent() {
                return Some(parent.to_path_buf());
            }
        }
        // 3. VCINSTALLDIR (set by vcvars*.bat).
        if let Ok(vc) = std::env::var("VCINSTALLDIR") {
            let candidate = PathBuf::from(vc).join("bin").join("Hostx64").join("x64");
            if candidate.join("cl.exe").is_file() {
                return Some(candidate);
            }
        }
        // 4. vswhere — Microsoft's blessed VS-locator. Usually at the
        //    fixed path below regardless of which VS edition is installed.
        let vswhere = PathBuf::from(
            "C:\\Program Files (x86)\\Microsoft Visual Studio\\Installer\\vswhere.exe",
        );
        if vswhere.is_file() {
            let out = Command::new(&vswhere)
                .args([
                    "-latest",
                    "-products",
                    "*",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property",
                    "installationPath",
                ])
                .output();
            if let Ok(out) = out {
                if out.status.success() {
                    let vs_path = String::from_utf8_lossy(&out.stdout).trim().to_string();
                    if !vs_path.is_empty() {
                        let msvc_root = PathBuf::from(&vs_path).join("VC\\Tools\\MSVC");
                        // Pick the newest version subdir.
                        if let Ok(entries) = std::fs::read_dir(&msvc_root) {
                            let mut versions: Vec<PathBuf> =
                                entries.filter_map(|e| e.ok()).map(|e| e.path()).collect();
                            versions.sort();
                            for v in versions.iter().rev() {
                                let candidate =
                                    v.join("bin").join("Hostx64").join("x64");
                                if candidate.join("cl.exe").is_file() {
                                    return Some(candidate);
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    #[cfg(not(target_os = "windows"))]
    fn locate_msvc_cl_exe() -> Option<PathBuf> {
        None
    }

    /// Find nvcc — prefer PATH, then CUDA_PATH/CUDA_HOME bin/, then
    /// the heuristics from find_cuda_helper.
    fn locate_nvcc(cuda_paths: &[PathBuf]) -> PathBuf {
        if let Ok(p) = which("nvcc") {
            return p;
        }
        for base in cuda_paths {
            let candidate = base.join("bin").join(nvcc_exe());
            if candidate.exists() {
                return candidate;
            }
        }
        // Fall back: trust PATH and hope nvcc is there. The Command call
        // will fail with a useful error if not.
        PathBuf::from("nvcc")
    }

    fn nvcc_exe() -> &'static str {
        if cfg!(windows) { "nvcc.exe" } else { "nvcc" }
    }

    /// Minimal stand-in for the `which` crate to avoid an extra dep.
    fn which(name: &str) -> Result<PathBuf, ()> {
        let exe_name = if cfg!(windows) && !name.ends_with(".exe") {
            format!("{name}.exe")
        } else {
            name.to_string()
        };
        let path_var = std::env::var_os("PATH").ok_or(())?;
        for dir in std::env::split_paths(&path_var) {
            let candidate = dir.join(&exe_name);
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
        Err(())
    }
}
