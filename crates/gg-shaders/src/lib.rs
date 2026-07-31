//! Slang compilation and reflection (§4.4). M0A scope is deliberately spike 1
//! only: prove the `shader-slang` bindings compile and reflect a module against
//! the Vulkan SDK's Slang, as a test that fails loudly the day it stops being
//! true. The offline path, hot path, and reflection→codegen arrive at M2.

use shader_slang as slang;
use slang::Downcast;

#[derive(Debug, thiserror::Error)]
pub enum ShaderError {
    #[error("slang global session could not be created (is the Vulkan SDK's Slang on PATH?)")]
    NoGlobalSession,
    #[error("slang session could not be created")]
    NoSession,
    #[error("shader path is not valid UTF-8/C-string data: {0}")]
    BadPath(String),
    #[error("slang: {0}")]
    Slang(String),
}

/// One compiled entry point: SPIR-V plus the identity the pipeline needs.
pub struct CompiledEntryPoint {
    pub name: String,
    pub spirv: Vec<u8>,
}

/// A compiled module with the reflection facts the M0A spike asserts on.
/// Real reflection→codegen (Pod structs, layout asserts) is M2 machinery.
pub struct CompiledModule {
    pub entry_points: Vec<CompiledEntryPoint>,
    pub global_parameter_count: u32,
}

/// Compile `module_name` (a `.slang` file under `search_dir`) to SPIR-V for all
/// of its `[shader(...)]` entry points, and surface the reflection counts.
pub fn compile_module(search_dir: &str, module_name: &str) -> Result<CompiledModule, ShaderError> {
    let global_session = slang::GlobalSession::new().ok_or(ShaderError::NoGlobalSession)?;

    let search_path =
        std::ffi::CString::new(search_dir).map_err(|e| ShaderError::BadPath(e.to_string()))?;
    let search_paths = [search_path.as_ptr()];

    let session_options = slang::CompilerOptions::default()
        .optimization(slang::OptimizationLevel::None)
        .matrix_layout_row(true);

    let target_desc = slang::TargetDesc::default()
        .format(slang::CompileTarget::Spirv)
        .profile(global_session.find_profile("glsl_450"));
    let targets = [target_desc];

    let session_desc = slang::SessionDesc::default()
        .targets(&targets)
        .search_paths(&search_paths)
        .options(&session_options);

    let session = global_session
        .create_session(&session_desc)
        .ok_or(ShaderError::NoSession)?;

    let module = session
        .load_module(module_name)
        .map_err(|e| ShaderError::Slang(format!("{e:?}")))?;

    let entry_points: Vec<slang::EntryPoint> = module.entry_points().collect();
    let mut components = vec![module.downcast().clone()];
    components.extend(entry_points.iter().map(|ep| ep.downcast().clone()));

    let program = session
        .create_composite_component_type(&components)
        .map_err(|e| ShaderError::Slang(format!("{e:?}")))?;
    let linked = program
        .link()
        .map_err(|e| ShaderError::Slang(format!("{e:?}")))?;

    let reflection = linked
        .layout(0)
        .map_err(|e| ShaderError::Slang(format!("{e:?}")))?;
    let global_parameter_count = reflection.parameter_count();

    let mut compiled = Vec::with_capacity(entry_points.len());
    for (index, reflected) in reflection.entry_points().enumerate() {
        let code = linked
            .entry_point_code(index as i64, 0)
            .map_err(|e| ShaderError::Slang(format!("{e:?}")))?;
        compiled.push(CompiledEntryPoint {
            name: reflected.name().to_owned(),
            spirv: code.as_slice().to_vec(),
        });
    }

    Ok(CompiledModule {
        entry_points: compiled,
        global_parameter_count,
    })
}

#[cfg(test)]
mod tests {
    // unwrap is permitted in tests (§2, Error handling row).
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    /// Spike 1 (§6 M0A): `shader-slang` compiles and reflects a trivial module
    /// against the Vulkan SDK's Slang. If this fails, the offline path falls
    /// back to `slangc` and M2 is rescoped before it starts (§8).
    #[test]
    fn spike1_slang_bindings_compile_and_reflect() {
        let dir = format!("{}/shaders", env!("CARGO_MANIFEST_DIR"));
        let module = super::compile_module(&dir, "spike.slang").unwrap();

        assert_eq!(module.entry_points.len(), 2, "vs_main + fs_main expected");
        let names: Vec<&str> = module
            .entry_points
            .iter()
            .map(|e| e.name.as_str())
            .collect();
        assert!(names.contains(&"vs_main") && names.contains(&"fs_main"));

        // The one global parameter is the `uniforms` constant buffer.
        assert_eq!(module.global_parameter_count, 1);

        for ep in &module.entry_points {
            let magic = u32::from_le_bytes([ep.spirv[0], ep.spirv[1], ep.spirv[2], ep.spirv[3]]);
            assert_eq!(magic, 0x0723_0203, "{} is not SPIR-V", ep.name);
        }
    }
}
