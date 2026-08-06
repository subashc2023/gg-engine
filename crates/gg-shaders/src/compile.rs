//! In-process compilation + reflection (§4.4). One call turns a `.slang`
//! module into per-entry-point SPIR-V plus the push-constant layout facts that
//! codegen freezes into Rust. Anything reflection reports that codegen cannot
//! represent is a loud [`ShaderError::Unsupported`], never a silent guess.

use crate::ShaderError;
use shader_slang as slang;
use slang::Downcast;

/// Pipeline stage of an entry point. Grows with its consumers (compute lands
/// with its first compute pass).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// `[shader("vertex")]`
    Vertex,
    /// `[shader("fragment")]`
    Fragment,
    /// `[shader("compute")]`
    Compute,
}

/// One compiled entry point: SPIR-V plus the identity a pipeline needs.
pub struct CompiledEntryPoint {
    /// Entry point function name in the Slang source.
    pub name: String,
    /// The `OpEntryPoint` name inside the SPIR-V — what
    /// `VkPipelineShaderStageCreateInfo::pName` must be.
    ///
    /// Read back out of the emitted blob rather than assumed. Slang renames
    /// entry points to the Vulkan-conventional `main`, but only when a blob
    /// holds exactly one — a multi-entry blob keeps the source names, and
    /// `-fvk-use-entrypoint-name` changes the rule outright. Hardcoding `main`
    /// would make that a silent pipeline-creation failure on the day either
    /// changes; the driver's error for a wrong `pName` names nothing useful.
    pub spirv_entry: String,
    /// Which stage the entry point targets.
    pub stage: Stage,
    /// The SPIR-V blob.
    pub spirv: Vec<u8>,
}

/// Scalar component type codegen supports. 16-bit joins with a consumer that
/// needs it; `uint64_t` arrived with M4A's buffer device addresses (§4.3),
/// which are the one 64-bit quantity a push-constant block carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scalar {
    /// `float`
    F32,
    /// `uint`
    U32,
    /// `int`
    I32,
    /// `uint64_t` — a buffer device address.
    U64,
}

/// A field's shape, restricted to what maps 1:1 onto a `repr(C)` Rust type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FieldType {
    /// A single scalar.
    Scalar(Scalar),
    /// `floatN` / `uintN` / `intN`, N in 2..=4.
    Vector(Scalar, u32),
    /// `floatRxC`, row-major (the session forces row-major layout). Only
    /// shapes whose std430 row stride equals the C row stride are accepted.
    Matrix {
        /// Row count.
        rows: u32,
        /// Column count.
        cols: u32,
    },
}

impl FieldType {
    /// The Rust type this field generates.
    pub fn rust_type(&self) -> String {
        match self {
            FieldType::Scalar(s) => s.rust_name().to_string(),
            FieldType::Vector(s, n) => format!("[{}; {n}]", s.rust_name()),
            FieldType::Matrix { rows, cols } => format!("[[f32; {cols}]; {rows}]"),
        }
    }

    /// Size of the generated Rust type in bytes.
    pub fn cpu_size(&self) -> usize {
        match self {
            FieldType::Scalar(s) => s.size(),
            FieldType::Vector(s, n) => s.size() * *n as usize,
            FieldType::Matrix { rows, cols } => 4 * (*rows as usize) * (*cols as usize),
        }
    }
}

impl Scalar {
    fn rust_name(self) -> &'static str {
        match self {
            Scalar::F32 => "f32",
            Scalar::U32 => "u32",
            Scalar::I32 => "i32",
            Scalar::U64 => "u64",
        }
    }

    fn size(self) -> usize {
        match self {
            Scalar::F32 | Scalar::U32 | Scalar::I32 => 4,
            Scalar::U64 => 8,
        }
    }
}

/// One reflected struct field.
#[derive(Clone, Debug)]
pub struct StructField {
    /// Field name in the shader.
    pub name: String,
    /// Byte offset within the block (std430, as reflected).
    pub offset: usize,
    /// Reflected size in bytes. Must equal the generated Rust field's size or
    /// compilation of the module fails with [`ShaderError::Unsupported`].
    pub size: usize,
    /// The field's shape.
    pub ty: FieldType,
}

/// A reflected uniform block layout (M2: the push-constant block).
#[derive(Clone, Debug)]
pub struct StructLayout {
    /// The shader-side struct name; the generated Rust struct reuses it.
    pub name: String,
    /// Total block size in bytes (std430, as reflected).
    pub size: usize,
    /// Fields in declaration order.
    pub fields: Vec<StructField>,
}

/// A compiled module with the facts M2's consumers need.
pub struct CompiledModule {
    /// All `[shader(...)]` entry points, compiled.
    pub entry_points: Vec<CompiledEntryPoint>,
    /// The push-constant block (Slang's `vk` `push_constant` attribute), if
    /// the module declares one. At most one — a second is
    /// [`ShaderError::Unsupported`].
    pub push_constants: Option<StructLayout>,
    /// Global parameter count, as reflected (spike 1's canary assertion).
    pub global_parameter_count: u32,
}

/// The version tag of the Slang actually linked into this process.
///
/// This is a *runtime* fact, not a lockfile one, and that is the whole point:
/// `shader-slang` pins no Slang version — its `-sys` crate runs bindgen against
/// whatever `slang.h` the build machine has and links the installed `slang`
/// dylib, so the compiler is a property of the machine. Two hosts building the
/// same commit can therefore reflect and emit through different compilers,
/// while `xtask shaders --check` requires the *generated Rust* to be
/// byte-identical across them. Pinning what this returns is what makes that
/// requirement rest on a controlled input instead of on luck.
pub fn slang_build_tag() -> Result<String, ShaderError> {
    let global_session = slang::GlobalSession::new().ok_or(ShaderError::NoGlobalSession)?;
    Ok(global_session.build_tag_string().to_owned())
}

/// Compile `module_name` (a `.slang` file under `search_dir`) to SPIR-V for
/// all of its entry points and reflect the push-constant layout.
pub fn compile_module(search_dir: &str, module_name: &str) -> Result<CompiledModule, ShaderError> {
    let global_session = slang::GlobalSession::new().ok_or(ShaderError::NoGlobalSession)?;

    // `search_path` must outlive `session_desc`: `SessionDesc::search_paths` takes
    // raw pointers and ties no lifetime to them (upstream slang-rs #31, open —
    // and unsound by signature, not by our use). Binding the `CString` to a
    // named local is the whole guard; folding it into the array expression makes
    // it a temporary, and the dangle compiles.
    let search_path =
        std::ffi::CString::new(search_dir).map_err(|e| ShaderError::BadPath(e.to_string()))?;
    let search_paths = [search_path.as_ptr()];

    // Optimization stays off in both paths for now: hot reload wants the
    // latency, drivers re-optimize anyway, and one configuration means the
    // embedded SPIR-V is the SPIR-V the hot path produced.
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
        .map_err(|e| ShaderError::Slang(e.to_string()))?;

    let entry_points: Vec<slang::EntryPoint> = module.entry_points().collect();
    let mut components = vec![module.downcast().clone()];
    components.extend(entry_points.iter().map(|ep| ep.downcast().clone()));

    let program = session
        .create_composite_component_type(&components)
        .map_err(|e| ShaderError::Slang(e.to_string()))?;
    let linked = program
        .link()
        .map_err(|e| ShaderError::Slang(e.to_string()))?;

    let reflection = linked
        .layout(0)
        .map_err(|e| ShaderError::Slang(e.to_string()))?;
    let global_parameter_count = reflection.parameter_count();
    let push_constants = reflect_push_constants(reflection)?;

    let mut compiled = Vec::with_capacity(entry_points.len());
    for (index, reflected) in reflection.entry_points().enumerate() {
        let stage = match reflected.stage() {
            slang::Stage::Vertex => Stage::Vertex,
            slang::Stage::Fragment => Stage::Fragment,
            slang::Stage::Compute => Stage::Compute,
            other => {
                return Err(ShaderError::Unsupported(format!(
                    "entry point `{}` targets stage {other:?} — M2 covers vertex/fragment/compute",
                    reflected.name()
                )));
            }
        };
        let code = linked
            .entry_point_code(index as i64, 0)
            .map_err(|e| ShaderError::Slang(e.to_string()))?;
        let spirv = code.as_slice().to_vec();
        let spirv_entry = spirv_entry_name(&spirv, reflected.name())?;
        compiled.push(CompiledEntryPoint {
            name: reflected.name().to_owned(),
            spirv_entry,
            stage,
            spirv,
        });
    }

    Ok(CompiledModule {
        entry_points: compiled,
        push_constants,
        global_parameter_count,
    })
}

/// Read the single `OpEntryPoint`'s name out of a SPIR-V blob.
///
/// This is the fact `spirv_entry` used to assume. The parse is trivial — a
/// fixed 5-word header, then instructions whose first word packs
/// `(word_count << 16) | opcode` — and `OpEntryPoint` (opcode 15) lays out as
/// execution model, entry id, then a null-terminated, word-padded literal
/// string. Slang emits one entry point per blob, so more than one here means
/// the compilation shape changed underneath us and the generated `*_ENTRY`
/// constant can no longer be a single value; that is an error, not a guess.
fn spirv_entry_name(spirv: &[u8], source_name: &str) -> Result<String, ShaderError> {
    const MAGIC: u32 = 0x0723_0203;
    const OP_ENTRY_POINT: u32 = 15;
    const HEADER_WORDS: usize = 5;
    /// Execution model + entry id + at least one word of name.
    const MIN_ENTRY_POINT_WORDS: usize = 4;

    let malformed = |why: &str| {
        ShaderError::Unsupported(format!(
            "entry point `{source_name}`: emitted SPIR-V is malformed ({why})"
        ))
    };

    if !spirv.len().is_multiple_of(4) || spirv.len() < HEADER_WORDS * 4 {
        return Err(malformed(
            "shorter than a SPIR-V header, or not word-aligned",
        ));
    }
    let word = |i: usize| -> u32 {
        u32::from_le_bytes([
            spirv[i * 4],
            spirv[i * 4 + 1],
            spirv[i * 4 + 2],
            spirv[i * 4 + 3],
        ])
    };
    if word(0) != MAGIC {
        // Includes the byte-swapped magic: every target in the Determinism
        // Contract is little-endian, so a big-endian module is a surprise
        // worth failing on rather than silently byte-swapping around.
        return Err(malformed("bad magic word"));
    }

    let words = spirv.len() / 4;
    let mut found: Option<String> = None;
    let mut i = HEADER_WORDS;
    while i < words {
        let count = (word(i) >> 16) as usize;
        let opcode = word(i) & 0xFFFF;
        if count == 0 || i + count > words {
            return Err(malformed("instruction word count runs past the blob"));
        }
        if opcode == OP_ENTRY_POINT {
            if count < MIN_ENTRY_POINT_WORDS {
                return Err(malformed("OpEntryPoint too short to carry a name"));
            }
            let bytes = &spirv[(i + 3) * 4..(i + count) * 4];
            let nul = bytes
                .iter()
                .position(|b| *b == 0)
                .ok_or_else(|| malformed("OpEntryPoint name is not null-terminated"))?;
            let name = std::str::from_utf8(&bytes[..nul])
                .map_err(|_| malformed("OpEntryPoint name is not UTF-8"))?;
            if found.is_some() {
                return Err(ShaderError::Unsupported(format!(
                    "entry point `{source_name}`: emitted SPIR-V declares more than one \
                     OpEntryPoint, so a single generated *_ENTRY constant cannot name it — \
                     Slang's one-entry-point-per-blob shape is what the codegen assumes"
                )));
            }
            found = Some(name.to_owned());
        }
        i += count;
    }

    found.ok_or_else(|| malformed("no OpEntryPoint found"))
}

/// Walk the global parameters for the push-constant block and
/// extract its layout. The reflected offsets/sizes are the ground truth the
/// generated assertions freeze (§4.4).
fn reflect_push_constants(
    reflection: &slang::reflection::Shader,
) -> Result<Option<StructLayout>, ShaderError> {
    let mut found: Option<StructLayout> = None;
    for param in reflection.parameters() {
        let category = param.category();
        // A layout spanning several categories reflects as Mixed, which the
        // PushConstantBuffer filter below would skip *silently* — leaving the
        // block un-reflected, no struct generated, and the failure surfacing
        // as a confusing compile error in the consumer. §1.10: fail here, loudly.
        if category == slang::ParameterCategory::Mixed {
            return Err(ShaderError::Unsupported(format!(
                "global parameter `{}` reflects as ParameterCategory::Mixed — its layout spans \
                 several categories, so push-constant offsets cannot be read off one category; \
                 give the push-constant block its own parameter",
                param.name().unwrap_or("<anonymous>")
            )));
        }
        if category != slang::ParameterCategory::PushConstantBuffer {
            continue;
        }
        if found.is_some() {
            return Err(ShaderError::Unsupported(
                "more than one push-constant block — Vulkan gives one range, keep one block".into(),
            ));
        }
        // The parameter is ConstantBuffer<T>; element_type_layout is T.
        //
        // The kind check is load-bearing, not decorative: on shader-slang 0.1.0
        // `element_type_layout()` builds a `&TypeLayout` directly from the C++
        // return value with no null check (upstream issue #22, fixed only on an
        // unreleased main), and Slang returns null for a type that has no
        // element. Establishing the type *is* a container first is what keeps
        // that call from constructing a null reference.
        let param_layout = param.type_layout();
        let kind = param_layout.kind();
        if !matches!(
            kind,
            slang::TypeKind::ConstantBuffer | slang::TypeKind::ParameterBlock
        ) {
            return Err(ShaderError::Unsupported(format!(
                "push-constant parameter `{}` reflects as {kind:?}, not a constant buffer — \
                 declare it as a `ConstantBuffer<T>` carrying Slang's `vk` `push_constant` \
                 attribute",
                param.name().unwrap_or("<anonymous>")
            )));
        }
        let block = param_layout.element_type_layout();
        let name = block
            .name()
            .unwrap_or("PushConstants") // anonymous block: give it a stable name
            .to_string();
        let size = block.size(slang::ParameterCategory::Uniform);

        let mut fields = Vec::new();
        for field in block.fields() {
            let field_name = field.name().unwrap_or_default().to_string();
            let offset = field.offset(slang::ParameterCategory::Uniform);
            let field_layout = field.type_layout();
            let field_size = field_layout.size(slang::ParameterCategory::Uniform);

            // The session asks for row-major (`matrix_layout_row(true)`), and
            // nothing downstream can confirm it got it: Slang lowers a matrix
            // to a struct wrapping a strided array and bakes the transposition
            // into the access code, so the emitted SPIR-V carries no
            // RowMajor/ColMajor decoration to check. If the option ever stopped
            // applying, every generated size and offset assertion would still
            // pass and the draw would simply come out transposed. Reflection is
            // the only place the mode is observable, so assert it here.
            if field_layout.kind() == slang::TypeKind::Matrix {
                let mode = field_layout.matrix_layout_mode();
                if mode != slang::MatrixLayoutMode::RowMajor {
                    return Err(ShaderError::Unsupported(format!(
                        "field `{field_name}` reflects matrix layout {mode:?}, not RowMajor — \
                         the session requested row-major and the generated struct assumes it; a \
                         mismatch renders transposed with every layout assertion still passing"
                    )));
                }
            }

            let ty = field_type(&field_name, field_layout)?;
            if ty.cpu_size() != field_size {
                return Err(ShaderError::Unsupported(format!(
                    "field `{field_name}`: shader size {field_size} B differs from its repr(C) \
                     size {} B ({}) — this shape has interior std430 padding no C type matches; \
                     use a shape whose strides agree (e.g. float4x4 over float3x3)",
                    ty.cpu_size(),
                    ty.rust_type(),
                )));
            }
            fields.push(StructField {
                name: field_name,
                offset,
                size: field_size,
                ty,
            });
        }
        found = Some(StructLayout { name, size, fields });
    }
    Ok(found)
}

fn scalar(context: &str, st: Option<slang::ScalarType>) -> Result<Scalar, ShaderError> {
    match st {
        Some(slang::ScalarType::Float32) => Ok(Scalar::F32),
        Some(slang::ScalarType::Uint32) => Ok(Scalar::U32),
        Some(slang::ScalarType::Int32) => Ok(Scalar::I32),
        Some(slang::ScalarType::Uint64) => Ok(Scalar::U64),
        other => Err(ShaderError::Unsupported(format!(
            "field `{context}`: scalar type {other:?} — codegen covers 32-bit float/uint/int \
             and uint64_t"
        ))),
    }
}

fn field_type(name: &str, tl: &slang::reflection::TypeLayout) -> Result<FieldType, ShaderError> {
    match tl.kind() {
        slang::TypeKind::Scalar => Ok(FieldType::Scalar(scalar(name, tl.scalar_type())?)),
        slang::TypeKind::Vector => {
            let count = tl.element_count().ok_or_else(|| {
                ShaderError::Unsupported(format!("field `{name}`: vector without a size"))
            })?;
            // getScalarType on a vector reports the element scalar directly.
            Ok(FieldType::Vector(
                scalar(name, tl.scalar_type())?,
                count as u32,
            ))
        }
        slang::TypeKind::Matrix => {
            let (Some(rows), Some(cols)) = (tl.row_count(), tl.column_count()) else {
                return Err(ShaderError::Unsupported(format!(
                    "field `{name}`: matrix without a shape"
                )));
            };
            if !matches!(scalar(name, tl.scalar_type()), Ok(Scalar::F32)) {
                return Err(ShaderError::Unsupported(format!(
                    "field `{name}`: non-f32 matrix — codegen covers float matrices"
                )));
            }
            Ok(FieldType::Matrix { rows, cols })
        }
        other => Err(ShaderError::Unsupported(format!(
            "field `{name}`: type kind {other:?} — codegen covers scalars, vectors, and matrices \
             (arrays and nested structs join with their first consumer)"
        ))),
    }
}
