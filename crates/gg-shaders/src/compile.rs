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
    /// `VkPipelineShaderStageCreateInfo::pName` must be. Slang emits each
    /// entry point into its own blob under the Vulkan-conventional `main`.
    pub spirv_entry: &'static str,
    /// Which stage the entry point targets.
    pub stage: Stage,
    /// The SPIR-V blob.
    pub spirv: Vec<u8>,
}

/// Scalar component type codegen supports (32-bit only, deliberately: 16/64-bit
/// join with a consumer that needs them).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scalar {
    /// `float`
    F32,
    /// `uint`
    U32,
    /// `int`
    I32,
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
            FieldType::Scalar(_) => 4,
            FieldType::Vector(_, n) => 4 * *n as usize,
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

/// Compile `module_name` (a `.slang` file under `search_dir`) to SPIR-V for
/// all of its entry points and reflect the push-constant layout.
pub fn compile_module(search_dir: &str, module_name: &str) -> Result<CompiledModule, ShaderError> {
    let global_session = slang::GlobalSession::new().ok_or(ShaderError::NoGlobalSession)?;

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
        compiled.push(CompiledEntryPoint {
            name: reflected.name().to_owned(),
            spirv_entry: "main",
            stage,
            spirv: code.as_slice().to_vec(),
        });
    }

    Ok(CompiledModule {
        entry_points: compiled,
        push_constants,
        global_parameter_count,
    })
}

/// Walk the global parameters for the push-constant block and
/// extract its layout. The reflected offsets/sizes are the ground truth the
/// generated assertions freeze (§4.4).
fn reflect_push_constants(
    reflection: &slang::reflection::Shader,
) -> Result<Option<StructLayout>, ShaderError> {
    let mut found: Option<StructLayout> = None;
    for param in reflection.parameters() {
        if param.category() != slang::ParameterCategory::PushConstantBuffer {
            continue;
        }
        if found.is_some() {
            return Err(ShaderError::Unsupported(
                "more than one push-constant block — Vulkan gives one range, keep one block".into(),
            ));
        }
        // The parameter is ConstantBuffer<T>; element_type_layout is T.
        let block = param.type_layout().element_type_layout();
        let name = block
            .name()
            .unwrap_or("PushConstants") // anonymous block: give it a stable name
            .to_string();
        let size = block.size(slang::ParameterCategory::Uniform);

        let mut fields = Vec::new();
        for field in block.fields() {
            let field_name = field.name().unwrap_or_default().to_string();
            let offset = field.offset(slang::ParameterCategory::Uniform);
            let field_size = field.type_layout().size(slang::ParameterCategory::Uniform);
            let ty = field_type(&field_name, field.type_layout())?;
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
        other => Err(ShaderError::Unsupported(format!(
            "field `{context}`: scalar type {other:?} — codegen covers 32-bit float/uint/int"
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
            if scalar(name, tl.scalar_type()).is_err() {
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
