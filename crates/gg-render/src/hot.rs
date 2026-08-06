//! Shader hot reload for the passes this crate owns (§4.4): watch `shaders/`,
//! recompile the edited module in-process, rebuild its pipelines behind the
//! frame timeline. Dev-tier lab equipment — this module exists only under
//! `hot-reload`, and the dist gate proves `gg-shaders`/`notify` absent (§5.8).
//!
//! The shell gets this without a line of its own: [`crate::Renderer::frame`]
//! polls here, so §9's "any shader edit is on screen in under half a second"
//! is the engine's property, not one demo's.

use std::time::Instant;

use gg_shaders::hot::ShaderWatcher;
use gg_shaders::{CompiledEntryPoint, CompiledModule};
use tracing::{error, info, warn};

use crate::scene::ScenePass;
use crate::ui::UiPass;
use crate::{BoxPass, GpuHost, PipelineDesc, PipelineHandle, shader};

/// The four modules the passes draw with. Every watcher fires on any `.slang`
/// event under the directory (`include/pbr.slang` is included by two of them),
/// so one edit recompiles all four — deliberate: per-file dependency tracking
/// would be a build system, and four in-process compiles fit the half-second
/// budget with room to spare.
enum Module {
    Ugly,
    Post,
    Scene,
    Ui,
}

/// The watchers and the save-to-screen clock, held by both renderers.
pub(crate) struct Shaders {
    watchers: Vec<(Module, ShaderWatcher)>,
    /// Set when rebuilt pipelines swap in; the next presented (or read-back)
    /// frame closes the measurement.
    swapped_at: Option<Instant>,
}

impl Shaders {
    /// Watch this crate's `shaders/` source tree. `GG_SHADER_SRC` overrides
    /// the directory — the offscreen gate edits a *copy*, never the tree.
    ///
    /// `None`, with a warning, when watching fails: dev convenience must not
    /// kill the shell — e.g. a binary run away from the source tree.
    pub(crate) fn new() -> Option<Self> {
        let dir = std::env::var("GG_SHADER_SRC")
            .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/shaders").to_string());
        let mut watchers = Vec::new();
        for (name, module) in [
            ("ugly", Module::Ugly),
            ("post", Module::Post),
            ("scene", Module::Scene),
            ("ui", Module::Ui),
        ] {
            match ShaderWatcher::new(&dir, name) {
                Ok(watcher) => watchers.push((module, watcher)),
                Err(e) => {
                    warn!("shader hot reload disabled: {e}");
                    return None;
                }
            }
        }
        Some(Shaders {
            watchers,
            swapped_at: None,
        })
    }

    /// Poll for edits; on a successful recompile swap that module's pipelines,
    /// keeping last-good on any compile or creation failure. Call between
    /// frames — the retired pipelines go through the RHI's deferred destroy,
    /// so in-flight frames finish on the handles they recorded.
    pub(crate) fn poll(
        &mut self,
        rhi: &mut impl GpuHost,
        pass: &mut BoxPass,
        scene: &mut ScenePass,
        ui: &mut UiPass,
    ) {
        for (module, watcher) in &mut self.watchers {
            let Some(result) = watcher.poll() else {
                continue;
            };
            let recompiled = match result {
                Ok(r) => r,
                Err(e) => {
                    error!("shader edit failed to compile — keeping last-good pipelines:\n{e}");
                    continue;
                }
            };
            let swapped = match module {
                Module::Ugly => pass.swap_ugly(rhi, &recompiled.module),
                Module::Post => pass.swap_post(rhi, &recompiled.module),
                Module::Scene => scene.swap_shaders(rhi, &recompiled.module),
                Module::Ui => ui.swap_shaders(rhi, &recompiled.module),
            };
            match swapped {
                Ok(()) => {
                    info!(
                        compile_ms = recompiled.compile_time.as_secs_f64() * 1e3,
                        "hot reload: pipelines rebuilt behind the timeline"
                    );
                    // One save fires several watchers; the earliest event keeps
                    // the reported save-to-screen time honest.
                    self.swapped_at = Some(
                        self.swapped_at
                            .map_or(recompiled.saved_at, |prev| prev.min(recompiled.saved_at)),
                    );
                }
                Err(e) => error!("shader hot reload — keeping last-good pipelines: {e}"),
            }
        }
    }

    /// Call after a presented (or read-back) frame: closes the save-to-screen
    /// clock the §4.4 half-second budget is measured on.
    pub(crate) fn presented(&mut self) {
        if let Some(saved_at) = self.swapped_at.take() {
            info!(
                save_to_screen_ms = saved_at.elapsed().as_secs_f64() * 1e3,
                "hot reload: new shader on screen (§4.4 budget: 500 ms)"
            );
        }
    }
}

/// A module's vertex/fragment entry pair, refused if either went missing or
/// the push-constant block no longer matches the layout the CPU side was
/// compiled against — that edit needs `cargo xtask shaders` and a rebuild
/// (§4.4 codegen), and swapping it in would corrupt every draw instead.
pub(crate) fn pair<'m>(
    module: &'m CompiledModule,
    vs: &str,
    fs: &str,
    push_size: usize,
) -> Result<(&'m CompiledEntryPoint, &'m CompiledEntryPoint), String> {
    let find = |name: &str| module.entry_points.iter().find(|ep| ep.name == name);
    let (Some(vs_ep), Some(fs_ep)) = (find(vs), find(fs)) else {
        return Err(format!("the edit dropped entry point `{vs}` or `{fs}`"));
    };
    let declared = module.push_constants.as_ref().map_or(0, |p| p.size);
    if declared != push_size {
        return Err(format!(
            "the edit changed the push-constant layout ({declared} B where this build expects \
             {push_size} B) — that needs `cargo xtask shaders` and a rebuild (§4.4 codegen)"
        ));
    }
    Ok((vs_ep, fs_ep))
}

/// Create every new pipeline before retiring any old one: a failure mid-list
/// keeps the whole last-good set rather than leaving a chimera of old and new.
pub(crate) fn swap_all(
    rhi: &mut impl GpuHost,
    swaps: &mut [(&mut PipelineHandle, PipelineDesc<'_>)],
) -> Result<(), String> {
    let mut created = Vec::new();
    for (_, desc) in swaps.iter() {
        match rhi.create_pipeline(desc) {
            Ok(handle) => created.push(handle),
            Err(e) => {
                let name = desc.name;
                for handle in created {
                    if let Err(e) = rhi.destroy_pipeline(handle) {
                        warn!("retiring a half-built pipeline set: {e}");
                    }
                }
                return Err(format!("`{name}` failed to build: {e}"));
            }
        }
    }
    for ((slot, _), new) in swaps.iter_mut().zip(created) {
        if let Err(e) = rhi.destroy_pipeline(**slot) {
            warn!("retiring an old pipeline: {e}");
        }
        **slot = new;
    }
    Ok(())
}

impl BoxPass {
    /// Rebuild the three `ugly.slang` pipelines from a hot recompile.
    pub(crate) fn swap_ugly(
        &mut self,
        rhi: &mut impl GpuHost,
        module: &CompiledModule,
    ) -> Result<(), String> {
        let push = core::mem::size_of::<shader::UglyPush>();
        let (vs_main, fs_main) = pair(module, "vs_main", "fs_main", push)?;
        let (vs_depth, fs_depth) = pair(module, "vs_depth", "fs_depth", push)?;
        let (vs_shadow, _) = pair(module, "vs_shadow", "fs_depth", push)?;
        swap_all(
            rhi,
            &mut [
                (
                    &mut self.prepass,
                    PipelineDesc {
                        vs_spirv: &vs_depth.spirv,
                        vs_entry: &vs_depth.spirv_entry,
                        fs_spirv: &fs_depth.spirv,
                        fs_entry: &fs_depth.spirv_entry,
                        ..crate::prepass_desc()
                    },
                ),
                (
                    &mut self.forward,
                    PipelineDesc {
                        vs_spirv: &vs_main.spirv,
                        vs_entry: &vs_main.spirv_entry,
                        fs_spirv: &fs_main.spirv,
                        fs_entry: &fs_main.spirv_entry,
                        ..crate::forward_desc()
                    },
                ),
                (
                    &mut self.shadow,
                    PipelineDesc {
                        vs_spirv: &vs_shadow.spirv,
                        vs_entry: &vs_shadow.spirv_entry,
                        fs_spirv: &fs_depth.spirv,
                        fs_entry: &fs_depth.spirv_entry,
                        ..crate::shadow_desc()
                    },
                ),
            ],
        )
    }

    /// Rebuild the two `post.slang` pipelines from a hot recompile.
    pub(crate) fn swap_post(
        &mut self,
        rhi: &mut impl GpuHost,
        module: &CompiledModule,
    ) -> Result<(), String> {
        let push = core::mem::size_of::<crate::post_shader::PostPush>();
        let (vs_main, fs_main) = pair(module, "vs_main", "fs_main", push)?;
        let (_, fs_downsample) = pair(module, "vs_main", "fs_downsample", push)?;
        swap_all(
            rhi,
            &mut [
                (
                    &mut self.post,
                    PipelineDesc {
                        vs_spirv: &vs_main.spirv,
                        vs_entry: &vs_main.spirv_entry,
                        fs_spirv: &fs_main.spirv,
                        fs_entry: &fs_main.spirv_entry,
                        ..crate::post_desc()
                    },
                ),
                (
                    &mut self.downsample,
                    PipelineDesc {
                        vs_spirv: &vs_main.spirv,
                        vs_entry: &vs_main.spirv_entry,
                        fs_spirv: &fs_downsample.spirv,
                        fs_entry: &fs_downsample.spirv_entry,
                        ..crate::downsample_desc()
                    },
                ),
            ],
        )
    }
}
