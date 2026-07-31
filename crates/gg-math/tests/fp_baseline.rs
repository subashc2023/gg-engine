//! The M0 FP determinism baseline (§6 spike 3, §4.2.1). A fixed sequence of
//! `gg_math::sim` operations in both widths — arithmetic, `sqrt`, `mul_add`
//! (hardware-FMA vs. software-fallback is the belief this converts to a
//! check), every libm transcendental, a vector/quaternion/matrix chain —
//! whose bit patterns must be identical across all three targets of the §5
//! execution matrix *and* across the dev and dist profiles (§5.6d). One
//! deliberately produced NaN per width is recorded per architecture — the
//! sole baseline lines permitted to differ across targets (hazard 6).
//!
//! **There is deliberately no `--bless`.** A failing run writes the fresh
//! report beside the checked-in baseline and names the first differing line;
//! replacing the baseline is a reviewed edit that must name the cause
//! (toolchain bump, libm bump, hardware change — each a Determinism Contract
//! event, §4.2.1). Regenerating it to make CI green is the one move that
//! turns this check into a rubber stamp.

#![allow(clippy::excessive_precision, clippy::unwrap_used, clippy::expect_used)]

use std::fmt::Write as _;
use std::hint::black_box;
use std::path::{Path, PathBuf};

use gg_math::sim::{self, *};

type Report = Vec<(String, String)>;

/// One width's full operation sequence. Inputs go through `black_box` so the
/// baseline measures target codegen, not compile-time constant folding.
macro_rules! width_report {
    ($out:expr, $p:literal, $hexw:literal, $t:ty,
     $Vec3:ident, $Vec4:ident, $Quat:ident, $Mat3:ident, $Mat4:ident) => {{
        let out: &mut Report = $out;
        let hex = |v: $t| format!(concat!("0x{:0", $hexw, "X}"), v.to_bits());
        let mut push = |label: &str, v: $t| out.push((format!("{}/{label}", $p), hex(v)));
        let bb = black_box::<$t>;

        // Transcendentals — every banned-method counterpart, one line each.
        let a = bb(0.62345);
        let neg = bb(-1.7);
        let big = bb(1.2345678);
        push("sin", sim::sin(a));
        push("cos", sim::cos(a));
        let (s, c) = sim::sin_cos(a);
        push("sin_cos.s", s);
        push("sin_cos.c", c);
        push("tan", sim::tan(a));
        push("asin", sim::asin(a));
        push("acos", sim::acos(a));
        push("atan", sim::atan(a));
        push("atan2", sim::atan2(a, neg));
        push("sinh", sim::sinh(a));
        push("cosh", sim::cosh(a));
        push("tanh", sim::tanh(a));
        push("exp", sim::exp(a));
        push("exp2", sim::exp2(a));
        push("exp_m1", sim::exp_m1(a));
        push("ln", sim::ln(big));
        push("ln_1p", sim::ln_1p(a));
        push("log2", sim::log2(big));
        push("log10", sim::log10(big));
        push("log", sim::log(bb(7.3), bb(2.9)));
        push("powf", sim::powf(big, bb(2.3456789)));
        push("cbrt", sim::cbrt(big));
        push("hypot", sim::hypot(bb(3.1), bb(4.2)));

        // IEEE core ops. `mul_add` beside its unfused spelling: if the
        // compiler ever contracts `x*x + c` into an FMA (it must not — §4.2.1),
        // the unfused line changes and this test is the tripwire.
        push("sqrt.2", sim::sqrt(bb(2.0)));
        push("sqrt.big", sim::sqrt(bb(12345.6789)));
        let m = bb(1.0000001);
        push("mul_add", sim::mul_add(m, m, bb(-1.0)));
        push("unfused_mul_add", m * m + bb(-1.0));

        // Arithmetic chain, order pinned as written.
        let mut acc = bb(1.000001);
        acc = acc * bb(1.0000001) + bb(0.999999);
        acc /= bb(3.000001);
        acc -= bb(0.111111);
        acc *= acc;
        acc += a;
        acc *= bb(7.7);
        push("chain.arith", acc);

        // Vector / quaternion / matrix chain.
        let v = $Vec3::new(bb(0.1), bb(-0.2), bb(0.3));
        let u = $Vec3::new(bb(1.5), bb(-2.5), bb(3.5));
        push("chain.dot", v.dot(u));
        push("chain.cross.x", v.cross(u).x);
        push("chain.length", u.length());

        let axis1 = $Vec3::new(bb(1.0), bb(2.0), bb(3.0))
            .try_normalize()
            .unwrap();
        let axis2 = $Vec3::new(bb(-3.0), bb(1.0), bb(2.0))
            .try_normalize()
            .unwrap();
        let q = $Quat::from_axis_angle(axis1, bb(0.7));
        let q2 = $Quat::from_axis_angle(axis2, bb(-1.3));
        let qq = q.mul(q2);
        push("chain.quat.x", qq.x);
        push("chain.quat.w", qq.w);
        let r = qq.rotate(u);
        push("chain.rotate.y", r.y);

        let ma = $Mat4::from_rotation_translation(qq, $Vec3::new(bb(0.1), bb(-0.2), bb(0.3)));
        let mb = $Mat4::from_rotation_translation(q, $Vec3::new(bb(5.0), bb(6.0), bb(7.0)));
        let mm = ma.mul(mb);
        let p = mm.transform_point3(v);
        push("chain.mat.p.x", p.x);
        push("chain.mat.p.y", p.y);
        push("chain.mat.p.z", p.z);
        push("chain.mat.det", $Mat3::from_quat(qq).determinant());

        // Hazard 6: the one architecture-tagged section. IEEE 754 leaves NaN
        // sign/payload unspecified and SSE2 vs AArch64 genuinely differ; the
        // divergence is pinned here rather than rediscovered.
        let z = bb(0.0);
        let nan = z / z;
        push(&format!("nan_div@{}", std::env::consts::ARCH), nan);
    }};
}

fn build_report() -> Report {
    let mut out = Report::new();
    width_report!(&mut out, "f32", "8", f32, Vec3, Vec4, Quat, Mat3, Mat4);
    width_report!(
        &mut out, "f64", "16", f64, DVec3, DVec4, DQuat, DMat3, DMat4
    );
    out
}

fn baseline_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("baseline")
}

fn write_new_report(report: &Report) -> PathBuf {
    let path = baseline_dir().join(format!("fp_baseline.new.{}.txt", std::env::consts::ARCH));
    let mut text = String::from(
        "# Fresh FP baseline report — NOT the baseline. Promote lines into\n\
         # fp_baseline.txt only in a reviewed edit that names the cause (§4.2.1).\n",
    );
    for (label, value) in report {
        let _ = writeln!(text, "{label} = {value}");
    }
    std::fs::create_dir_all(baseline_dir()).expect("create baseline dir");
    std::fs::write(&path, text).expect("write fresh report");
    path
}

#[test]
fn fp_baseline_matches_checked_in_bits() {
    let report = build_report();
    let baseline_path = baseline_dir().join("fp_baseline.txt");

    let Ok(baseline_text) = std::fs::read_to_string(&baseline_path) else {
        let new = write_new_report(&report);
        panic!(
            "no checked-in FP baseline at {} — inspect the fresh report at {} \
             and commit it as the baseline (initial creation is a reviewed act, \
             not a bless)",
            baseline_path.display(),
            new.display()
        );
    };

    let baseline: Vec<(&str, &str)> = baseline_text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .filter_map(|l| l.split_once(" = "))
        .collect();

    let mut problems: Vec<String> = Vec::new();

    // Every line this target produced must match the baseline exactly.
    for (label, value) in &report {
        match baseline.iter().find(|(l, _)| l == label) {
            Some((_, expected)) if expected == value => {}
            Some((_, expected)) => {
                problems.push(format!("{label}: baseline {expected}, this run {value}"))
            }
            None => problems.push(format!("{label}: missing from baseline")),
        }
    }

    // Every baseline line must be exercised, except sections tagged for a
    // different architecture (the NaN lines, hazard 6).
    let arch_tag = format!("@{}", std::env::consts::ARCH);
    for (label, _) in &baseline {
        let other_arch = label.contains('@') && !label.ends_with(&arch_tag);
        if !other_arch && !report.iter().any(|(l, _)| l == label) {
            problems.push(format!("{label}: in baseline but not produced by this run"));
        }
    }

    if !problems.is_empty() {
        let new = write_new_report(&report);
        panic!(
            "FP baseline diverged ({} line(s)); first: {}\n\
             fresh report written to {} — replacing the baseline requires \
             naming the cause (§4.2.1: toolchain/libm/hardware are Determinism \
             Contract events)\nall problems:\n{}",
            problems.len(),
            problems[0],
            new.display(),
            problems.join("\n")
        );
    }
}
