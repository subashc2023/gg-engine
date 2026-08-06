//! `gg_math::sim` — deterministic scalar math (§4.2.1). Two widths of our own
//! types (`Vec*`/`Quat`/`Mat*` in `f32`, `DVec*`/`DQuat`/`DMat*` in `f64`),
//! all `bytemuck::Pod`, no SIMD, no platform libm: transcendentals come from
//! the `libm` crate so the same Rust source compiles to the same IEEE
//! operations on every target (hazard 1). Operation order in every compound
//! formula is pinned by writing it out — never by iterator folds whose order
//! a rewrite could shuffle (hazard 2).
//!
//! Transcendentals are **free functions** (`sim::sin(x)`), overload-selected
//! by width through the sealed [`SimFloat`] trait — never extension methods,
//! because a `.sin()` method would be textually indistinguishable from the
//! lint-banned `f32::sin` (§4.2.1).
//!
//! Sim-space *positions* are `f64` ([`DVec3`]; Expanse scale needs the ULP
//! headroom); most other quantities stay `f32`, chosen per component as a
//! gameplay decision (§4.2.1). Narrowing to render space goes through
//! [`crate::render::camera_relative`] — there is no `From` impl on purpose.

use bytemuck::{Pod, Zeroable};

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Width selection for the free-function transcendentals. Sealed: exactly
/// `f32` and `f64`, matching the two type families of this module.
///
/// The `*_impl` methods are plumbing, not API — call the free functions.
pub trait SimFloat: sealed::Sealed + Copy {
    #[doc(hidden)]
    fn sin_impl(self) -> Self;
    #[doc(hidden)]
    fn cos_impl(self) -> Self;
    #[doc(hidden)]
    fn sin_cos_impl(self) -> (Self, Self);
    #[doc(hidden)]
    fn tan_impl(self) -> Self;
    #[doc(hidden)]
    fn asin_impl(self) -> Self;
    #[doc(hidden)]
    fn acos_impl(self) -> Self;
    #[doc(hidden)]
    fn atan_impl(self) -> Self;
    #[doc(hidden)]
    fn atan2_impl(self, x: Self) -> Self;
    #[doc(hidden)]
    fn sinh_impl(self) -> Self;
    #[doc(hidden)]
    fn cosh_impl(self) -> Self;
    #[doc(hidden)]
    fn tanh_impl(self) -> Self;
    #[doc(hidden)]
    fn exp_impl(self) -> Self;
    #[doc(hidden)]
    fn exp2_impl(self) -> Self;
    #[doc(hidden)]
    fn exp_m1_impl(self) -> Self;
    #[doc(hidden)]
    fn ln_impl(self) -> Self;
    #[doc(hidden)]
    fn ln_1p_impl(self) -> Self;
    #[doc(hidden)]
    fn log2_impl(self) -> Self;
    #[doc(hidden)]
    fn log10_impl(self) -> Self;
    #[doc(hidden)]
    fn powf_impl(self, e: Self) -> Self;
    #[doc(hidden)]
    fn cbrt_impl(self) -> Self;
    #[doc(hidden)]
    fn hypot_impl(self, y: Self) -> Self;
    #[doc(hidden)]
    fn sqrt_impl(self) -> Self;
    #[doc(hidden)]
    fn mul_add_impl(self, b: Self, c: Self) -> Self;
    #[doc(hidden)]
    fn div_impl(self, rhs: Self) -> Self;
}

impl SimFloat for f32 {
    fn sin_impl(self) -> Self {
        libm::sinf(self)
    }
    fn cos_impl(self) -> Self {
        libm::cosf(self)
    }
    fn sin_cos_impl(self) -> (Self, Self) {
        libm::sincosf(self)
    }
    fn tan_impl(self) -> Self {
        libm::tanf(self)
    }
    fn asin_impl(self) -> Self {
        libm::asinf(self)
    }
    fn acos_impl(self) -> Self {
        libm::acosf(self)
    }
    fn atan_impl(self) -> Self {
        libm::atanf(self)
    }
    fn atan2_impl(self, x: Self) -> Self {
        libm::atan2f(self, x)
    }
    fn sinh_impl(self) -> Self {
        libm::sinhf(self)
    }
    fn cosh_impl(self) -> Self {
        libm::coshf(self)
    }
    fn tanh_impl(self) -> Self {
        libm::tanhf(self)
    }
    fn exp_impl(self) -> Self {
        libm::expf(self)
    }
    fn exp2_impl(self) -> Self {
        libm::exp2f(self)
    }
    fn exp_m1_impl(self) -> Self {
        libm::expm1f(self)
    }
    fn ln_impl(self) -> Self {
        libm::logf(self)
    }
    fn ln_1p_impl(self) -> Self {
        libm::log1pf(self)
    }
    fn log2_impl(self) -> Self {
        libm::log2f(self)
    }
    fn log10_impl(self) -> Self {
        libm::log10f(self)
    }
    fn powf_impl(self, e: Self) -> Self {
        libm::powf(self, e)
    }
    fn cbrt_impl(self) -> Self {
        libm::cbrtf(self)
    }
    fn hypot_impl(self, y: Self) -> Self {
        libm::hypotf(self, y)
    }
    fn sqrt_impl(self) -> Self {
        // IEEE 754 core op, correctly rounded everywhere — std is fine here.
        self.sqrt()
    }
    fn mul_add_impl(self, b: Self, c: Self) -> Self {
        self.mul_add(b, c)
    }
    fn div_impl(self, rhs: Self) -> Self {
        self / rhs
    }
}

impl SimFloat for f64 {
    fn sin_impl(self) -> Self {
        libm::sin(self)
    }
    fn cos_impl(self) -> Self {
        libm::cos(self)
    }
    fn sin_cos_impl(self) -> (Self, Self) {
        libm::sincos(self)
    }
    fn tan_impl(self) -> Self {
        libm::tan(self)
    }
    fn asin_impl(self) -> Self {
        libm::asin(self)
    }
    fn acos_impl(self) -> Self {
        libm::acos(self)
    }
    fn atan_impl(self) -> Self {
        libm::atan(self)
    }
    fn atan2_impl(self, x: Self) -> Self {
        libm::atan2(self, x)
    }
    fn sinh_impl(self) -> Self {
        libm::sinh(self)
    }
    fn cosh_impl(self) -> Self {
        libm::cosh(self)
    }
    fn tanh_impl(self) -> Self {
        libm::tanh(self)
    }
    fn exp_impl(self) -> Self {
        libm::exp(self)
    }
    fn exp2_impl(self) -> Self {
        libm::exp2(self)
    }
    fn exp_m1_impl(self) -> Self {
        libm::expm1(self)
    }
    fn ln_impl(self) -> Self {
        libm::log(self)
    }
    fn ln_1p_impl(self) -> Self {
        libm::log1p(self)
    }
    fn log2_impl(self) -> Self {
        libm::log2(self)
    }
    fn log10_impl(self) -> Self {
        libm::log10(self)
    }
    fn powf_impl(self, e: Self) -> Self {
        libm::pow(self, e)
    }
    fn cbrt_impl(self) -> Self {
        libm::cbrt(self)
    }
    fn hypot_impl(self, y: Self) -> Self {
        libm::hypot(self, y)
    }
    fn sqrt_impl(self) -> Self {
        self.sqrt()
    }
    fn mul_add_impl(self, b: Self, c: Self) -> Self {
        self.mul_add(b, c)
    }
    fn div_impl(self, rhs: Self) -> Self {
        self / rhs
    }
}

// ---- free-function transcendentals (hazard 1) ---------------------------

/// Portable sine (libm).
pub fn sin<T: SimFloat>(x: T) -> T {
    x.sin_impl()
}
/// Portable cosine (libm).
pub fn cos<T: SimFloat>(x: T) -> T {
    x.cos_impl()
}
/// Portable sine and cosine in one call (libm `sincos`).
pub fn sin_cos<T: SimFloat>(x: T) -> (T, T) {
    x.sin_cos_impl()
}
/// Portable tangent (libm).
pub fn tan<T: SimFloat>(x: T) -> T {
    x.tan_impl()
}
/// Portable arcsine (libm).
pub fn asin<T: SimFloat>(x: T) -> T {
    x.asin_impl()
}
/// Portable arccosine (libm).
pub fn acos<T: SimFloat>(x: T) -> T {
    x.acos_impl()
}
/// Portable arctangent (libm).
pub fn atan<T: SimFloat>(x: T) -> T {
    x.atan_impl()
}
/// Portable two-argument arctangent (libm).
pub fn atan2<T: SimFloat>(y: T, x: T) -> T {
    y.atan2_impl(x)
}
/// Portable hyperbolic sine (libm).
pub fn sinh<T: SimFloat>(x: T) -> T {
    x.sinh_impl()
}
/// Portable hyperbolic cosine (libm).
pub fn cosh<T: SimFloat>(x: T) -> T {
    x.cosh_impl()
}
/// Portable hyperbolic tangent (libm).
pub fn tanh<T: SimFloat>(x: T) -> T {
    x.tanh_impl()
}
/// Portable natural exponential (libm).
pub fn exp<T: SimFloat>(x: T) -> T {
    x.exp_impl()
}
/// Portable base-2 exponential (libm).
pub fn exp2<T: SimFloat>(x: T) -> T {
    x.exp2_impl()
}
/// Portable `e^x - 1` (libm `expm1`).
pub fn exp_m1<T: SimFloat>(x: T) -> T {
    x.exp_m1_impl()
}
/// Portable natural logarithm (libm).
pub fn ln<T: SimFloat>(x: T) -> T {
    x.ln_impl()
}
/// Portable `ln(1 + x)` (libm `log1p`).
pub fn ln_1p<T: SimFloat>(x: T) -> T {
    x.ln_1p_impl()
}
/// Portable base-2 logarithm (libm).
pub fn log2<T: SimFloat>(x: T) -> T {
    x.log2_impl()
}
/// Portable base-10 logarithm (libm).
pub fn log10<T: SimFloat>(x: T) -> T {
    x.log10_impl()
}
/// Arbitrary-base logarithm, *defined* as `ln(x) / ln(base)` — composed from
/// two portable ops and one IEEE division, so it is deterministic, unlike
/// `f32::log` whose composition is the CRT's business.
pub fn log<T: SimFloat>(x: T, base: T) -> T {
    x.ln_impl().div_impl(base.ln_impl())
}
/// Portable power (libm).
pub fn powf<T: SimFloat>(x: T, e: T) -> T {
    x.powf_impl(e)
}
/// Portable cube root (libm).
pub fn cbrt<T: SimFloat>(x: T) -> T {
    x.cbrt_impl()
}
/// Portable `sqrt(x² + y²)` without intermediate overflow (libm).
pub fn hypot<T: SimFloat>(x: T, y: T) -> T {
    x.hypot_impl(y)
}
/// Square root — an IEEE 754 core op, correctly rounded on every target; the
/// free function exists so sim code reads uniformly, not because std differs.
pub fn sqrt<T: SimFloat>(x: T) -> T {
    x.sqrt_impl()
}
/// Fused multiply-add `a*b + c` with one rounding — an IEEE 754 core op;
/// hardware-FMA and software-fallback agree bit-for-bit (the M0 FP baseline
/// keeps that a tested fact, §4.2.1).
pub fn mul_add<T: SimFloat>(a: T, b: T, c: T) -> T {
    a.mul_add_impl(b, c)
}

// ---- the type families, macro-generated per width -----------------------

macro_rules! sim_types {
    ($t:ty, $Vec2:ident, $Vec3:ident, $Vec4:ident, $Quat:ident, $Mat3:ident, $Mat4:ident) => {
        #[repr(C)]
        #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
        pub struct $Vec2 {
            pub x: $t,
            pub y: $t,
        }

        #[repr(C)]
        #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
        pub struct $Vec3 {
            pub x: $t,
            pub y: $t,
            pub z: $t,
        }

        #[repr(C)]
        #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
        pub struct $Vec4 {
            pub x: $t,
            pub y: $t,
            pub z: $t,
            pub w: $t,
        }

        /// Rotation quaternion, `(x, y, z, w)` with `w` the scalar part.
        #[repr(C)]
        #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
        pub struct $Quat {
            pub x: $t,
            pub y: $t,
            pub z: $t,
            pub w: $t,
        }

        /// Column-major 3×3 matrix.
        #[repr(C)]
        #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
        pub struct $Mat3 {
            pub x_axis: $Vec3,
            pub y_axis: $Vec3,
            pub z_axis: $Vec3,
        }

        /// Column-major 4×4 matrix.
        #[repr(C)]
        #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
        pub struct $Mat4 {
            pub x_axis: $Vec4,
            pub y_axis: $Vec4,
            pub z_axis: $Vec4,
            pub w_axis: $Vec4,
        }

        impl $Vec2 {
            pub const ZERO: Self = Self::splat(0.0);
            pub const ONE: Self = Self::splat(1.0);

            pub const fn new(x: $t, y: $t) -> Self {
                Self { x, y }
            }
            pub const fn splat(v: $t) -> Self {
                Self { x: v, y: v }
            }
            pub fn dot(self, rhs: Self) -> $t {
                self.x * rhs.x + self.y * rhs.y
            }
            pub fn length_squared(self) -> $t {
                self.dot(self)
            }
            pub fn length(self) -> $t {
                sqrt(self.length_squared())
            }
            pub fn distance_squared(self, rhs: Self) -> $t {
                (rhs - self).length_squared()
            }
            pub fn distance(self, rhs: Self) -> $t {
                (rhs - self).length()
            }
            /// `None` when the length is zero — the caller decides the policy;
            /// a silent NaN in sim state is hazard 6.
            pub fn try_normalize(self) -> Option<Self> {
                let len = self.length();
                if len > 0.0 { Some(self / len) } else { None }
            }
            /// Pinned order: `a + (b - a) * t`.
            pub fn lerp(self, rhs: Self, t: $t) -> Self {
                self + (rhs - self) * t
            }
        }

        impl $Vec3 {
            pub const ZERO: Self = Self::splat(0.0);
            pub const ONE: Self = Self::splat(1.0);
            pub const X: Self = Self::new(1.0, 0.0, 0.0);
            pub const Y: Self = Self::new(0.0, 1.0, 0.0);
            pub const Z: Self = Self::new(0.0, 0.0, 1.0);

            pub const fn new(x: $t, y: $t, z: $t) -> Self {
                Self { x, y, z }
            }
            pub const fn splat(v: $t) -> Self {
                Self { x: v, y: v, z: v }
            }
            pub fn dot(self, rhs: Self) -> $t {
                self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
            }
            /// Pinned order per component: `a.y*b.z - a.z*b.y`, etc.
            pub fn cross(self, rhs: Self) -> Self {
                Self {
                    x: self.y * rhs.z - self.z * rhs.y,
                    y: self.z * rhs.x - self.x * rhs.z,
                    z: self.x * rhs.y - self.y * rhs.x,
                }
            }
            pub fn length_squared(self) -> $t {
                self.dot(self)
            }
            pub fn length(self) -> $t {
                sqrt(self.length_squared())
            }
            pub fn distance_squared(self, rhs: Self) -> $t {
                (rhs - self).length_squared()
            }
            pub fn distance(self, rhs: Self) -> $t {
                (rhs - self).length()
            }
            /// `None` when the length is zero — the caller decides the policy;
            /// a silent NaN in sim state is hazard 6.
            pub fn try_normalize(self) -> Option<Self> {
                let len = self.length();
                if len > 0.0 { Some(self / len) } else { None }
            }
            /// Pinned order: `a + (b - a) * t`.
            pub fn lerp(self, rhs: Self, t: $t) -> Self {
                self + (rhs - self) * t
            }
        }

        impl $Vec4 {
            pub const ZERO: Self = Self::splat(0.0);
            pub const ONE: Self = Self::splat(1.0);

            pub const fn new(x: $t, y: $t, z: $t, w: $t) -> Self {
                Self { x, y, z, w }
            }
            pub const fn splat(v: $t) -> Self {
                Self {
                    x: v,
                    y: v,
                    z: v,
                    w: v,
                }
            }
            pub fn dot(self, rhs: Self) -> $t {
                self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
            }
            pub fn length_squared(self) -> $t {
                self.dot(self)
            }
            pub fn length(self) -> $t {
                sqrt(self.length_squared())
            }
            /// Pinned order: `a + (b - a) * t`.
            pub fn lerp(self, rhs: Self, t: $t) -> Self {
                self + (rhs - self) * t
            }
        }

        impl $Quat {
            pub const IDENTITY: Self = Self {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            };

            pub const fn from_xyzw(x: $t, y: $t, z: $t, w: $t) -> Self {
                Self { x, y, z, w }
            }
            /// `axis` must be normalized; `angle` is radians.
            pub fn from_axis_angle(axis: $Vec3, angle: $t) -> Self {
                const HALF: $t = 0.5;
                let (s, c) = sin_cos(angle * HALF);
                Self {
                    x: axis.x * s,
                    y: axis.y * s,
                    z: axis.z * s,
                    w: c,
                }
            }
            pub fn dot(self, rhs: Self) -> $t {
                self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
            }
            pub fn length(self) -> $t {
                sqrt(self.dot(self))
            }
            /// `None` when the length is zero (hazard 6: no silent NaN).
            pub fn try_normalize(self) -> Option<Self> {
                let len = self.length();
                if len > 0.0 {
                    Some(Self {
                        x: self.x / len,
                        y: self.y / len,
                        z: self.z / len,
                        w: self.w / len,
                    })
                } else {
                    None
                }
            }
            pub fn conjugate(self) -> Self {
                Self {
                    x: -self.x,
                    y: -self.y,
                    z: -self.z,
                    w: self.w,
                }
            }
            /// Hamilton product, term order pinned as written.
            #[allow(clippy::should_implement_trait)]
            pub fn mul(self, rhs: Self) -> Self {
                let (ax, ay, az, aw) = (self.x, self.y, self.z, self.w);
                let (bx, by, bz, bw) = (rhs.x, rhs.y, rhs.z, rhs.w);
                Self {
                    x: aw * bx + ax * bw + ay * bz - az * by,
                    y: aw * by - ax * bz + ay * bw + az * bx,
                    z: aw * bz + ax * by - ay * bx + az * bw,
                    w: aw * bw - ax * bx - ay * by - az * bz,
                }
            }
            /// Rotate a vector: `v + w*t + qv × t` with `t = 2 * (qv × v)` —
            /// one formula, written out, same bits every target.
            pub fn rotate(self, v: $Vec3) -> $Vec3 {
                const TWO: $t = 2.0;
                let qv = $Vec3::new(self.x, self.y, self.z);
                let t = qv.cross(v) * TWO;
                v + t * self.w + qv.cross(t)
            }
        }

        impl $Mat3 {
            pub const IDENTITY: Self = Self {
                x_axis: $Vec3::new(1.0, 0.0, 0.0),
                y_axis: $Vec3::new(0.0, 1.0, 0.0),
                z_axis: $Vec3::new(0.0, 0.0, 1.0),
            };

            pub const fn from_cols(x_axis: $Vec3, y_axis: $Vec3, z_axis: $Vec3) -> Self {
                Self {
                    x_axis,
                    y_axis,
                    z_axis,
                }
            }
            /// Rotation matrix of a unit quaternion; formula written out once.
            pub fn from_quat(q: $Quat) -> Self {
                const TWO: $t = 2.0;
                let (x, y, z, w) = (q.x, q.y, q.z, q.w);
                let x2 = x * TWO;
                let y2 = y * TWO;
                let z2 = z * TWO;
                let xx = x * x2;
                let xy = x * y2;
                let xz = x * z2;
                let yy = y * y2;
                let yz = y * z2;
                let zz = z * z2;
                let wx = w * x2;
                let wy = w * y2;
                let wz = w * z2;
                Self {
                    x_axis: $Vec3::new(1.0 - (yy + zz), xy + wz, xz - wy),
                    y_axis: $Vec3::new(xy - wz, 1.0 - (xx + zz), yz + wx),
                    z_axis: $Vec3::new(xz + wy, yz - wx, 1.0 - (xx + yy)),
                }
            }
            /// Pinned order: `x_axis*v.x + y_axis*v.y + z_axis*v.z`, left to right.
            pub fn mul_vec3(self, v: $Vec3) -> $Vec3 {
                self.x_axis * v.x + self.y_axis * v.y + self.z_axis * v.z
            }
            /// Column-by-column, each column via [`Self::mul_vec3`].
            #[allow(clippy::should_implement_trait)]
            pub fn mul(self, rhs: Self) -> Self {
                Self {
                    x_axis: self.mul_vec3(rhs.x_axis),
                    y_axis: self.mul_vec3(rhs.y_axis),
                    z_axis: self.mul_vec3(rhs.z_axis),
                }
            }
            pub fn transpose(self) -> Self {
                Self {
                    x_axis: $Vec3::new(self.x_axis.x, self.y_axis.x, self.z_axis.x),
                    y_axis: $Vec3::new(self.x_axis.y, self.y_axis.y, self.z_axis.y),
                    z_axis: $Vec3::new(self.x_axis.z, self.y_axis.z, self.z_axis.z),
                }
            }
            /// Scalar triple product `x · (y × z)`, order pinned as written.
            pub fn determinant(self) -> $t {
                self.x_axis.dot(self.y_axis.cross(self.z_axis))
            }
        }

        impl $Mat4 {
            pub const IDENTITY: Self = Self {
                x_axis: $Vec4::new(1.0, 0.0, 0.0, 0.0),
                y_axis: $Vec4::new(0.0, 1.0, 0.0, 0.0),
                z_axis: $Vec4::new(0.0, 0.0, 1.0, 0.0),
                w_axis: $Vec4::new(0.0, 0.0, 0.0, 1.0),
            };

            pub const fn from_cols(
                x_axis: $Vec4,
                y_axis: $Vec4,
                z_axis: $Vec4,
                w_axis: $Vec4,
            ) -> Self {
                Self {
                    x_axis,
                    y_axis,
                    z_axis,
                    w_axis,
                }
            }
            pub const fn from_translation(t: $Vec3) -> Self {
                Self {
                    w_axis: $Vec4::new(t.x, t.y, t.z, 1.0),
                    ..Self::IDENTITY
                }
            }
            pub fn from_rotation_translation(q: $Quat, t: $Vec3) -> Self {
                let r = $Mat3::from_quat(q);
                Self {
                    x_axis: $Vec4::new(r.x_axis.x, r.x_axis.y, r.x_axis.z, 0.0),
                    y_axis: $Vec4::new(r.y_axis.x, r.y_axis.y, r.y_axis.z, 0.0),
                    z_axis: $Vec4::new(r.z_axis.x, r.z_axis.y, r.z_axis.z, 0.0),
                    w_axis: $Vec4::new(t.x, t.y, t.z, 1.0),
                }
            }
            /// Pinned order: column combination left to right.
            pub fn mul_vec4(self, v: $Vec4) -> $Vec4 {
                self.x_axis * v.x + self.y_axis * v.y + self.z_axis * v.z + self.w_axis * v.w
            }
            /// Column-by-column, each column via [`Self::mul_vec4`].
            #[allow(clippy::should_implement_trait)]
            pub fn mul(self, rhs: Self) -> Self {
                Self {
                    x_axis: self.mul_vec4(rhs.x_axis),
                    y_axis: self.mul_vec4(rhs.y_axis),
                    z_axis: self.mul_vec4(rhs.z_axis),
                    w_axis: self.mul_vec4(rhs.w_axis),
                }
            }
            /// Transform a point (`w = 1`).
            pub fn transform_point3(self, p: $Vec3) -> $Vec3 {
                let v = self.mul_vec4($Vec4::new(p.x, p.y, p.z, 1.0));
                $Vec3::new(v.x, v.y, v.z)
            }
            /// Transform a direction (`w = 0`; translation does not apply).
            pub fn transform_vector3(self, d: $Vec3) -> $Vec3 {
                let v = self.mul_vec4($Vec4::new(d.x, d.y, d.z, 0.0));
                $Vec3::new(v.x, v.y, v.z)
            }
            pub fn transpose(self) -> Self {
                Self {
                    x_axis: $Vec4::new(self.x_axis.x, self.y_axis.x, self.z_axis.x, self.w_axis.x),
                    y_axis: $Vec4::new(self.x_axis.y, self.y_axis.y, self.z_axis.y, self.w_axis.y),
                    z_axis: $Vec4::new(self.x_axis.z, self.y_axis.z, self.z_axis.z, self.w_axis.z),
                    w_axis: $Vec4::new(self.x_axis.w, self.y_axis.w, self.z_axis.w, self.w_axis.w),
                }
            }
        }

        sim_vec_ops!($t, $Vec2, x, y);
        sim_vec_ops!($t, $Vec3, x, y, z);
        sim_vec_ops!($t, $Vec4, x, y, z, w);
    };
}

/// Componentwise operator impls; expansion order is per-field declaration
/// order, and every component op is independent, so nothing here can reorder.
macro_rules! sim_vec_ops {
    ($t:ty, $V:ident, $($f:ident),+) => {
        impl core::ops::Add for $V {
            type Output = Self;
            fn add(self, rhs: Self) -> Self {
                Self { $($f: self.$f + rhs.$f),+ }
            }
        }
        impl core::ops::Sub for $V {
            type Output = Self;
            fn sub(self, rhs: Self) -> Self {
                Self { $($f: self.$f - rhs.$f),+ }
            }
        }
        impl core::ops::Neg for $V {
            type Output = Self;
            fn neg(self) -> Self {
                Self { $($f: -self.$f),+ }
            }
        }
        impl core::ops::Mul<$t> for $V {
            type Output = Self;
            fn mul(self, rhs: $t) -> Self {
                Self { $($f: self.$f * rhs),+ }
            }
        }
        impl core::ops::Div<$t> for $V {
            type Output = Self;
            fn div(self, rhs: $t) -> Self {
                Self { $($f: self.$f / rhs),+ }
            }
        }
        impl core::ops::AddAssign for $V {
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }
        impl core::ops::SubAssign for $V {
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }
    };
}

sim_types!(f32, Vec2, Vec3, Vec4, Quat, Mat3, Mat4);
sim_types!(f64, DVec2, DVec3, DVec4, DQuat, DMat3, DMat4);

/// A fly camera's `(forward, right)` at `yaw`/`pitch`, in sim space.
///
/// Yaw about world +Y, then pitch about the rotated right axis, never roll — the
/// one order `gg_render::View::rotation` builds it in, so a pick and a picture
/// cannot disagree about which way the eye faces. `f64` trig through `libm`
/// (§4.2.1): the same bits on every target in the contract, which a `glam` basis
/// would not be.
///
/// Every fly-camera demo and the editor's pick ray call this, so the blessed
/// replays pin the arithmetic — an edit here moves hashes in five demos at once.
#[must_use]
pub fn fly_basis(yaw: f32, pitch: f32) -> (DVec3, DVec3) {
    let (sin_yaw, cos_yaw) = sin_cos(f64::from(yaw));
    let (sin_pitch, cos_pitch) = sin_cos(f64::from(pitch));
    let forward = DVec3::new(-cos_pitch * sin_yaw, sin_pitch, -cos_pitch * cos_yaw);
    let right = DVec3::new(cos_yaw, 0.0, -sin_yaw);
    (forward, right)
}

/// A deterministic random source, sized and shaped to live *in* the world
/// (§6 M18).
///
/// Integer operations only, so it is bit-identical on every target this engine
/// claims — the aarch64 leg covers it for free and no `libm` question arises.
/// `Pod`, so a game keeps it in a component and it is hashed, snapshotted,
/// replayed and migrated like any other state. An RNG that lives *beside* the
/// world instead of in it is one the replay gate silently stops covering.
///
/// # Why SplitMix64 and not PCG
///
/// `World::restore` zeroes a field whose type changed and reports it (§4.3), and
/// `Zeroable` means all-zero must be a *valid* value. PCG's increment has to be
/// odd, so a zeroed PCG is a generator stuck on one output — a migration would
/// silently produce a game whose randomness stopped. SplitMix64 is a counter
/// plus a mix, so zero is an ordinary seed and [`Rng::default`] is a real
/// stream. The degenerate case is designed out rather than asserted against.
///
/// Quality is BigCrush-passing, which is far past what shuffling seven pieces
/// asks. There is deliberately **no float output**: nothing needs one yet, and
/// §3's rule about widgets applies to this crate too.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct Rng {
    /// The counter. Public bits are the mix's, never this.
    state: u64,
}

/// SplitMix64's additive constant — the 64-bit golden ratio, odd, so the
/// counter visits every state before repeating.
const GOLDEN_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;

impl Rng {
    /// A stream from `seed`. Every seed is valid, zero included.
    #[must_use]
    pub const fn from_seed(seed: u64) -> Rng {
        Rng { state: seed }
    }

    /// The stream position — the whole of the state, and what the canonical
    /// state hash absorbs. `from_bits(rng.to_bits())` is the identity.
    ///
    /// Public because a generator the world holds is a generator the world must
    /// be able to *say*; the bits are still not an output — draw with
    /// [`Rng::next_u64`], which mixes them.
    #[must_use]
    pub const fn to_bits(self) -> u64 {
        self.state
    }

    /// The inverse of [`Rng::to_bits`].
    #[must_use]
    pub const fn from_bits(bits: u64) -> Rng {
        Rng { state: bits }
    }

    /// The next 64 bits, advancing the stream.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GOLDEN_GAMMA);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// The next 32 bits. The *high* half, because SplitMix64's low bits are the
    /// weakest part of its output and a `% n` on them is the classic way to
    /// find that out.
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// A uniform value in `0..bound`, or `None` for an empty range.
    ///
    /// Lemire's multiply-and-reject: unbiased, and the rejection is *itself*
    /// deterministic — the same stream rejects at the same draws on every
    /// machine, so a replay crosses it unchanged. A plain `% bound` would be
    /// biased toward the low values, which over a 7-bag is a piece that comes
    /// up more often than the rules say.
    pub fn below(&mut self, bound: u32) -> Option<u32> {
        if bound == 0 {
            return None;
        }
        let threshold = (bound.wrapping_neg()) % bound; // 2^32 mod bound
        loop {
            let product = u64::from(self.next_u32()) * u64::from(bound);
            #[expect(
                clippy::cast_possible_truncation,
                reason = "the low half is the fractional part Lemire tests"
            )]
            let low = product as u32;
            if low >= threshold {
                return Some((product >> 32) as u32); // < bound by construction
            }
        }
    }

    /// Fisher–Yates, in place — what a 7-bag is.
    ///
    /// Descending, drawing from `0..=i`: the ascending variant with `0..len` is
    /// the one that is subtly *not* uniform, and the two are one character
    /// apart.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "a slice this long cannot exist in a world that hashes"
            )]
            let bound = (i + 1) as u32;
            if let Some(j) = self.below(bound) {
                items.swap(i, j as usize);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    fn assert_pod<T: Pod>() {}

    /// The anchor. These four are SplitMix64's *published* reference outputs for
    /// seed 0, so this pins the implementation to the algorithm rather than to
    /// itself — a self-consistent generator that drifted would still pass a test
    /// written from its own output. Every replay baseline in the tree stands on
    /// this sequence; changing it is re-blessing every one of them.
    #[test]
    fn the_stream_is_splitmix64_and_matches_its_published_vectors() {
        let mut rng = Rng::from_seed(0);
        assert_eq!(rng.next_u64(), 0xe220_a839_7b1d_cdaf);
        assert_eq!(rng.next_u64(), 0x6e78_9e6a_a1b9_65f4);
        assert_eq!(rng.next_u64(), 0x06c4_5d18_8009_454f);
        assert_eq!(rng.next_u64(), 0xf88b_b8a8_724c_81ec);
    }

    #[test]
    fn a_zeroed_rng_is_a_working_rng() {
        // The whole reason this is SplitMix64: `World::restore` zeroes a retyped
        // field, and a generator that stopped there would be a game whose
        // randomness quietly died at a schema change.
        let (mut zeroed, mut seeded) = (Rng::zeroed(), Rng::from_seed(0));
        assert_eq!(zeroed, Rng::default());
        let drawn: Vec<u64> = (0..8).map(|_| zeroed.next_u64()).collect();
        assert_eq!(drawn, (0..8).map(|_| seeded.next_u64()).collect::<Vec<_>>());
        assert!(
            drawn.windows(2).any(|w| w[0] != w[1]),
            "a zeroed generator repeated itself: {drawn:?}"
        );
    }

    #[test]
    fn the_same_seed_replays_and_a_different_one_diverges() {
        let draw = |seed| {
            let mut rng = Rng::from_seed(seed);
            (0..32).map(|_| rng.next_u64()).collect::<Vec<_>>()
        };
        assert_eq!(draw(12345), draw(12345));
        assert_ne!(draw(12345), draw(12346));
    }

    #[test]
    fn below_stays_in_range_and_rejects_an_empty_one() {
        let mut rng = Rng::from_seed(7);
        assert_eq!(rng.below(0), None);
        assert_eq!(rng.below(1), Some(0));
        for bound in [2u32, 7, 10, 255, 256, 1000] {
            for _ in 0..2000 {
                let v = rng.below(bound).expect("non-empty range");
                assert!(v < bound, "{v} is not below {bound}");
            }
        }
    }

    /// Lemire's rejection buys uniformity, so it is worth proving it is there:
    /// a bound that does not divide 2^32 is exactly where `%` would skew. Seven
    /// is the bag size, which makes this the case the game depends on.
    #[test]
    fn below_is_flat_over_a_bound_that_does_not_divide_the_word() {
        let mut rng = Rng::from_seed(99);
        let mut counts = [0u32; 7];
        const DRAWS: u32 = 700_000;
        for _ in 0..DRAWS {
            counts[rng.below(7).expect("non-empty range") as usize] += 1;
        }
        let expected = DRAWS / 7;
        for (face, count) in counts.iter().enumerate() {
            let drift = count.abs_diff(expected);
            assert!(
                drift * 100 < expected,
                "face {face} came up {count} times against {expected} expected — over 1% off, \
                 which is where a biased reduction shows"
            );
        }
    }

    #[test]
    fn shuffle_is_a_permutation_and_actually_moves_things() {
        let mut rng = Rng::from_seed(2024);
        let mut moved = 0;
        for _ in 0..64 {
            let mut bag: [u8; 7] = [0, 1, 2, 3, 4, 5, 6];
            rng.shuffle(&mut bag);
            let mut sorted = bag;
            sorted.sort_unstable();
            assert_eq!(sorted, [0, 1, 2, 3, 4, 5, 6], "shuffle lost or duplicated");
            if bag != [0, 1, 2, 3, 4, 5, 6] {
                moved += 1;
            }
        }
        // 1/5040 of shuffles are the identity, so 64 all-identity is not chance.
        assert!(moved > 55, "only {moved}/64 shuffles reordered the bag");
    }

    #[test]
    fn shuffle_handles_the_degenerate_lengths() {
        let mut rng = Rng::from_seed(1);
        let mut empty: [u8; 0] = [];
        rng.shuffle(&mut empty);
        let mut one = [42u8];
        rng.shuffle(&mut one);
        assert_eq!(one, [42]);
    }

    #[test]
    fn every_type_is_pod_with_expected_layout() {
        assert_pod::<Vec2>();
        assert_pod::<Vec3>();
        assert_pod::<Vec4>();
        assert_pod::<Quat>();
        assert_pod::<Mat3>();
        assert_pod::<Mat4>();
        assert_pod::<DVec2>();
        assert_pod::<DVec3>();
        assert_pod::<DVec4>();
        assert_pod::<DQuat>();
        assert_pod::<DMat3>();
        assert_pod::<DMat4>();
        // §6 M18: an RNG a game keeps in a component is hashed state like the
        // rest, which is exactly what `Pod` is the entry fee for.
        assert_pod::<Rng>();
        assert_eq!(size_of::<Rng>(), 8);
        assert_eq!(size_of::<Vec2>(), 8);
        assert_eq!(size_of::<Vec3>(), 12);
        assert_eq!(size_of::<Vec4>(), 16);
        assert_eq!(size_of::<Quat>(), 16);
        assert_eq!(size_of::<Mat3>(), 36);
        assert_eq!(size_of::<Mat4>(), 64);
        assert_eq!(size_of::<DVec2>(), 16);
        assert_eq!(size_of::<DVec3>(), 24);
        assert_eq!(size_of::<DVec4>(), 32);
        assert_eq!(size_of::<DQuat>(), 32);
        assert_eq!(size_of::<DMat3>(), 72);
        assert_eq!(size_of::<DMat4>(), 128);
    }

    #[test]
    fn transcendentals_dispatch_to_libm_in_both_widths() {
        // The free functions must be *exactly* libm — bit-compare, not approx.
        assert_eq!(sin(0.62345f32).to_bits(), libm::sinf(0.62345).to_bits());
        assert_eq!(sin(0.62345f64).to_bits(), libm::sin(0.62345).to_bits());
        assert_eq!(
            powf(1.234f32, 2.345).to_bits(),
            libm::powf(1.234, 2.345).to_bits()
        );
        assert_eq!(
            powf(1.234f64, 2.345).to_bits(),
            libm::pow(1.234, 2.345).to_bits()
        );
        let (s32, c32) = sin_cos(0.9f32);
        assert_eq!((s32, c32), libm::sincosf(0.9));
        let (s64, c64) = sin_cos(0.9f64);
        assert_eq!((s64, c64), libm::sincos(0.9));
    }

    #[test]
    fn exact_identities() {
        assert_eq!(sin(0.0f32).to_bits(), 0.0f32.to_bits());
        assert_eq!(cos(0.0f64).to_bits(), 1.0f64.to_bits());
        assert_eq!(exp(0.0f32).to_bits(), 1.0f32.to_bits());
        assert_eq!(ln(1.0f64).to_bits(), 0.0f64.to_bits());
        assert_eq!(sqrt(4.0f32), 2.0);
        assert_eq!(sqrt(2.25f64), 1.5);
        assert_eq!(mul_add(2.0f32, 3.0, 4.0), 10.0);
        assert_eq!(mul_add(2.0f64, 3.0, 4.0), 10.0);
        assert_eq!(log(8.0f64, 2.0), ln(8.0f64) / ln(2.0f64));
    }

    #[test]
    fn vector_algebra() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, -5.0, 6.0);
        assert_eq!(a.dot(b), 4.0 - 10.0 + 18.0);
        assert_eq!(Vec3::X.cross(Vec3::Y), Vec3::Z);
        assert_eq!(Vec3::Y.cross(Vec3::X), -Vec3::Z);
        assert_eq!(a + b, Vec3::new(5.0, -3.0, 9.0));
        assert_eq!(a - b, Vec3::new(-3.0, 7.0, -3.0));
        assert_eq!(a * 2.0, Vec3::new(2.0, 4.0, 6.0));
        assert_eq!(a / 2.0, Vec3::new(0.5, 1.0, 1.5));
        assert_eq!(Vec3::new(3.0, 4.0, 0.0).length(), 5.0);
        assert_eq!(a.lerp(b, 0.0), a);
        assert_eq!(a.lerp(b, 1.0), b);
        assert_eq!(Vec3::ZERO.try_normalize(), None);
        let n = Vec3::new(3.0, 0.0, 0.0).try_normalize();
        assert_eq!(n, Some(Vec3::X));

        let d = DVec2::new(3.0, 4.0);
        assert_eq!(d.length(), 5.0);
        assert_eq!(d.distance(DVec2::ZERO), 5.0);
        assert_eq!(DVec4::splat(1.0).dot(DVec4::splat(2.0)), 8.0);

        let mut acc = Vec2::ZERO;
        acc += Vec2::new(1.0, 2.0);
        acc -= Vec2::new(0.5, 0.5);
        assert_eq!(acc, Vec2::new(0.5, 1.5));
    }

    #[test]
    fn quaternion_rotation() {
        let close = |a: DVec3, b: DVec3| (a - b).length() < 1e-12;
        assert!(DQuat::IDENTITY.rotate(DVec3::X) == DVec3::X);

        // 90° about Z: X -> Y -> -X.
        let half_pi = core::f64::consts::FRAC_PI_2;
        let q = DQuat::from_axis_angle(DVec3::Z, half_pi);
        assert!(close(q.rotate(DVec3::X), DVec3::Y));
        assert!(close(q.mul(q).rotate(DVec3::X), -DVec3::X));

        // Conjugate undoes the rotation.
        let v = DVec3::new(0.3, -1.2, 2.5);
        assert!(close(q.conjugate().rotate(q.rotate(v)), v));

        // from_axis_angle of a unit axis is unit-length.
        let axis = DVec3::new(1.0, 2.0, 3.0)
            .try_normalize()
            .unwrap_or(DVec3::X);
        let q2 = DQuat::from_axis_angle(axis, 0.7);
        assert!((q2.length() - 1.0).abs() < 1e-15);
        assert_eq!(DQuat::IDENTITY.try_normalize(), Some(DQuat::IDENTITY));
    }

    #[test]
    fn matrix_algebra() {
        let close = |a: DVec3, b: DVec3| (a - b).length() < 1e-12;
        assert_eq!(Mat3::IDENTITY.determinant(), 1.0);
        assert_eq!(
            Mat3::IDENTITY.mul_vec3(Vec3::new(1.0, 2.0, 3.0)),
            Vec3::new(1.0, 2.0, 3.0)
        );
        assert_eq!(Mat4::IDENTITY.mul(Mat4::IDENTITY), Mat4::IDENTITY);

        // Rotation via matrix matches rotation via quaternion.
        let q = DQuat::from_axis_angle(DVec3::Z, 0.7);
        let m = DMat3::from_quat(q);
        let v = DVec3::new(1.5, -0.25, 3.0);
        assert!(close(m.mul_vec3(v), q.rotate(v)));
        assert!((m.determinant() - 1.0).abs() < 1e-15);

        // Transpose of a rotation is its inverse.
        assert!(close(m.transpose().mul_vec3(m.mul_vec3(v)), v));
        assert_eq!(m.transpose().transpose(), m);

        // Mat4: translation applies to points, not vectors.
        let t = DVec3::new(10.0, 20.0, 30.0);
        let m4 = DMat4::from_rotation_translation(DQuat::IDENTITY, t);
        assert!(close(m4.transform_point3(v), v + t));
        assert!(close(m4.transform_vector3(v), v));
        assert_eq!(DMat4::from_translation(t).transform_point3(DVec3::ZERO), t);

        // Composition order: (A * B).transform == A.transform(B.transform).
        let a = DMat4::from_rotation_translation(q, t);
        let b = DMat4::from_translation(DVec3::new(-1.0, 2.0, -3.0));
        assert!(close(
            a.mul(b).transform_point3(v),
            a.transform_point3(b.transform_point3(v))
        ));
        assert_eq!(a.transpose().transpose(), a);
    }
}
