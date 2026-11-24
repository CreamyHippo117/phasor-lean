// phasor complex-math header: prepended to every kernel source (SPEC "GPU and CUDA plan").
// `real` is float by default and double when compiled with -DPHASOR_F64 (precision policy).
// Plain C++ that NVRTC and a host compiler both accept, so kernels can be checked on a
// machine without a GPU.
#ifdef PHASOR_F64
typedef double real;
#else
typedef float real;
#endif
#define R(x) ((real)(x))
#define PHASOR_PI 3.14159265358979323846

struct cplx { real re, im; };

static __device__ __forceinline__ cplx c_make(real re, real im) { cplx r; r.re = re; r.im = im; return r; }
static __device__ __forceinline__ cplx c_add(cplx a, cplx b) { return c_make(a.re + b.re, a.im + b.im); }
static __device__ __forceinline__ cplx c_sub(cplx a, cplx b) { return c_make(a.re - b.re, a.im - b.im); }
static __device__ __forceinline__ cplx c_neg(cplx a) { return c_make(-a.re, -a.im); }
static __device__ __forceinline__ cplx c_conj(cplx a) { return c_make(a.re, -a.im); }
static __device__ __forceinline__ cplx c_mul(cplx a, cplx b) {
    return c_make(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}
static __device__ __forceinline__ cplx c_div(cplx a, cplx b) {
    real d = b.re * b.re + b.im * b.im;
    return c_make((a.re * b.re + a.im * b.im) / d, (a.im * b.re - a.re * b.im) / d);
}
static __device__ __forceinline__ real c_abs(cplx a) { return (real)hypot(a.re, a.im); }
static __device__ __forceinline__ real c_arg(cplx a) { return (real)atan2(a.im, a.re); }
static __device__ __forceinline__ cplx c_exp(cplx a) {
    real e = (real)exp(a.re);
    return c_make(e * (real)cos(a.im), e * (real)sin(a.im));
}
static __device__ __forceinline__ cplx c_log(cplx a) { return c_make((real)log(c_abs(a)), c_arg(a)); }
static __device__ __forceinline__ cplx c_sqrt(cplx a) {
    real m = c_abs(a);
    real re = (real)sqrt(R(0.5) * (m + a.re) > 0 ? R(0.5) * (m + a.re) : 0);
    real im = (real)sqrt(R(0.5) * (m - a.re) > 0 ? R(0.5) * (m - a.re) : 0);
    return c_make(re, a.im < 0 ? -im : im);
}
static __device__ __forceinline__ cplx c_sin(cplx a) {
    return c_make((real)sin(a.re) * (real)cosh(a.im), (real)cos(a.re) * (real)sinh(a.im));
}
static __device__ __forceinline__ cplx c_cos(cplx a) {
    return c_make((real)cos(a.re) * (real)cosh(a.im), -(real)sin(a.re) * (real)sinh(a.im));
}
static __device__ __forceinline__ cplx c_tan(cplx a) { return c_div(c_sin(a), c_cos(a)); }
static __device__ __forceinline__ cplx c_sinh(cplx a) {
    return c_make((real)sinh(a.re) * (real)cos(a.im), (real)cosh(a.re) * (real)sin(a.im));
}
static __device__ __forceinline__ cplx c_cosh(cplx a) {
    return c_make((real)cosh(a.re) * (real)cos(a.im), (real)sinh(a.re) * (real)sin(a.im));
}
static __device__ __forceinline__ cplx c_tanh(cplx a) { return c_div(c_sinh(a), c_cosh(a)); }
// Integer power by repeated squaring; negative exponents invert.
static __device__ __forceinline__ cplx c_powi(cplx a, int n) {
    int k = n < 0 ? -n : n;
    cplx r = c_make(R(1), R(0));
    while (k > 0) {
        if (k & 1) r = c_mul(r, a);
        a = c_mul(a, a);
        k >>= 1;
    }
    return n < 0 ? c_div(c_make(R(1), R(0)), r) : r;
}
// General power on the principal branch; 0 ** w is 0.
static __device__ __forceinline__ cplx c_pow(cplx a, cplx w) {
    if (a.re == 0 && a.im == 0) return c_make(R(0), R(0));
    return c_exp(c_mul(w, c_log(a)));
}
