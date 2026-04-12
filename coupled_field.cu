#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>

#ifdef _WIN32
#include <windows.h>
#include <conio.h>
static void enable_ansi(void) {
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode = 0;
    GetConsoleMode(h, &mode);
    SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
}
static int poll_key_nonblock(void) {
    return _kbhit() ? _getch() : -1;
}
static void cli_sleep_ms(int ms) {
    Sleep((DWORD)ms);
}
#else
static void enable_ansi(void) {}
static int poll_key_nonblock(void) { return -1; }
static void cli_sleep_ms(int ms) { (void)ms; }
#endif

/* ── dashboard ──────────────────────────────────────────────────────────── */
#define ANSI_CLEAR   "\033[2J\033[H"
#define ANSI_BOLD    "\033[1m"
#define ANSI_DIM     "\033[2m"
#define ANSI_CYAN    "\033[36m"
#define ANSI_GREEN   "\033[32m"
#define ANSI_YELLOW  "\033[33m"
#define ANSI_RED     "\033[31m"
#define ANSI_RESET   "\033[0m"

static void draw_bar(FILE* f, double val, double max_val, int width, char fill) {
    int filled = (max_val > 0) ? (int)(val / max_val * width) : 0;
    if (filled > width) filled = width;
    fprintf(f, ANSI_GREEN);
    for (int i = 0; i < filled; i++) fputc(fill, f);
    fprintf(f, ANSI_DIM);
    for (int i = filled; i < width; i++) fputc(0xB7, f); /* middle dot */
    fprintf(f, ANSI_RESET);
}

static void fmt_num(char* buf, size_t n, double v) {
    if      (v >= 1e9)  snprintf(buf, n, "%.2fB", v/1e9);
    else if (v >= 1e6)  snprintf(buf, n, "%.2fM", v/1e6);
    else if (v >= 1e3)  snprintf(buf, n, "%.2fK", v/1e3);
    else                snprintf(buf, n, "%.4g",  v);
}

typedef struct {
    int     tick;
    int     ticks_goal;
    unsigned long long active_cells;
    double  wave_energy;
    double  delta_energy;
    int     edge_count;
    int     natural_count;
    int     snapshots;
    float   grad_norm;         /* global gradient norm — clipped above 1.0 */
    int     alerts;            /* bitmask: 1=collapse 2=diverge */
    char    device_name[64];
    float   tick_ms;           /* ms per tick (rolling) */
    double  nca_loss;          /* mean squared prediction error */
    double  nca_residual;      /* mean residual magnitude injected into wave */
    double  nca_loss_prev;     /* previous loss for trend arrow */
    float   tune_c2;
    float   tune_damping;
    float   tune_drive;
    float   tune_kwtp;
    float   tune_fb;
    float   tune_gravity;
    float   tune_lr;
    float   tune_triad;
    int     paused;
    int     interactive;
} DashState;

typedef struct {
    float c2;
    float damping;
    float drive;
    float kwtp;
    float feedback;
    float gravity;
    float lr;
    float triad;
    int paused;
    int step_once;
} LiveTunables;

static void draw_dashboard(const DashState* d) {
    const int cells_total = 1920 * 1080;
    double pct = (d->ticks_goal > 0) ? (d->tick * 100.0 / d->ticks_goal) : 0;
    double act_pct = (d->active_cells * 100.0 / cells_total);
    char nbuf[32];

    fputs(ANSI_CLEAR, stdout);

    /* header */
    fprintf(stdout, ANSI_BOLD ANSI_CYAN
        "  ╔══════════════════════════════════════════╗\n"
        "  ║       COUPLED FIELD COMPUTER             ║\n"
        "  ╚══════════════════════════════════════════╝\n"
        ANSI_RESET);
    fprintf(stdout, "  " ANSI_DIM "device: %s" ANSI_RESET "\n\n", d->device_name);

    /* progress */
    fprintf(stdout, "  " ANSI_BOLD "Tick" ANSI_RESET "  %d / %d  (%.1f%%)\n  ",
            d->tick, d->ticks_goal, pct);
    draw_bar(stdout, d->tick, d->ticks_goal, 42, 0xDB);
    fprintf(stdout, "\n\n");

    /* active cells */
    fmt_num(nbuf, sizeof(nbuf), (double)d->active_cells);
    fprintf(stdout, "  " ANSI_BOLD "Active cells" ANSI_RESET "  %s / %.1fM  (%.1f%%)\n  ",
            nbuf, cells_total/1e6, act_pct);
    draw_bar(stdout, d->active_cells, cells_total, 42, 0xB2);
    fprintf(stdout, "\n\n");

    /* energies */
    fmt_num(nbuf, sizeof(nbuf), d->wave_energy);
    fprintf(stdout, "  " ANSI_BOLD "Wave energy " ANSI_RESET "  %-10s", nbuf);
    fmt_num(nbuf, sizeof(nbuf), d->delta_energy);
    fprintf(stdout, "  " ANSI_BOLD "Delta energy" ANSI_RESET "  %s\n\n", nbuf);

    /* entanglement */
    fprintf(stdout, "  " ANSI_BOLD "Edges      " ANSI_RESET "  %-6d"
                    "      " ANSI_BOLD "Natural ent." ANSI_RESET "  %d"
                    "      " ANSI_BOLD "Snapshots" ANSI_RESET "  %d\n\n",
            d->edge_count, d->natural_count, d->snapshots);

    /* NCA learning */
    {
        const char* trend = "  ";
        if (d->nca_loss_prev > 0) {
            if (d->nca_loss < d->nca_loss_prev * 0.999)       trend = ANSI_GREEN "▼" ANSI_RESET " ";
            else if (d->nca_loss > d->nca_loss_prev * 1.001)  trend = ANSI_RED   "▲" ANSI_RESET " ";
            else                                               trend = ANSI_DIM   "─" ANSI_RESET " ";
        }
        fmt_num(nbuf, sizeof(nbuf), d->nca_loss);
        fprintf(stdout, "  " ANSI_BOLD "NCA loss   " ANSI_RESET "  %s%-10s", trend, nbuf);
        fmt_num(nbuf, sizeof(nbuf), d->nca_residual);
        fprintf(stdout, "  " ANSI_BOLD "Residual   " ANSI_RESET "  %s\n", nbuf);
        if (d->grad_norm > 1.0f)
            fprintf(stdout, "  " ANSI_YELLOW "  grad norm %.3f — clipping active" ANSI_RESET "\n", d->grad_norm);
        else
            fprintf(stdout, "  " ANSI_DIM    "  grad norm %.3f" ANSI_RESET "\n", d->grad_norm);
        fprintf(stdout, "\n");
    }

    /* speed */
    if (d->tick_ms > 0)
        fprintf(stdout, "  " ANSI_BOLD "Speed" ANSI_RESET "  %.1f ms/tick  (~%.0f ticks/sec)\n\n",
                d->tick_ms, 1000.0f / d->tick_ms);

    fprintf(stdout,
        "  Tune  c2=%.4f  damp=%.5f  drive=%.7g  kwtp=%.4f  fb=%.6f  grav=%.7f  lr=%.6f  triad=%.6f\n",
        d->tune_c2, d->tune_damping, d->tune_drive, d->tune_kwtp, d->tune_fb, d->tune_gravity, d->tune_lr, d->tune_triad);
    fprintf(stdout,
        "  Keys: q/a=c2  w/s=damp  e/d=drive  r/f=kwtp  t/g=fb  y/h=grav  u/j=lr  i/k=triad  SPC=pause  n=step  x=quit\n");
    if (d->paused) {
        fprintf(stdout, "  " ANSI_YELLOW "PAUSED" ANSI_RESET " (space=run  n=step)\n\n");
    } else {
        fprintf(stdout, "\n");
    }

    /* alerts */
    if (d->alerts & 1)
        fprintf(stdout, "  " ANSI_YELLOW "⚠  delta collapse — field quiet" ANSI_RESET "\n");
    if (d->alerts & 2)
        fprintf(stdout, "  " ANSI_RED    "⚠  wave divergence >1e10" ANSI_RESET "\n");
    if (!d->alerts)
        fprintf(stdout, "  " ANSI_GREEN  "✓  field stable" ANSI_RESET "\n");

    fprintf(stdout, "\n  " ANSI_DIM "Ctrl+C to stop" ANSI_RESET "\n");
    fflush(stdout);
}

/*
Build:
  nvcc -O3 -arch=sm_89 -use_fast_math -lineinfo -o coupled_field coupled_field.cu

Quantum seed:
  ./coupled_field --seed quantum C:/Users/jwest/.gemini/antigravity/scratch/SHA256_Analyzer/quantum_seeds.log
*/

/* ---------- quantum seed parser ------------------------------------------ */
/* Returns a malloc'd byte pool from REAPED lines in the ANU QRNG daemon log.
   Caller must free(). Sets *out_len to number of bytes available. */
static uint8_t* load_quantum_bytes(const char* log_path, size_t* out_len) {
    *out_len = 0;
    FILE* f = fopen(log_path, "r");
    if (!f) {
        fprintf(stderr, "[quantum] cannot open %s\n", log_path);
        return NULL;
    }

    /* Allocate a growing pool. Each REAPED line yields up to 1024 bytes. */
    size_t cap = 1024 * 1024; /* 1 MB initial */
    uint8_t* pool = (uint8_t*)malloc(cap);
    if (!pool) { fclose(f); return NULL; }

    char line[8192];
    size_t total = 0;
    while (fgets(line, sizeof(line), f)) {
        /* Match lines containing "REAPED" and "Top Seeds:" */
        char* start = strstr(line, "Top Seeds: [");
        if (!start) continue;
        start += strlen("Top Seeds: [");
        /* Walk the comma-separated integers until ']' */
        char* p = start;
        while (*p && *p != ']') {
            while (*p == ' ' || *p == ',') p++;
            if (*p == ']' || *p == '\0') break;
            int val = atoi(p);
            /* Grow pool if needed */
            if (total >= cap) {
                cap *= 2;
                uint8_t* np = (uint8_t*)realloc(pool, cap);
                if (!np) goto done;
                pool = np;
            }
            pool[total++] = (uint8_t)(val & 0xFF);
            /* advance past digits */
            while (*p && *p != ',' && *p != ']') p++;
        }
    }
done:
    fclose(f);
    *out_len = total;
    fprintf(stderr, "[quantum] loaded %zu bytes from %s\n", total, log_path);
    return pool;
}

#define WIDTH 1920
#define HEIGHT 1080
#define CELLS ((size_t)WIDTH * (size_t)HEIGHT)

/* 256-bin color→frequency lookup table in constant memory.
   Each hue bin gets a unit 4D vector. Properties:
   - bin[i] = -bin[i+128]  (complementary hues fully cancel)
   - bin[i]·bin[i+85] = cos(120°) = -0.5  (triadic hues partially interfere)
   - Populated at startup by init_freq_table() */
__constant__ float4 c_freq_table[256];

static void init_freq_table(void) {
    float4 h_table[256];
    const float TRI = 2.09439510239f; /* 2π/3 */
    for (int i = 0; i < 256; i++) {
        float theta = (float)i * (6.28318530718f / 256.0f);
        /* 4D unit vector on hue circle using triadic basis.
           ch0,ch1,ch2 = triadic projections, ch3 = quadrature.
           f(theta+pi) = -f(theta) guaranteed by odd symmetry of sin. */
        float x = sinf(theta);
        float y = sinf(theta + TRI);
        float z = sinf(theta + TRI * 2.0f);
        float w = cosf(theta);
        /* Normalize to unit length */
        float inv_len = 1.0f / sqrtf(x*x + y*y + z*z + w*w);
        h_table[i] = make_float4(x*inv_len, y*inv_len, z*inv_len, w*inv_len);
    }
    cudaMemcpyToSymbol(c_freq_table, h_table, sizeof(h_table));
}

#define BX 32
#define BY 8
#define THREADS (BX * BY)
#define TILE_W (BX + 2)
#define TILE_H (BY + 2)

#define MAX_SUBSTRATES 16
#define HIST_TICKS 8
#define NCA_IN 136
#define NCA_H 32
#define NCA_OUT 8
#define NCA_W1 (NCA_IN * NCA_H)
#define NCA_B1 (NCA_H)
#define NCA_W2 (NCA_H * NCA_OUT)
#define NCA_B2 (NCA_OUT)
#define NCA_PARAMS (NCA_W1 + NCA_B1 + NCA_W2 + NCA_B2)

#define EXPERIMENT_TICKS 256
#define EXPERIMENT_PERIOD 128
#define COMB_PERIOD 65536
#define SNAPSHOT_PERIOD 256
#define NCA_DUMP_PERIOD 65536

#define EDGE_DENSITY 0.01f
#define DEFAULT_SUBSTRATES 2
#define MAX_CANDIDATES 262144

/* Collide-interference strength multiplier: kwtp × this factor determines
   how strongly destructive wave zero-crossings flip pixels toward complement.
   Higher = more entropy/chaos; lower = slower complementary disruption. */
#define COLLIDE_STRENGTH_MULT 2.0f

#define BASE_C2 0.18f
#define BASE_DAMPING 0.9998f
#define BASE_DRIVE 0.002f       /* increased: stronger hue-frequency injection into wave */
#define BASE_KWTP 0.020f
#define BASE_FEEDBACK 0.00005f
#define BASE_GRAVITY 0.006f     /* increased: faster hue-cluster accumulation */
#define BASE_LR 0.001f          /* Adam default — much higher than SGD needs */

/* Adam hyperparameters — kept fixed (not tunable) */
#define ADAM_BETA1 0.9f
#define ADAM_BETA2 0.999f
#define ADAM_EPS   1.0e-8f

/* NCA gradient accumulation: run NCA forward/backward every NCA_FWD_PERIOD
   ticks, but only update weights every NCA_UPDATE_PERIOD ticks.
   NCA_UPDATE_PERIOD / NCA_FWD_PERIOD = number of backward passes accumulated
   per weight update (effective mini-batch over time). */
#define NCA_FWD_PERIOD    4     /* run NCA forward + backward every N ticks */
#define NCA_UPDATE_PERIOD 16    /* apply Adam step every N ticks (= 4 accumulations) */
/* NCA_ACCUMULATIONS = number of backward passes accumulated per Adam update */
#define NCA_ACCUMULATIONS (NCA_UPDATE_PERIOD / NCA_FWD_PERIOD)

/* Chromatic saturation thresholds — shared across kernels */
#define MIN_SATURATION_ACHROMATIC 0.05f  /* below this S, pixel is considered achromatic */
#define GRAVITY_MIN_SATURATION    0.75f  /* pixel_gravity_kernel saturation floor */
#define COMPLEMENT_MIN_SATURATION 0.70f  /* complementary_neighbor_kernel sat floor */
#define FLIP_MIN_SATURATION       0.75f  /* xor_broadcast_kernel saturation during flip */
#define COLLIDE_MIN_SATURATION    0.80f  /* collide_interference_kernel saturation boost */
#define MIN_COLLIDE_MAGNITUDE     0.02f  /* minimum wave zero-crossing strength to act */

#define CHECK_CUDA(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while (0)

typedef struct {
    int ticks;
    int device;
    int substrates;
    int seed_mode; /* 0=random 1=impulse 2=sparse 3=quantum */
    int interactive;
    int snap_every;      /* write pixel PNG every N ticks (0 = use SNAPSHOT_PERIOD) */
    char snap_dir[512];  /* directory for PNG output (empty = current dir) */
    char resume_path[512];
    char quantum_log[512];
} HostOptions;

typedef struct {
    void* pixel_prev;
    void* pixel_curr;
    void* pixel_next;

    void* wave_prev;
    void* wave_curr;
    void* wave_next;

    void* delta_curr;
    void* delta_next;
    void* delta_mag;

    void* c2;
    void* damping;
    void* drive_gain;
    void* k_wave_to_px;
    void* feedback_gain;

    /* Comb accumulators */
    uint32_t* act_count;
    float* ewma_delta;
    float* dir_hist; /* [cells][8] */

    /* NCA per-substrate state */
    float* ring;      /* [HIST][cells][8] */
    int ring_head;
    float* input;     /* [cells][136] */
    float* hidden;    /* [cells][32] */
    float* output;    /* [cells][8] */
    float* target;    /* [cells][8] */
    float* prev_target; /* [cells][8], one-tick-lag target state */
    float* residual;  /* [cells][8] */

    size_t pixel_pitch;
    size_t wave_pitch;
    size_t delta_pitch;
    size_t scalar_pitch;
} Plane;

typedef struct {
    int id;
    int active;
    int write_locked;
    int temporary;
    int parent;
    Plane p;
} Substrate;

typedef struct {
    int active_count;
    Substrate s[MAX_SUBSTRATES];
} SubstratePool;

typedef struct {
    uint16_t src_sub;
    uint16_t dst_sub;
    uint32_t src_x;
    uint32_t src_y;
    uint32_t dst_x;
    uint32_t dst_y;
    float strength;
} EntEdge;

typedef struct {
    uint32_t a;
    uint32_t b;
    float score;
} PairCandidate;

typedef struct {
    uint64_t tick;
    uint32_t a;
    uint32_t b;
    float baseline;
    float experiment;
} NaturalEntRecord;

typedef struct {
    const void* px_next;
    const void* wv_next;
    size_t px_pitch;
    size_t wv_pitch;
} DevSrcView;

typedef struct {
    void* px_next;
    void* wv_next;
    size_t px_pitch;
    size_t wv_pitch;
} DevDstView;

__host__ __device__ __forceinline__
float4* row_f4(void* base, size_t pitch, int y) {
    return (float4*)((char*)base + (size_t)y * pitch);
}

__host__ __device__ __forceinline__
const float4* row_f4_const(const void* base, size_t pitch, int y) {
    return (const float4*)((const char*)base + (size_t)y * pitch);
}

__host__ __device__ __forceinline__
float* row_f1(void* base, size_t pitch, int y) {
    return (float*)((char*)base + (size_t)y * pitch);
}

__host__ __device__ __forceinline__
const float* row_f1_const(const void* base, size_t pitch, int y) {
    return (const float*)((const char*)base + (size_t)y * pitch);
}

__device__ __forceinline__ float clamp01(float x) { return fminf(1.0f, fmaxf(0.0f, x)); }
__device__ __forceinline__ int clampi(int v, int lo, int hi) { return max(lo, min(hi, v)); }

__device__ __forceinline__ void rgb_to_hsv(float3 rgb, float* h, float* s, float* v) {
    float r = rgb.x, g = rgb.y, b = rgb.z;
    float cmax = fmaxf(r, fmaxf(g, b));
    float cmin = fminf(r, fminf(g, b));
    float d = cmax - cmin;
    *v = cmax;
    *s = (cmax > 1.0e-8f) ? (d / cmax) : 0.0f;
    float hh = 0.0f;
    if (d > 1.0e-8f) {
        if (cmax == r) hh = fmodf((g - b) / d, 6.0f);
        else if (cmax == g) hh = (b - r) / d + 2.0f;
        else hh = (r - g) / d + 4.0f;
        if (hh < 0.0f) hh += 6.0f;
    }
    *h = hh / 6.0f;
}

/* XOR broadcast system — two kernel passes per tick.
   Pass 1: xor_collect_kernel — finds every wave crossing point, records the
           trigger color (the pixel sitting under that crossing) into a small
           accumulator buffer: one float4 per XOR slot.
   Pass 2: xor_broadcast_kernel — for every pixel on the canvas, measures how
           closely it matches any recorded trigger color and applies the
           complement transformation weighted by that match.
   Result: a single XOR crossing fires the transformation on every instance of
           that color across the entire canvas simultaneously. Clusters act as
           single logical units — a crossing on any member fires all members. */

#define MAX_XOR_SLOTS 64   /* max distinct trigger colors captured per tick */

/* Compute wave Laplacian into a flat float4 buffer for XOR crossing detection */
__global__ void wave_laplacian_kernel(
    const void* wv_base, size_t wv_pitch,
    float4* out_lap,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int xl = max(x-1,0), xr = min(x+1,width-1);
    int yt = max(y-1,0), yb = min(y+1,height-1);
    float4 c  = row_f4_const(wv_base, wv_pitch, y)[x];
    float4 l  = row_f4_const(wv_base, wv_pitch, y)[xl];
    float4 r  = row_f4_const(wv_base, wv_pitch, y)[xr];
    float4 t  = row_f4_const(wv_base, wv_pitch, yt)[x];
    float4 b  = row_f4_const(wv_base, wv_pitch, yb)[x];
    out_lap[y * width + x] = make_float4(
        l.x+r.x+t.x+b.x - 4.0f*c.x,
        l.y+r.y+t.y+b.y - 4.0f*c.y,
        l.z+r.z+t.z+b.z - 4.0f*c.z,
        l.w+r.w+t.w+b.w - 4.0f*c.w);
}

/* Pass 1 — collect XOR trigger colors from crossing points.
   xor_buf layout: [slot*5 + 0..3] = RGBA trigger color,
                   [slot*5 + 4]    = crossing strength */
__global__ void xor_collect_kernel(
    const void*  px_base,  size_t px_pitch,
    const float4* wv_lap_buf,          /* pre-computed Laplacian field */
    float*       xor_buf,
    int*         xor_count,
    float        kwtp,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float4 wlap = wv_lap_buf[y * width + x];
    float lap_mag = sqrtf(wlap.x*wlap.x + wlap.y*wlap.y + wlap.z*wlap.z + wlap.w*wlap.w);
    float strength = lap_mag * kwtp;
    if (strength < 0.08f) return;   /* lowered threshold — fires more often */

    float4 px = row_f4_const(px_base, px_pitch, y)[x];

    /* Claim a slot atomically */
    int slot = atomicAdd(xor_count, 1);
    if (slot >= MAX_XOR_SLOTS) return;

    xor_buf[slot * 5 + 0] = clamp01(px.x);
    xor_buf[slot * 5 + 1] = clamp01(px.y);
    xor_buf[slot * 5 + 2] = clamp01(px.z);
    xor_buf[slot * 5 + 3] = clamp01(px.w);
    xor_buf[slot * 5 + 4] = fminf(1.0f, strength);
}

/* Pass 2 — broadcast HSV complement transformation to all matching pixels.
   For each pixel, check if its hue matches:
     (A) the trigger hue → flip to complement (hue + π)
     (B) the complement-neighbor zone (hue within 15°–65° of trigger complement)
         → those pixels also flip to their complement (= toward subject zone)
   This implements: subject+neighbors expel the complement; complement pair
   expels the neighbors; both groups shift toward each other in XOR fashion. */
__global__ void xor_broadcast_kernel(
    const void* px_cur_base, size_t px_cur_pitch,
    void*       px_nxt_base, size_t px_nxt_pitch,
    const float* xor_buf,
    int          xor_count,
    float        broadcast_strength,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int n = min(xor_count, MAX_XOR_SLOTS);
    if (n == 0) return;

    float4 me = row_f4_const(px_cur_base, px_cur_pitch, y)[x];
    float mr = clamp01(me.x), mg = clamp01(me.y), mb = clamp01(me.z);
    float my_h, my_s, my_v;
    rgb_to_hsv_full(mr, mg, mb, &my_h, &my_s, &my_v);

    const float PI = 3.14159265358979f;
    float hue_flip_weight = 0.0f;

    /* Guard against achromatic (gray) pixels — rgb_to_hue_rad returns 0 for
       r=g=b.  We skip the flip for near-achromatic pixels since they have no
       meaningful hue to preserve and adding a hue rotation to gray produces
       oversaturated color spikes. */
    if (my_s < MIN_SATURATION_ACHROMATIC) return;

    for (int s = 0; s < n; ++s) {
        float tr = xor_buf[s*5+0];
        float tg = xor_buf[s*5+1];
        float tb = xor_buf[s*5+2];
        float xstr = xor_buf[s*5+4];
        float t_h = rgb_to_hue_rad(tr, tg, tb);
        /* Skip achromatic trigger colors */
        float t_v = fmaxf(tr, fmaxf(tg, tb));
        float t_mn = fminf(tr, fminf(tg, tb));
        if ((t_v < 1e-6f) || (t_v > 1e-6f && (t_v - t_mn) / t_v < MIN_SATURATION_ACHROMATIC)) continue;

        /* --- Group A: pixels near the trigger hue (subject zone, within ~30°) --- */
        float dh_a = my_h - t_h;
        if (dh_a >  PI) dh_a -= 2.0f * PI;
        if (dh_a < -PI) dh_a += 2.0f * PI;
        float sim_a = fmaxf(0.0f, 1.0f - fabsf(dh_a) * (1.0f / 0.52f));
        sim_a = sim_a * sim_a;

        /* --- Group B: complement-neighbor zone (15°–65° from trigger complement) --- */
        float comp_h = t_h + PI;
        float dh_from_comp = my_h - comp_h;
        if (dh_from_comp >  PI) dh_from_comp -= 2.0f * PI;
        if (dh_from_comp < -PI) dh_from_comp += 2.0f * PI;
        float d_fc = fabsf(dh_from_comp);
        const float NB_INNER = 0.26f; /* ~15° inner dead zone (exact complement handled by gravity) */
        const float NB_OUTER = 1.13f; /* ~65° outer edge of complement-neighbor band */
        float sim_b = 0.0f;
        if (d_fc > NB_INNER && d_fc < NB_OUTER) {
            float t = (d_fc - NB_INNER) / (NB_OUTER - NB_INNER);
            sim_b = sinf(t * PI); /* smooth bump, 0 at edges, 1 at middle */
        }

        float w = (sim_a + sim_b * 0.80f) * xstr * broadcast_strength;
        if (w > 0.0f) hue_flip_weight += w;
    }

    /* Normalize so N simultaneous triggers don't stack unboundedly */
    float flip = fminf(1.0f, hue_flip_weight);
    if (flip < 0.001f) return;

    /* HSV complement flip: rotate hue by π × flip_amount, preserve saturation+value.
       This keeps colors vivid throughout the flip — no desaturation from RGB blending. */
    float flipped_h = my_h + flip * PI;
    float new_s = fmaxf(my_s, FLIP_MIN_SATURATION); /* keep saturated through the flip */
    float3 flipped_rgb = hsv_to_rgb_full(flipped_h, new_s, my_v);

    float4 out = row_f4(px_nxt_base, px_nxt_pitch, y)[x];
    out.x = clamp01(clamp01(out.x) + (flipped_rgb.x - mr) * flip);
    out.y = clamp01(clamp01(out.y) + (flipped_rgb.y - mg) * flip);
    out.z = clamp01(clamp01(out.z) + (flipped_rgb.z - mb) * flip);
    row_f4(px_nxt_base, px_nxt_pitch, y)[x] = out;
}

/* ---------- HSV <-> RGB conversions --------------------------------------- */
/* hsv: h in [0, 2π), s in [0,1], v in [0,1] */
__device__ __forceinline__ float3 hsv_to_rgb_full(float h, float s, float v) {
    float h6 = h * (3.0f / 3.14159265358979f); /* [0,6) */
    h6 = h6 - floorf(h6 / 6.0f) * 6.0f;       /* wrap to [0,6) */
    float c = v * s;
    float x = c * (1.0f - fabsf(fmodf(h6, 2.0f) - 1.0f));
    float m = v - c;
    float r, g, b;
    if      (h6 < 1.0f) { r=c; g=x; b=0; }
    else if (h6 < 2.0f) { r=x; g=c; b=0; }
    else if (h6 < 3.0f) { r=0; g=c; b=x; }
    else if (h6 < 4.0f) { r=0; g=x; b=c; }
    else if (h6 < 5.0f) { r=x; g=0; b=c; }
    else                 { r=c; g=0; b=x; }
    return make_float3(r+m, g+m, b+m);
}

__device__ __forceinline__ void rgb_to_hsv_full(float r, float g, float b,
                                                 float* h, float* s, float* v) {
    float M = fmaxf(r, fmaxf(g, b));
    float m = fminf(r, fminf(g, b));
    float C = M - m;
    *v = M;
    *s = (M < 1e-6f) ? 0.0f : C / M;
    if (C < 1e-6f) { *h = 0.0f; return; }
    float hh;
    if      (M == r) hh = fmodf((g - b) / C + 6.0f, 6.0f);
    else if (M == g) hh = (b - r) / C + 2.0f;
    else             hh = (r - g) / C + 4.0f;
    *h = hh * (3.14159265358979f / 3.0f); /* [0, 2π) */
}

__device__ __forceinline__ float rgb_to_hue_rad(float r, float g, float b) {
    float M = fmaxf(r, fmaxf(g, b));
    float m = fminf(r, fminf(g, b));
    float C = M - m;
    if (C < 1e-6f) return 0.0f;
    float h;
    if (M == r)      h = fmodf((g - b) / C + 6.0f, 6.0f);
    else if (M == g) h = (b - r) / C + 2.0f;
    else             h = (r - g) / C + 4.0f;
    h *= (3.14159265358979f / 3.0f);
    return h;
}

__device__ __forceinline__ float3 hue_to_rgb(float h) {
    float hh = h - floorf(h);
    float h6 = hh * 6.0f;
    float x = 1.0f - fabsf(fmodf(h6, 2.0f) - 1.0f);
    if (h6 < 1.0f) return make_float3(1.f, x, 0.f);
    if (h6 < 2.0f) return make_float3(x, 1.f, 0.f);
    if (h6 < 3.0f) return make_float3(0.f, 1.f, x);
    if (h6 < 4.0f) return make_float3(0.f, x, 1.f);
    if (h6 < 5.0f) return make_float3(x, 0.f, 1.f);
    return make_float3(1.f, 0.f, x);
}

__global__ __launch_bounds__(THREADS)
void seed_kernel(void* px_base, size_t px_pitch,
                 void* wv_base, size_t wv_pitch,
                 int mode, uint32_t seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;

    float4* p = row_f4(px_base, px_pitch, y);
    float4* w = row_f4(wv_base, wv_pitch, y);

    if (mode == 1) {
        p[x] = make_float4(0.f, 0.f, 0.f, 1.f);
        w[x] = make_float4(0.f, 0.f, 0.f, 0.f);
        if (x == WIDTH / 2 && y == HEIGHT / 2) {
            p[x] = make_float4(1.f, 1.f, 1.f, 1.f);
        }
        return;
    }

    uint32_t s = seed ^ (uint32_t)(x * 73856093u) ^ (uint32_t)(y * 19349663u);
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;

    /* Seed in HSV so every pixel starts as a vivid, fully-saturated color.
       Random hue across the full wheel, saturation 0.85-1.0, value 0.7-1.0.
       This ensures yellows, oranges, purples etc are present from tick 0. */
    float hue = (float)(s & 0xFFFFu) * (6.28318530f / 65536.0f); /* [0, 2π) */
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    float sat = 0.85f + (float)(s & 63u) * (0.15f / 63.0f);      /* [0.85, 1.0] */
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    float val = 0.70f + (float)(s & 63u) * (0.30f / 63.0f);      /* [0.70, 1.0] */

    float3 rgb = hsv_to_rgb_full(hue, sat, val);

    if (mode == 2) {
        float keep = ((s & 31u) == 0u) ? 1.0f : 0.0f;
        p[x] = make_float4(rgb.x * keep, rgb.y * keep, rgb.z * keep, 1.0f);
    } else {
        p[x] = make_float4(rgb.x, rgb.y, rgb.z, 1.0f);
    }
    w[x] = make_float4(0.f, 0.f, 0.f, 0.f);
}

/* quantum_seed_kernel: initialize pixel field from a pre-uploaded byte pool.
   Each pixel consumes 3 bytes (R,G,B); if pool exhausted, falls back to xorshift. */
__global__ __launch_bounds__(THREADS)
void quantum_seed_kernel(void* px_base, size_t px_pitch,
                         void* wv_base, size_t wv_pitch,
                         const uint8_t* qbytes, size_t qlen) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;

    float4* pix = row_f4(px_base, px_pitch, y);
    float4* wv  = row_f4(wv_base,  wv_pitch,  y);

    size_t idx = (size_t)y * WIDTH + (size_t)x;
    size_t byte_base = idx * 3;

    /* Still pond: low-amplitude disturbances. Quantum bytes set hue/direction,
       but amplitude stays small — vectors can cross naturally as energy builds. */
    /* Use quantum bytes to seed hue — full saturation vivid colors like random seed */
    uint32_t s;
    if (byte_base + 2 < qlen) {
        s = ((uint32_t)qbytes[byte_base] << 16) |
            ((uint32_t)qbytes[byte_base+1] << 8) |
             (uint32_t)qbytes[byte_base+2];
    } else {
        s = 0x12345678u;
        if (byte_base     < qlen) s ^= (uint32_t)qbytes[byte_base]     << 24;
        if (byte_base + 1 < qlen) s ^= (uint32_t)qbytes[byte_base + 1] << 16;
        s ^= (uint32_t)idx * 0x9E3779B9u;
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    }
    float hue = (float)(s & 0xFFFFu) * (6.28318530f / 65536.0f);
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    float sat = 0.85f + (float)(s & 63u) * (0.15f / 63.0f);
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    float val = 0.70f + (float)(s & 63u) * (0.30f / 63.0f);
    float3 rgb = hsv_to_rgb_full(hue, sat, val);
    pix[x] = make_float4(rgb.x, rgb.y, rgb.z, 1.0f);
    wv[x]  = make_float4(0.f, 0.f, 0.f, 0.f);
}

__global__ __launch_bounds__(THREADS)
void init_params_kernel(void* c2_base, size_t c2_pitch,
                        void* damping_base, size_t damping_pitch,
                        void* drive_base, size_t drive_pitch,
                        void* kwtp_base, size_t kwtp_pitch,
                        void* fb_base, size_t fb_pitch,
                        float c2, float damping, float drive, float kwtp, float fb) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;

    /* Per-cell variation — breaks global synchrony so regions evolve
       at their own rate. Each cell gets a unique local phase offset. */
    uint32_t s = (uint32_t)(y * 1920 + x) * 0x9E3779B9u;
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    float n0 = (float)(s & 0xFFFFu) / 65535.0f; /* 0..1 */

    /* c2: wave speed varies ±20% per cell */
    row_f1(c2_base, c2_pitch, y)[x] = c2 * (0.80f + n0 * 0.40f);

    /* damping: varies ±0.001 per cell */
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    float m0 = (float)(s & 0xFFFFu) / 65535.0f;
    row_f1(damping_base, damping_pitch, y)[x] = damping - 0.001f + m0 * 0.002f;

    /* drive: varies ±30% — local color injection strength differs */
    row_f1(drive_base, drive_pitch, y)[x] = drive * (0.70f + n0 * 0.60f);
    row_f1(kwtp_base,  kwtp_pitch,  y)[x] = kwtp;
    row_f1(fb_base,    fb_pitch,    y)[x] = fb;
}

__global__ __launch_bounds__(THREADS)
void physics_tick_kernel(const void* px_cur_base, size_t px_cur_pitch,
                         const void* px_prev_base, size_t px_prev_pitch,
                         const void* wv_cur_base, size_t wv_cur_pitch,
                         const void* wv_prev_base, size_t wv_prev_pitch,
                         void* px_next_base, size_t px_next_pitch,
                         void* wv_next_base, size_t wv_next_pitch,
                         void* delta_next_base, size_t delta_next_pitch,
                         void* delta_mag_base, size_t delta_mag_pitch,
                         const void* c2_base, size_t c2_pitch,
                         const void* damping_base, size_t damping_pitch,
                         const void* drive_base, size_t drive_pitch,
                         const void* kwtp_base, size_t kwtp_pitch) {
    __shared__ float4 s_px[TILE_H][TILE_W];
    __shared__ float4 s_wv[TILE_H][TILE_W];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gx = blockIdx.x * BX + tx;
    int gy = blockIdx.y * BY + ty;

    int lid = ty * BX + tx;
    int total = TILE_W * TILE_H;
    for (int i = lid; i < total; i += THREADS) {
        int lx = i % TILE_W;
        int ly = i / TILE_W;
        int sx = blockIdx.x * BX + lx - 1;
        int sy = blockIdx.y * BY + ly - 1;
        sx = clampi(sx, 0, WIDTH - 1);
        sy = clampi(sy, 0, HEIGHT - 1);
        s_px[ly][lx] = __ldg(&row_f4_const(px_cur_base, px_cur_pitch, sy)[sx]);
        /* Sanitize wave at read — catches any injection from gravity/NCA since last tick */
        float4 _wv = __ldg(&row_f4_const(wv_cur_base, wv_cur_pitch, sy)[sx]);
        _wv.x = tanhf(_wv.x * 0.5f) * 2.0f;
        _wv.y = tanhf(_wv.y * 0.5f) * 2.0f;
        _wv.z = tanhf(_wv.z * 0.5f) * 2.0f;
        _wv.w = tanhf(_wv.w * 0.5f) * 2.0f;
        s_wv[ly][lx] = _wv;
    }
    __syncthreads();

    if (gx >= WIDTH || gy >= HEIGHT) return;

    int sx = tx + 1;
    int sy = ty + 1;

    float4 px = s_px[sy][sx];
    float4 wv = s_wv[sy][sx];

    float4 px_l = s_px[sy][sx - 1];
    float4 px_r = s_px[sy][sx + 1];
    float4 px_t = s_px[sy - 1][sx];
    float4 px_b = s_px[sy + 1][sx];

    float4 wv_l = s_wv[sy][sx - 1];
    float4 wv_r = s_wv[sy][sx + 1];
    float4 wv_t = s_wv[sy - 1][sx];
    float4 wv_b = s_wv[sy + 1][sx];

    float4 px_prev = row_f4_const(px_prev_base, px_prev_pitch, gy)[gx];
    float4 wv_prev_raw = row_f4_const(wv_prev_base, wv_prev_pitch, gy)[gx];
    float4 wv_prev = make_float4(
        tanhf(wv_prev_raw.x * 0.5f) * 2.0f,
        tanhf(wv_prev_raw.y * 0.5f) * 2.0f,
        tanhf(wv_prev_raw.z * 0.5f) * 2.0f,
        tanhf(wv_prev_raw.w * 0.5f) * 2.0f);

    float c2 = row_f1_const(c2_base, c2_pitch, gy)[gx];
    c2 = fminf(0.499f, fmaxf(0.0f, c2));
    float damp = row_f1_const(damping_base, damping_pitch, gy)[gx];
    float drive_gain = row_f1_const(drive_base, drive_pitch, gy)[gx];
    float kwtp = row_f1_const(kwtp_base, kwtp_pitch, gy)[gx];

    float3 rgb = make_float3(clamp01(px.x), clamp01(px.y), clamp01(px.z));
    float alpha = clamp01(px.w);

    /* Colors ARE frequencies. R,G,B directly drive wave channels 0,1,2.
       Center each component (subtract mean) → bipolar drive so channels
       don't monotonically accumulate. Gray pixels → zero hue drive on
       ch0-2. Complementary colors exactly cancel: red(1,0,0) + cyan(0,1,1)
       both reduce to zero net drive vector when summed.
       As the fractal accumulates color the local mean (from color_gravity
       5×5 average) grows and pulls this cell's drive along with it. */
    float lum = (rgb.x + rgb.y + rgb.z) * 0.33333f;
    float4 drive = make_float4(
        (rgb.x - lum) * alpha * drive_gain,
        (rgb.y - lum) * alpha * drive_gain,
        (rgb.z - lum) * alpha * drive_gain,
        (lum - 0.5f)  * alpha * drive_gain   /* centered luminance on ch3 */
    );

    float4 wv_lap = make_float4(
        wv_l.x + wv_r.x + wv_t.x + wv_b.x - 4.0f * wv.x,
        wv_l.y + wv_r.y + wv_t.y + wv_b.y - 4.0f * wv.y,
        wv_l.z + wv_r.z + wv_t.z + wv_b.z - 4.0f * wv.z,
        wv_l.w + wv_r.w + wv_t.w + wv_b.w - 4.0f * wv.w
    );

    float4 wv_next = make_float4(
        damp * (2.0f * wv.x - wv_prev.x + c2 * wv_lap.x + drive.x),
        damp * (2.0f * wv.y - wv_prev.y + c2 * wv_lap.y + drive.y),
        damp * (2.0f * wv.z - wv_prev.z + c2 * wv_lap.z + drive.z),
        damp * (2.0f * wv.w - wv_prev.w + c2 * wv_lap.w + drive.w)
    );
    /* Clamp at write too — nothing rogue ever gets stored in the buffer */
    wv_next.x = tanhf(wv_next.x * 0.5f) * 2.0f;
    wv_next.y = tanhf(wv_next.y * 0.5f) * 2.0f;
    wv_next.z = tanhf(wv_next.z * 0.5f) * 2.0f;
    wv_next.w = tanhf(wv_next.w * 0.5f) * 2.0f;

    float4 px_lap = make_float4(
        px_l.x + px_r.x + px_t.x + px_b.x - 4.0f * px.x,
        px_l.y + px_r.y + px_t.y + px_b.y - 4.0f * px.y,
        px_l.z + px_r.z + px_t.z + px_b.z - 4.0f * px.z,
        px_l.w + px_r.w + px_t.w + px_b.w - 4.0f * px.w
    );

    /* Pixels are matter, not waves — they hold their state.
       Only chromatic gravity, triad force, and wave disruption move them.
       No px_prev inertia, no Laplacian propagation. */
    float4 px_inert = px;

    /* Wave crossing disruption: at every point where two wave arcs cross or
       touch, inject the magnetically repelled opponent color — the hue 180°
       away (true complement) from whatever color is sitting there.  That's
       the color the local cluster rejects most strongly via the complement
       magnetism rule, so it destroys the cluster structure and forces
       rearrangement.  Destructive interference drives entropy; constructive
       interference spreads wave hue outward.  A single wave passing through
       barely moves the Laplacian; two waves crossing spike it hard. */
    float lap_mag = sqrtf(fmaxf(0.0f,
        wv_lap.x * wv_lap.x + wv_lap.y * wv_lap.y +
        wv_lap.z * wv_lap.z + wv_lap.w * wv_lap.w));
    float cross_strength = clamp01(lap_mag * fmaxf(0.0f, kwtp));

    /* Determine constructive vs destructive interference:
       dot(wave, Laplacian) > 0 → Laplacian reinforces wave → constructive
       dot(wave, Laplacian) < 0 → Laplacian opposes wave  → destructive
       Normalize to get the cosine alignment in (-1, 1]. */
    float wv_sq  = wv.x*wv.x  + wv.y*wv.y  + wv.z*wv.z  + wv.w*wv.w;
    float lap_sq = wv_lap.x*wv_lap.x + wv_lap.y*wv_lap.y +
                   wv_lap.z*wv_lap.z + wv_lap.w*wv_lap.w;
    float cos_align = (wv_sq * lap_sq > 1.0e-16f)
        ? (wv.x*wv_lap.x + wv.y*wv_lap.y + wv.z*wv_lap.z + wv.w*wv_lap.w)
          / sqrtf(wv_sq * lap_sq)
        : 0.0f;
    /* constructive_frac ∈ [0,1]: 1=fully constructive, 0=fully destructive */
    float constructive_frac = (cos_align + 1.0f) * 0.5f;

    /* Wave-encoded hue disruption.
       Waves are driven by pixel RGB, so their chromatic channels carry the hue
       of wherever they originated.  Decode that hue and push the local pixel
       toward it (constructive) or toward the complement (destructive).
       We use the *current* wave value for hue direction and the Laplacian
       magnitude for disruption strength. */
    float wr = wv.x, wg = wv.y, wb = wv.z;
    float wlum = (wr + wg + wb) * 0.33333f;
    float wcr = wr - wlum, wcg = wg - wlum, wcb = wb - wlum;
    /* Project chromatic content onto the hue wheel via atan2 */
    float wave_h_rad = atan2f(wcg - wcb, wcr - (wcg + wcb) * 0.5f);
    if (wave_h_rad < 0.0f) wave_h_rad += 6.28318530f;
    float wave_hue01 = wave_h_rad * (1.0f / 6.28318530f);
    float wave_chroma = sqrtf(wcr*wcr + wcg*wcg + wcb*wcb);
    float3 wave_color = hue_to_rgb(wave_hue01);

    /* Destructive target: true complement (180°) — the color that disrupts
       and creates entropy, keeping the system from locking into attractors. */
    float my_hue01 = rgb_to_hue_rad(clamp01(px.x), clamp01(px.y), clamp01(px.z))
                     * (1.0f / 6.28318530f);
    float comp_hue01 = my_hue01 + 0.5f;   /* 180° = true complement */
    if (comp_hue01 >= 1.0f) comp_hue01 -= 1.0f;
    float3 comp = hue_to_rgb(comp_hue01);

    float chroma_blend = clamp01(wave_chroma * 4.0f); /* 0=complement, 1=wave-hue */

    /* Blend constructive target (wave hue) and destructive target (complement)
       based on alignment.  Constructive → spread wave hue (color diversity).
       Destructive → push toward complement (entropy/chaos). */
    float3 construct_target = make_float3(
        wave_color.x * chroma_blend + comp.x * (1.0f - chroma_blend),
        wave_color.y * chroma_blend + comp.y * (1.0f - chroma_blend),
        wave_color.z * chroma_blend + comp.z * (1.0f - chroma_blend)
    );
    float3 destruct_target = comp;  /* pure complement push */

    float3 target = make_float3(
        construct_target.x * constructive_frac + destruct_target.x * (1.0f - constructive_frac),
        construct_target.y * constructive_frac + destruct_target.y * (1.0f - constructive_frac),
        construct_target.z * constructive_frac + destruct_target.z * (1.0f - constructive_frac)
    );

    float4 px_next = make_float4(
        clamp01(px.x + (target.x - px.x) * cross_strength),
        clamp01(px.y + (target.y - px.y) * cross_strength),
        clamp01(px.z + (target.z - px.z) * cross_strength),
        clamp01(px.w)
    );

    float4 d = make_float4(px_next.x - px.x, px_next.y - px.y, px_next.z - px.z, px_next.w - px.w);
    float dm = sqrtf(d.x * d.x + d.y * d.y + d.z * d.z + d.w * d.w);

    /* Wave pays for the disruption it caused — small drain proportional to
       pixel delta so waves stay alive to keep driving the system. */
    float drain = 1.0f - dm * cross_strength * 0.08f;
    if (drain < 0.0f) drain = 0.0f;
    wv_next.x *= drain;
    wv_next.y *= drain;
    wv_next.z *= drain;
    wv_next.w *= drain;

    row_f4(px_next_base, px_next_pitch, gy)[gx] = px_next;
    row_f4(wv_next_base, wv_next_pitch, gy)[gx] = wv_next;
    row_f4(delta_next_base, delta_next_pitch, gy)[gx] = d;
    row_f1(delta_mag_base, delta_mag_pitch, gy)[gx] = dm;
}

__global__ void ring_write_kernel(const void* delta_base, size_t delta_pitch,
                                  const void* delta0_base, size_t delta0_pitch,
                                  float* ring, int slot) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;
    int idx = y * WIDTH + x;

    float4 d = row_f4_const(delta_base, delta_pitch, y)[x];
    float4 c = row_f4_const(delta0_base, delta0_pitch, y)[x];

    float* dst = ring + (((size_t)slot * CELLS + (size_t)idx) * 8);
    dst[0] = d.x; dst[1] = d.y; dst[2] = d.z; dst[3] = d.w;
    dst[4] = c.x; dst[5] = c.y; dst[6] = c.z; dst[7] = c.w;
}

/* Compute local hue coherence: how much does this pixel's hue match its
   3x3 neighborhood.  Returns [0,1]: 1=perfect match, 0=opposite hue.
   Also returns the weighted mean RGB of the neighborhood. */
__device__ float pixel_coherence(const void* px_base, size_t px_pitch,
                                  int x, int y, float3* out_avg) {
    float my_r = clamp01(row_f4_const(px_base, px_pitch, y)[x].x);
    float my_g = clamp01(row_f4_const(px_base, px_pitch, y)[x].y);
    float my_b = clamp01(row_f4_const(px_base, px_pitch, y)[x].z);
    float theta = rgb_to_hue_rad(my_r, my_g, my_b);

    float acc_r = 0.f, acc_g = 0.f, acc_b = 0.f;
    float coh_sum = 0.f;
    int count = 0;
    for (int oy = -1; oy <= 1; ++oy) {
        int yy = clampi(y + oy, 0, HEIGHT - 1);
        for (int ox = -1; ox <= 1; ++ox) {
            if (ox == 0 && oy == 0) continue;
            int xx = clampi(x + ox, 0, WIDTH - 1);
            float4 nb = row_f4_const(px_base, px_pitch, yy)[xx];
            float nr = clamp01(nb.x), ng = clamp01(nb.y), nbb = clamp01(nb.z);
            float phi = rgb_to_hue_rad(nr, ng, nbb);
            float dh = phi - theta;
            if (dh >  3.14159265f) dh -= 6.28318530f;
            if (dh < -3.14159265f) dh += 6.28318530f;
            float sim = cosf(dh * 0.5f); sim = sim * sim;
            coh_sum += sim;
            acc_r += nr; acc_g += ng; acc_b += nbb;
            count++;
        }
    }
    float inv = 1.0f / (float)count;
    out_avg->x = acc_r * inv;
    out_avg->y = acc_g * inv;
    out_avg->z = acc_b * inv;
    return coh_sum * inv; /* mean coherence [0,1] */
}

/* NCA IO builder — HSV-based features with direct 3×3 neighborhood reads.
   Old approach: called pixel_coherence() for each of 9 cells, each reading 8
   neighbors, totalling ~180 global reads and ~81 trig-heavy HSV conversions.
   New approach: reads 9 cells directly (18 total — curr and prev tick), computes
   HSV of each cell once.  Coherence is computed cheaply from the already-extracted
   hue angles.

   Input layout (NCA_IN = 136):
     [72]  9 cells × 8 features each:
           Center cell (flat index 4):  (sin_h, cos_h, sat, val, coh, coh_delta, sin_h0, cos_h0)
           Other 8 cells:               (sin_h, cos_h, sat, val, sin_h0, cos_h0, sat0, val0)
     [64]  History ring: HIST_TICKS × 8 floats per tick (unchanged)

   Target layout (NCA_OUT = 8):  one-tick-lag HSV state of this pixel:
           (sin_h, cos_h, sat, val, coh, coh_delta, sin_h0, cos_h0) */
__global__ void nca_build_io_kernel(const void* px_base,  size_t px_pitch,
                                    const void* px0_base, size_t px0_pitch,
                                    const float* ring, int ring_head,
                                    float* input, float* target, float* prev_target) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;

    int idx = y * WIDTH + x;
    float* in  = input       + (size_t)idx * NCA_IN;
    float* tgt = target      + (size_t)idx * NCA_OUT;
    float* prev_tgt = prev_target + (size_t)idx * NCA_OUT;

    /* Read 3×3 neighborhood for current and previous tick.
       Compute HSV once per cell — avoids the ~10× redundant reads of the old
       pixel_coherence-based approach. */
    float sh[9], ch[9], sa[9], va[9];     /* current tick sin/cos hue, sat, val */
    float sh0[9], ch0[9], sa0[9], va0[9]; /* previous tick */

    for (int oy = -1; oy <= 1; ++oy) {
        int yy = clampi(y + oy, 0, HEIGHT - 1);
        for (int ox = -1; ox <= 1; ++ox) {
            int xx = clampi(x + ox, 0, WIDTH  - 1);
            int flat = (oy + 1) * 3 + (ox + 1);

            float4 pc  = row_f4_const(px_base,  px_pitch,  yy)[xx];
            float4 pc0 = row_f4_const(px0_base, px0_pitch, yy)[xx];

            float hh, ss, vv;
            rgb_to_hsv_full(clamp01(pc.x), clamp01(pc.y), clamp01(pc.z), &hh, &ss, &vv);
            sh[flat] = sinf(hh); ch[flat] = cosf(hh);
            sa[flat] = ss; va[flat] = vv;

            float hh0, ss0, vv0;
            rgb_to_hsv_full(clamp01(pc0.x), clamp01(pc0.y), clamp01(pc0.z), &hh0, &ss0, &vv0);
            sh0[flat] = sinf(hh0); ch0[flat] = cosf(hh0);
            sa0[flat] = ss0; va0[flat] = vv0;
        }
    }

    /* Cheap coherence from pre-computed hue — no extra global reads.
       cos²(Δh/2): 1=same hue, 0=opposite.  Average over 8 neighbors. */
    const int ci = 4; /* center cell index in flat 3×3 array */
    float center_h  = atan2f(sh[ci],  ch[ci]);
    float center_h0 = atan2f(sh0[ci], ch0[ci]);
    float coh_sum = 0.f, coh_sum0 = 0.f;
    const float PI = 3.14159265f;
    for (int i = 0; i < 9; ++i) {
        if (i == ci) continue;
        float dh = atan2f(sh[i], ch[i]) - center_h;
        if (dh >  PI) dh -= 2.0f * PI;
        if (dh < -PI) dh += 2.0f * PI;
        float sim = cosf(dh * 0.5f); coh_sum += sim * sim;

        float dh0 = atan2f(sh0[i], ch0[i]) - center_h0;
        if (dh0 >  PI) dh0 -= 2.0f * PI;
        if (dh0 < -PI) dh0 += 2.0f * PI;
        float sim0 = cosf(dh0 * 0.5f); coh_sum0 += sim0 * sim0;
    }
    float coh  = coh_sum  * (1.0f / 8.0f);
    float coh0 = coh_sum0 * (1.0f / 8.0f);

    /* Build input: 8 features per cell in raster order (left-right, top-bottom).
       Center cell gets coherence features instead of duplicate sat0/val0 to give
       the NCA explicit access to the clustering signal at its location. */
    int t = 0;
    for (int flat = 0; flat < 9; ++flat) {
        if (flat == ci) {
            in[t++] = sh[flat];
            in[t++] = ch[flat];
            in[t++] = sa[flat];
            in[t++] = va[flat];
            in[t++] = coh;
            in[t++] = coh - coh0;
            in[t++] = sh0[flat];
            in[t++] = ch0[flat];
        } else {
            in[t++] = sh[flat];
            in[t++] = ch[flat];
            in[t++] = sa[flat];
            in[t++] = va[flat];
            in[t++] = sh0[flat];
            in[t++] = ch0[flat];
            in[t++] = sa0[flat];
            in[t++] = va0[flat];
        }
    }
    /* t == 72 here */

    /* History ring — unchanged, 8 floats per tick × HIST_TICKS ticks */
    for (int h = 0; h < HIST_TICKS; ++h) {
        int slot = (ring_head - h + HIST_TICKS) % HIST_TICKS;
        const float* src = ring + (((size_t)slot * CELLS + (size_t)idx) * 8);
        for (int c = 0; c < 8; ++c) in[t++] = src[c];
    }
    /* t == 136 == NCA_IN */

    /* One-tick-lag target: copy prev_target → target, store current state into
       prev_target so next invocation has this tick's physics state as its target.
       Target is the HSV state of THIS pixel (not neighborhood mean) so the NCA
       learns to predict per-pixel chromatic dynamics. */
    for (int c = 0; c < NCA_OUT; ++c) tgt[c] = prev_tgt[c];

    prev_tgt[0] = sh[ci];       /* sin(hue)        — periodic, no wrapping issues */
    prev_tgt[1] = ch[ci];       /* cos(hue)        — quadrature pair */
    prev_tgt[2] = sa[ci];       /* saturation      — vividity */
    prev_tgt[3] = va[ci];       /* value           — brightness */
    prev_tgt[4] = coh;          /* coherence       — clustering signal */
    prev_tgt[5] = coh - coh0;   /* coherence delta — direction of change */
    prev_tgt[6] = sh0[ci];      /* sin(prev_hue)   — hue trend */
    prev_tgt[7] = ch0[ci];      /* cos(prev_hue) */
}

/* Hue-frequency ripple tank driver.
   Each pixel's neighborhood has a dominant hue = a frequency on the color
   wheel.  Inject that frequency as a sin/cos pair into the wave so the
   ripple tank carries actual hue-frequency information.  Where same-hue
   pixels cluster (high coherence) they collectively reinforce a single
   wave frequency, causing resonant standing-wave patterns that form the
   fractal structures.  Pixels far from the neighborhood hue inject a
   smaller, less-coherent signal, naturally suppressing chaos in mixed
   regions and amplifying it in coherent clusters.

   Wave channel encoding:
     ch0 = sin(hue)        fundamental hue frequency (quadrature pair)
     ch1 = cos(hue)
     ch2 = sin(2·hue)      second harmonic — creates interference sub-bands
     ch3 = cos(2·hue)

   Hue similarity (cos²(Δh/2)) weights the injection so the wave from a
   cluster carries the cluster's own frequency, not a smeared average. */
__global__ void color_gravity_kernel(
    const void* px_base, size_t px_pitch,
    void* wv_base, size_t wv_pitch,
    int width, int height,
    float gravity_gain, int radius, float curve_power) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    /* Accumulate neighborhood RGB to find the dominant hue */
    float acc_r = 0.f, acc_g = 0.f, acc_b = 0.f;
    int count = 0;
    for (int oy = -radius; oy <= radius; ++oy) {
        int yy = clampi(y + oy, 0, height - 1);
        for (int ox = -radius; ox <= radius; ++ox) {
            int xx = clampi(x + ox, 0, width - 1);
            float4 p = row_f4_const(px_base, px_pitch, yy)[xx];
            acc_r += p.x; acc_g += p.y; acc_b += p.z;
            count++;
        }
    }

    float inv_count = 1.0f / fmaxf(1.0f, (float)count);
    float ar = clamp01(acc_r * inv_count);
    float ag = clamp01(acc_g * inv_count);
    float ab = clamp01(acc_b * inv_count);

    /* Compute neighborhood dominant hue */
    float avg_h, avg_s, avg_v;
    rgb_to_hsv_full(ar, ag, ab, &avg_h, &avg_s, &avg_v);

    /* Compute my own hue */
    float4 me = row_f4_const(px_base, px_pitch, y)[x];
    float my_h, my_s, my_v;
    rgb_to_hsv_full(clamp01(me.x), clamp01(me.y), clamp01(me.z), &my_h, &my_s, &my_v);

    /* Hue coherence: how closely does this pixel match the neighborhood hue?
       cos²(Δh/2): 1 at same hue, 0 at opposite hue.
       The injection is strongest where the pixel IS the neighborhood hue —
       coherent clusters inject a clean frequency signal. */
    float dh = avg_h - my_h;
    if (dh >  3.14159265f) dh -= 6.28318530f;
    if (dh < -3.14159265f) dh += 6.28318530f;
    float hue_coh = cosf(dh * 0.5f);
    hue_coh = hue_coh * hue_coh;

    float pull = hue_coh * fmaxf(0.0f, gravity_gain);

    /* Inject hue frequency: sin/cos of the neighborhood's dominant hue.
       Subtracting the pixel's own contribution prevents self-reinforcement
       (a pixel in a uniform cluster gets zero net injection; only pixels on
       cluster boundaries or in mixed regions see a signal). */
    float4 delta = make_float4(
        (sinf(avg_h) - sinf(my_h)) * pull,
        (cosf(avg_h) - cosf(my_h)) * pull,
        (sinf(avg_h * 2.0f) - sinf(my_h * 2.0f)) * pull * 0.5f,
        (cosf(avg_h * 2.0f) - cosf(my_h * 2.0f)) * pull * 0.5f
    );

    float4 wv = row_f4(wv_base, wv_pitch, y)[x];
    wv.x += delta.x;
    wv.y += delta.y;
    wv.z += delta.z;
    wv.w += delta.w;
    row_f4(wv_base, wv_pitch, y)[x] = wv;
}

/* Returns hue angle in [0, 2π).  Gray pixels return 0. */

/* Chromatic magnetic triad kernel.
   For every pixel, each neighbor exerts a force whose sign depends on where
   the neighbor sits on the hue wheel relative to this pixel:
     Δθ =  0        → 0      (same hue, neutral)
     Δθ = +120°     → +0.866 (triad partner ahead — attraction)
     Δθ = +240°     → −0.866 (triad partner behind — repulsion)
   The force function is sin(Δθ), smooth and periodic over the full circle.
   The force direction in RGB space is toward or away from the neighbor's color.
   Written additively into pixel_next (which already has physics-tick result). */

/* Pixel chromatic gravity — pulls pixels toward similar-hue neighbors by
   rotating the pixel's hue in HSV space.  Closer hue = stronger pull.
   Saturation and value are preserved (and saturation is gently restored if
   it drops below 0.75) so gravity clusters by hue without desaturating.
   This is the gravitational clustering force that builds the fractal structures. */
__global__ void pixel_gravity_kernel(
    const void* px_cur_base, size_t px_cur_pitch,
    void*       px_nxt_base, size_t px_nxt_pitch,
    int width, int height,
    float strength)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float4 me = row_f4_const(px_cur_base, px_cur_pitch, y)[x];
    float my_h, my_s, my_v;
    rgb_to_hsv_full(clamp01(me.x), clamp01(me.y), clamp01(me.z), &my_h, &my_s, &my_v);

    /* 8-neighbor for smoother fluid blending; diagonals weighted 0.707 */
    const int NX[8] = {-1, 1,  0, 0, -1,  1, -1,  1};
    const int NY[8] = { 0, 0, -1, 1, -1, -1,  1,  1};
    const float NW[8] = {1.f,1.f,1.f,1.f, 0.707f,0.707f,0.707f,0.707f};

    float hue_pull   = 0.0f;
    float weight_sum = 0.0f;
    float val_pull   = 0.0f;

    for (int i = 0; i < 8; ++i) {
        int nx = clampi(x + NX[i], 0, width  - 1);
        int ny = clampi(y + NY[i], 0, height - 1);
        float4 nb = row_f4_const(px_cur_base, px_cur_pitch, ny)[nx];
        float nb_h, nb_s, nb_v;
        rgb_to_hsv_full(clamp01(nb.x), clamp01(nb.y), clamp01(nb.z), &nb_h, &nb_s, &nb_v);

        float dh = nb_h - my_h;
        if (dh >  3.14159265f) dh -= 6.28318530f;
        if (dh < -3.14159265f) dh += 6.28318530f;

        /* Gravity: closer hue = stronger pull. cos²(dh/2): 1 at same, 0 at opposite. */
        float sim = cosf(dh * 0.5f);
        sim = sim * sim * NW[i];

        /* Accumulate hue rotation delta and weight */
        hue_pull   += dh * sim;
        weight_sum += sim;
        val_pull   += (nb_v - my_v) * sim;
    }

    if (weight_sum > 1.0e-6f) {
        float inv = strength / weight_sum;
        /* Rotate hue toward similar-hue neighbors */
        float new_h = my_h + hue_pull * inv;
        /* Keep saturation vivid — gravity must not desaturate */
        float new_s = my_s;
        if (new_s < GRAVITY_MIN_SATURATION) new_s = my_s + (GRAVITY_MIN_SATURATION - my_s) * 0.03f;
        /* Gentle value averaging keeps brightness coherent in clusters */
        float new_v = clamp01(my_v + val_pull * inv * 0.25f);

        float3 rgb = hsv_to_rgb_full(new_h, new_s, new_v);
        float4 out = row_f4(px_nxt_base, px_nxt_pitch, y)[x];
        out.x = clamp01(rgb.x);
        out.y = clamp01(rgb.y);
        out.z = clamp01(rgb.z);
        row_f4(px_nxt_base, px_nxt_pitch, y)[x] = out;
    }
}

/* Collide (destructive) interference kernel.
   When the wave changes sign at a pixel — indicating a wave collision /
   destructive cancellation — flip the local pixel and its complement to
   create entropic disruption.  This prevents the system from reaching stable
   attractors and keeps the field perpetually fluid.

   Detects zero crossings in the wave (sign of wave_curr ≠ sign of wave_prev)
   weighted by the amplitude of the crossing.  At each such event:
     • The subject pixel is pushed toward its HSV complement (hue + π).
     • Complement-zone pixels (globally) are handled by the XOR broadcast.
   This kernel applies only the LOCAL subject flip; the global complement
   response is picked up by the XOR broadcast on the same tick. */
__global__ void collide_interference_kernel(
    const void* wv_cur_base, size_t wv_cur_pitch,
    const void* wv_prev_base, size_t wv_prev_pitch,
    const void* px_cur_base, size_t px_cur_pitch,
    void*       px_nxt_base, size_t px_nxt_pitch,
    int width, int height,
    float strength)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float4 wv_cur  = row_f4_const(wv_cur_base,  wv_cur_pitch,  y)[x];
    float4 wv_prev = row_f4_const(wv_prev_base, wv_prev_pitch, y)[x];

    /* Detect sign change (zero crossing) in any wave channel.
       sign_prod < 0 means prev and curr had opposite signs → zero crossing.
       Magnitude contribution is sqrt(|prev * curr|) — proportional to the
       amplitudes at the crossing, favouring high-energy collisions. */
    float sign_prod_x = wv_prev.x * wv_cur.x;
    float sign_prod_y = wv_prev.y * wv_cur.y;
    float sign_prod_z = wv_prev.z * wv_cur.z;

    /* Collide event: sum sqrts of crossing magnitudes across channels */
    float collide_mag = sqrtf(
        fmaxf(0.0f, -sign_prod_x) +
        fmaxf(0.0f, -sign_prod_y) +
        fmaxf(0.0f, -sign_prod_z)
    ) * (1.0f / 1.7320508f); /* /sqrt(3) → normalized to [0,1] per channel */

    if (collide_mag < MIN_COLLIDE_MAGNITUDE) return; /* too weak — ignore */

    float4 me = row_f4_const(px_cur_base, px_cur_pitch, y)[x];
    float my_h, my_s, my_v;
    rgb_to_hsv_full(clamp01(me.x), clamp01(me.y), clamp01(me.z), &my_h, &my_s, &my_v);

    /* Flip toward HSV complement (hue + π = 180°) — true complement, not RGB */
    float comp_h = my_h + 3.14159265f;
    float flip_s = fmaxf(my_s, COLLIDE_MIN_SATURATION); /* collide boosts saturation for vivid effect */
    float3 comp_rgb = hsv_to_rgb_full(comp_h, flip_s, my_v);

    float w = fminf(1.0f, collide_mag * strength);

    float4 out = row_f4(px_nxt_base, px_nxt_pitch, y)[x];
    out.x = clamp01(clamp01(out.x) + (comp_rgb.x - clamp01(me.x)) * w);
    out.y = clamp01(clamp01(out.y) + (comp_rgb.y - clamp01(me.y)) * w);
    out.z = clamp01(clamp01(out.z) + (comp_rgb.z - clamp01(me.z)) * w);
    row_f4(px_nxt_base, px_nxt_pitch, y)[x] = out;
}


   Rule:
     exact complement (dh = π)         → ATTRACT (f = +1)
     one shade off complement (dh≈π±σ) → REPEL   (f < 0)
     same hue (dh ≈ 0)                 → neutral  (handled by gravity)
   This creates three emergent behaviors from one rule:
     - Pairing: opposite colors lock in stable pairs
     - Blocking: a color flanked by its hue-neighbors repels its complement
     - Exclusion: a locked pair expels the partners' neighboring shades
   Force is applied as a hue rotation in HSV space, preserving saturation.
   σ (sigma) controls the "one shade" width on the wheel — tunable as 'triad'. */
__global__ void complementary_neighbor_kernel(
    const void* px_cur_base, size_t px_cur_pitch,
    void*       px_nxt_base, size_t px_nxt_pitch,
    int width, int height,
    float strength)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float4 me = row_f4_const(px_cur_base, px_cur_pitch, y)[x];
    float my_h, my_s, my_v;
    rgb_to_hsv_full(clamp01(me.x), clamp01(me.y), clamp01(me.z), &my_h, &my_s, &my_v);

    const int NX[8] = {-1, 1,  0, 0, -1,  1, -1,  1};
    const int NY[8] = { 0, 0, -1, 1, -1, -1,  1,  1};
    const float NW[8] = {1.f,1.f,1.f,1.f, 0.5f,0.5f,0.5f,0.5f};

    /* σ = 0.35 rad ≈ 20° on the hue wheel — wider band than before so the
       "one shade off complement" repulsion zone is clearly visible.
       Repulsion zero-crosses at d = σ*sqrt(3) ≈ 0.61 rad ≈ 35°.
       Hues more than ~35° from the complement angle are left untouched,
       keeping most of the wheel free for continuous spectrum diversity. */
    const float sigma = 0.35f;
    const float sigma2 = sigma * sigma;
    const float PI = 3.14159265358979f;

    float hue_delta = 0.0f;
    float sat_delta = 0.0f;

    for (int i = 0; i < 8; ++i) {
        int nx = clampi(x + NX[i], 0, width  - 1);
        int ny = clampi(y + NY[i], 0, height - 1);
        float4 nb = row_f4_const(px_cur_base, px_cur_pitch, ny)[nx];
        float nb_h, nb_s, nb_v;
        rgb_to_hsv_full(clamp01(nb.x), clamp01(nb.y), clamp01(nb.z), &nb_h, &nb_s, &nb_v);

        /* Angular distance on the hue wheel, wrapped to (-π, π] */
        float dh = nb_h - my_h;
        if (dh >  PI) dh -= 2.0f * PI;
        if (dh < -PI) dh += 2.0f * PI;

        /* d = distance from the complement angle (π).
           d=0 → neighbor is my exact complement.
           d=π → neighbor is my same hue. */
        float d = PI - fabsf(dh);

        /* Mexican hat / Ricker wavelet centered at d=0 (complement):
             mex > 0 → attract (pull my hue toward being complement of neighbor)
             mex < 0 → repel  (push my hue away from being complement of neighbor) */
        float t2 = (d * d) / sigma2;
        float mex = (1.0f - t2) * expf(-t2 * 0.5f);

        /* Direction: to become the complement of this neighbor,
           I want my hue to be nb_h + π.
           hue_error = (nb_h + π) - my_h, wrapped to (-π, π] */
        float ideal = nb_h + PI;
        float hue_err = ideal - my_h;
        if (hue_err >  PI) hue_err -= 2.0f * PI;
        if (hue_err < -PI) hue_err += 2.0f * PI;

        hue_delta += hue_err * mex * NW[i];

        /* Saturation: attraction boosts saturation (vivid complementary pairs),
           repulsion slightly suppresses it (near-shades wash each other out). */
        sat_delta += mex * NW[i] * 0.15f;
    }

    /* Apply hue rotation and saturation shift in HSV, convert back to RGB */
    float new_h = my_h + hue_delta * strength;
    float new_s = clamp01(my_s + sat_delta * strength);
    /* Saturation floor: prevent graying out — keep colors vivid */
    if (new_s < COMPLEMENT_MIN_SATURATION) new_s = my_s + (COMPLEMENT_MIN_SATURATION - my_s) * 0.02f;

    float3 rgb = hsv_to_rgb_full(new_h, new_s, my_v);

    float4 out = row_f4(px_nxt_base, px_nxt_pitch, y)[x];
    out.x = clamp01(rgb.x);
    out.y = clamp01(rgb.y);
    out.z = clamp01(rgb.z);
    row_f4(px_nxt_base, px_nxt_pitch, y)[x] = out;
}

__global__ void nca_forward_kernel(const float* input, const float* w1, const float* b1,
                                   const float* w2, const float* b2,
                                   float* hidden, float* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int)CELLS) return;

    const float* in = input + (size_t)idx * NCA_IN;
    float* h = hidden + (size_t)idx * NCA_H;
    float* o = out + (size_t)idx * NCA_OUT;

    for (int j = 0; j < NCA_H; ++j) {
        float acc = b1[j];
        for (int i = 0; i < NCA_IN; ++i) {
            float v = in[i];
            if (isnan(v) || isinf(v)) v = 0.0f;
            acc += v * w1[i * NCA_H + j];
        }
        acc = isnan(acc) ? 0.0f : acc;
        h[j] = (acc > 0.0f) ? fminf(acc, 1.0e6f) : 0.0f;
    }

    for (int k = 0; k < NCA_OUT; ++k) {
        float acc = b2[k];
        for (int j = 0; j < NCA_H; ++j) acc += h[j] * w2[j * NCA_OUT + k];
        o[k] = isnan(acc) ? 0.0f : fmaxf(fminf(acc, 1.0e6f), -1.0e6f);
    }
}

__global__ void nca_backward_feedback_kernel(const float* input, const float* hidden,
                                             const float* out, const float* target,
                                             const float* w2,
                                             float* gw1, float* gb1,
                                             float* gw2, float* gb2,
                                             float* residual,
                                             void* wave_curr_base, size_t wave_pitch,
                                             const void* fb_base, size_t fb_pitch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;
    int idx = y * WIDTH + x;

    const float* in = input + (size_t)idx * NCA_IN;
    const float* h = hidden + (size_t)idx * NCA_H;
    const float* o = out + (size_t)idx * NCA_OUT;
    const float* t = target + (size_t)idx * NCA_OUT;
    float* r = residual + (size_t)idx * NCA_OUT;

    float dout[NCA_OUT];
    for (int k = 0; k < NCA_OUT; ++k) {
        float e = o[k] - t[k];
        r[k] = e;
        dout[k] = e;
        atomicAdd(&gb2[k], e);
    }

    float dh[NCA_H];
    for (int j = 0; j < NCA_H; ++j) {
        float acc = 0.0f;
        for (int k = 0; k < NCA_OUT; ++k) {
            atomicAdd(&gw2[j * NCA_OUT + k], h[j] * dout[k]);
            acc += dout[k] * w2[j * NCA_OUT + k];
        }
        dh[j] = (h[j] > 0.0f) ? acc : 0.0f;
        atomicAdd(&gb1[j], dh[j]);
    }

    for (int i = 0; i < NCA_IN; ++i) {
        float xi = in[i];
        for (int j = 0; j < NCA_H; ++j) {
            atomicAdd(&gw1[i * NCA_H + j], xi * dh[j]);
        }
    }

    float fb = row_f1_const(fb_base, fb_pitch, y)[x];
    float4 wv = row_f4(wave_curr_base, wave_pitch, y)[x];
    /* Clamp injection so NCA feedback can't drive rogue waves */
    #define FB_CLAMP 0.1f
    wv.x += fmaxf(-FB_CLAMP, fminf(FB_CLAMP, r[0] * fb));
    wv.y += fmaxf(-FB_CLAMP, fminf(FB_CLAMP, r[1] * fb));
    wv.z += fmaxf(-FB_CLAMP, fminf(FB_CLAMP, r[2] * fb));
    wv.w += fmaxf(-FB_CLAMP, fminf(FB_CLAMP, r[3] * fb));
    row_f4(wave_curr_base, wave_pitch, y)[x] = wv;
}

/* Compute sum of squared gradients into d_gnorm (must be zeroed before call) */
__global__ void grad_norm_kernel(const float* g, int n, float inv_n, float* d_gnorm) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = g[i] * inv_n;
    atomicAdd(d_gnorm, v * v);
}

/* Adam optimizer with gradient clipping.
   m[i] and v[i] are the first and second moment estimates; they persist across
   steps.  bc1 = 1/(1-beta1^t) and bc2 = 1/(1-beta2^t) are bias-correction
   factors computed on the host.  Clears g[i] after applying the update so the
   gradient accumulation buffer is ready for the next window. */
__global__ void adam_step_kernel(float* w, float* g, float* m, float* v, int n,
                                  float lr, float inv_n, float clip_scale,
                                  float beta1, float beta2, float eps,
                                  float bc1, float bc2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float grad = g[i] * inv_n * clip_scale;
    m[i] = beta1 * m[i] + (1.0f - beta1) * grad;
    v[i] = beta2 * v[i] + (1.0f - beta2) * grad * grad;
    float m_hat = m[i] * bc1;
    float v_hat = v[i] * bc2;
    w[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    g[i] = 0.0f;
}

__global__ void comb_update_kernel(const void* delta_mag_base, size_t delta_mag_pitch,
                                   const void* wave_base, size_t wave_pitch,
                                   uint32_t* act_count, float* ewma, float* hist) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;

    int idx = y * WIDTH + x;
    float dm = row_f1_const(delta_mag_base, delta_mag_pitch, y)[x];

    if (dm > 0.01f) atomicAdd(&act_count[idx], 1u);

    float a = 1.0f / 10000.0f;
    float old = ewma[idx];
    ewma[idx] = old + a * (dm - old);

    int xl = clampi(x - 1, 0, WIDTH - 1);
    int xr = clampi(x + 1, 0, WIDTH - 1);
    int yu = clampi(y - 1, 0, HEIGHT - 1);
    int yd = clampi(y + 1, 0, HEIGHT - 1);

    float4 wl = row_f4_const(wave_base, wave_pitch, y)[xl];
    float4 wr = row_f4_const(wave_base, wave_pitch, y)[xr];
    float4 wu = row_f4_const(wave_base, wave_pitch, yu)[x];
    float4 wd = row_f4_const(wave_base, wave_pitch, yd)[x];

    float gx = wr.x - wl.x;
    float gy = wd.x - wu.x;
    float ang = atan2f(gy, gx);
    float n = (ang + 3.14159265359f) * (8.0f / (2.0f * 3.14159265359f));
    int bin = ((int)floorf(n)) & 7;
    atomicAdd(&hist[(size_t)idx * 8 + bin], 1.0f);
}

__global__ void stats_kernel(const void* delta_mag_base, size_t delta_mag_pitch,
                             const void* wave_base, size_t wave_pitch,
                             unsigned long long* active,
                             float* wave_energy,
                             float* delta_energy) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;

    float dm = row_f1_const(delta_mag_base, delta_mag_pitch, y)[x];
    if (dm > 0.0001f) atomicAdd(active, 1ull);

    float4 wv = row_f4_const(wave_base, wave_pitch, y)[x];
    float we = wv.x*wv.x + wv.y*wv.y + wv.z*wv.z + wv.w*wv.w;
    float de = dm * dm;
    atomicAdd(wave_energy, we);
    atomicAdd(delta_energy, de);
}

/* Measures NCA learning quality per cell:
   loss     = mean squared error between output and target
   residual = mean magnitude of correction injected into wave */
__global__ void nca_learn_stats_kernel(const float* output, const float* target,
                                       const float* residual,
                                       float* d_loss, float* d_residual_mag) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int)CELLS) return;

    const float* o = output   + (size_t)idx * NCA_OUT;
    const float* t = target   + (size_t)idx * NCA_OUT;
    const float* r = residual + (size_t)idx * NCA_OUT;

    float mse = 0.f, rmag = 0.f;
    for (int k = 0; k < NCA_OUT; ++k) {
        float diff = o[k] - t[k];
        mse  += diff * diff;
        rmag += r[k] * r[k];
    }
    atomicAdd(d_loss,         mse  / NCA_OUT);
    atomicAdd(d_residual_mag, rmag / NCA_OUT);
}

__global__ void entangle_sync_kernel(const EntEdge* edges, int n_edges,
                                     const DevSrcView* src_views,
                                     const DevDstView* dst_views) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_edges) return;

    EntEdge e = edges[i];
    if (e.src_sub >= e.dst_sub) return;

    DevSrcView sv = src_views[e.src_sub];
    DevDstView dv = dst_views[e.dst_sub];

    int sx = (int)e.src_x;
    int sy = (int)e.src_y;
    int dx = (int)e.dst_x;
    int dy = (int)e.dst_y;
    if ((unsigned)sx >= WIDTH || (unsigned)dx >= WIDTH || (unsigned)sy >= HEIGHT || (unsigned)dy >= HEIGHT) return;

    float s = fminf(1.0f, fmaxf(0.0f, e.strength));

    float4 src_px = row_f4_const(sv.px_next, sv.px_pitch, sy)[sx];
    float4 src_wv = row_f4_const(sv.wv_next, sv.wv_pitch, sy)[sx];

    float4* dst_px_ptr = &row_f4(dv.px_next, dv.px_pitch, dy)[dx];
    float4* dst_wv_ptr = &row_f4(dv.wv_next, dv.wv_pitch, dy)[dx];

    float4 dst_px = *dst_px_ptr;
    float4 dst_wv = *dst_wv_ptr;

    dst_px.x = fmaf(src_px.x - dst_px.x, s, dst_px.x);
    dst_px.y = fmaf(src_px.y - dst_px.y, s, dst_px.y);
    dst_px.z = fmaf(src_px.z - dst_px.z, s, dst_px.z);
    dst_px.w = fmaf(src_px.w - dst_px.w, s, dst_px.w);

    dst_wv.x = fmaf(src_wv.x - dst_wv.x, s, dst_wv.x);
    dst_wv.y = fmaf(src_wv.y - dst_wv.y, s, dst_wv.y);
    dst_wv.z = fmaf(src_wv.z - dst_wv.z, s, dst_wv.z);
    dst_wv.w = fmaf(src_wv.w - dst_wv.w, s, dst_wv.w);

    *dst_px_ptr = dst_px;
    *dst_wv_ptr = dst_wv;
}

__global__ void candidate_pairs_kernel(const uint32_t* act, const float* ewma,
                                       PairCandidate* out, uint32_t* out_count,
                                       uint32_t max_out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;

    int idx = y * WIDTH + x;
    if (act[idx] < 16u) return;

    float a = ewma[idx];
    if (a <= 1.0e-6f) return;

    /* NOTE: O(cells * 17 * 17). Kept intentionally simple for correctness. */
    for (int oy = -8; oy <= 8; ++oy) {
        int yy = y + oy;
        if ((unsigned)yy >= HEIGHT) continue;
        for (int ox = -8; ox <= 8; ++ox) {
            if (ox == 0 && oy == 0) continue;
            int xx = x + ox;
            if ((unsigned)xx >= WIDTH) continue;
            int j = yy * WIDTH + xx;
            if (j <= idx) continue;
            if (act[j] < 16u) continue;

            float b = ewma[j];
            float mn = fminf(a, b);
            float mx = fmaxf(a, b);
            float corr = (mx > 1.0e-6f) ? (mn / mx) : 0.0f;
            if (corr > 0.8f) {
                uint32_t pos = atomicAdd(out_count, 1u);
                if (pos < max_out) {
                    out[pos].a = (uint32_t)idx;
                    out[pos].b = (uint32_t)j;
                    out[pos].score = corr;
                }
            }
        }
    }
}

__global__ void force_pair_average_kernel(void* px_base, size_t px_pitch,
                                          void* wv_base, size_t wv_pitch,
                                          uint32_t a, uint32_t b) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    int ax = (int)(a % WIDTH), ay = (int)(a / WIDTH);
    int bx = (int)(b % WIDTH), by = (int)(b / WIDTH);

    float4* apr = &row_f4(px_base, px_pitch, ay)[ax];
    float4* bpr = &row_f4(px_base, px_pitch, by)[bx];
    float4* awr = &row_f4(wv_base, wv_pitch, ay)[ax];
    float4* bwr = &row_f4(wv_base, wv_pitch, by)[bx];

    float4 ap = *apr, bp = *bpr;
    float4 aw = *awr, bw = *bwr;

    float4 mp = make_float4(0.5f * (ap.x + bp.x), 0.5f * (ap.y + bp.y), 0.5f * (ap.z + bp.z), 0.5f * (ap.w + bp.w));
    float4 mw = make_float4(0.5f * (aw.x + bw.x), 0.5f * (aw.y + bw.y), 0.5f * (aw.z + bw.z), 0.5f * (aw.w + bw.w));

    *apr = mp; *bpr = mp;
    *awr = mw; *bwr = mw;
}

__global__ void pair_corr_kernel(const void* delta_base, size_t delta_pitch,
                                 uint32_t a, uint32_t b,
                                 float* out_corr) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    int ax = (int)(a % WIDTH), ay = (int)(a / WIDTH);
    int bx = (int)(b % WIDTH), by = (int)(b / WIDTH);

    float4 da = row_f4_const(delta_base, delta_pitch, ay)[ax];
    float4 db = row_f4_const(delta_base, delta_pitch, by)[bx];

    float dot = da.x * db.x + da.y * db.y + da.z * db.z + da.w * db.w;
    float na = da.x * da.x + da.y * da.y + da.z * da.z + da.w * da.w;
    float nb = db.x * db.x + db.y * db.y + db.z * db.z + db.w * db.w;
    float c = dot / (sqrtf(fmaxf(na * nb, 1.0e-20f)));
    *out_corr = c;
}

/* ---------------- Minimal PNG writer (RGBA8) ---------------- */

static uint32_t crc32_table[256];
static int crc32_ready = 0;

static void crc32_init(void) {
    if (crc32_ready) return;
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t c = i;
        for (int k = 0; k < 8; ++k) {
            c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        }
        crc32_table[i] = c;
    }
    crc32_ready = 1;
}

static uint32_t crc32_calc(const unsigned char* data, size_t n) {
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) {
        c = crc32_table[(c ^ data[i]) & 0xFFu] ^ (c >> 8);
    }
    return c ^ 0xFFFFFFFFu;
}

static uint32_t adler32_calc(const unsigned char* data, size_t n) {
    uint32_t a = 1u, b = 0u;
    for (size_t i = 0; i < n; ++i) {
        a = (a + data[i]) % 65521u;
        b = (b + a) % 65521u;
    }
    return (b << 16) | a;
}

static void put_u32_be(unsigned char* p, uint32_t v) {
    p[0] = (unsigned char)((v >> 24) & 0xFFu);
    p[1] = (unsigned char)((v >> 16) & 0xFFu);
    p[2] = (unsigned char)((v >> 8) & 0xFFu);
    p[3] = (unsigned char)(v & 0xFFu);
}

static int write_png_rgba8(const char* path, const unsigned char* rgba, int w, int h) {
    crc32_init();

    size_t raw_stride = (size_t)w * 4u + 1u;
    size_t raw_size = raw_stride * (size_t)h;
    unsigned char* raw = (unsigned char*)malloc(raw_size);
    if (!raw) return -1;

    for (int y = 0; y < h; ++y) {
        unsigned char* row = raw + (size_t)y * raw_stride;
        row[0] = 0;
        memcpy(row + 1, rgba + (size_t)y * (size_t)w * 4u, (size_t)w * 4u);
    }

    size_t max_blocks = (raw_size + 65534u) / 65535u;
    size_t zsize = 2u + raw_size + max_blocks * 5u + 4u;
    unsigned char* z = (unsigned char*)malloc(zsize);
    if (!z) { free(raw); return -1; }

    size_t zp = 0;
    z[zp++] = 0x78;
    z[zp++] = 0x01;

    size_t rem = raw_size;
    size_t off = 0;
    while (rem > 0) {
        uint16_t len = (uint16_t)((rem > 65535u) ? 65535u : rem);
        uint16_t nlen = (uint16_t)~len;
        unsigned char bfinal = (rem <= 65535u) ? 1u : 0u;
        z[zp++] = bfinal;
        z[zp++] = (unsigned char)(len & 0xFFu);
        z[zp++] = (unsigned char)((len >> 8) & 0xFFu);
        z[zp++] = (unsigned char)(nlen & 0xFFu);
        z[zp++] = (unsigned char)((nlen >> 8) & 0xFFu);
        memcpy(z + zp, raw + off, len);
        zp += len;
        off += len;
        rem -= len;
    }

    uint32_t ad = adler32_calc(raw, raw_size);
    z[zp++] = (unsigned char)((ad >> 24) & 0xFFu);
    z[zp++] = (unsigned char)((ad >> 16) & 0xFFu);
    z[zp++] = (unsigned char)((ad >> 8) & 0xFFu);
    z[zp++] = (unsigned char)(ad & 0xFFu);

    FILE* f = fopen(path, "wb");
    if (!f) { free(z); free(raw); return -1; }

    static const unsigned char sig[8] = {137,80,78,71,13,10,26,10};
    fwrite(sig, 1, 8, f);

    unsigned char ihdr[13];
    put_u32_be(&ihdr[0], (uint32_t)w);
    put_u32_be(&ihdr[4], (uint32_t)h);
    ihdr[8] = 8;
    ihdr[9] = 6;
    ihdr[10] = 0;
    ihdr[11] = 0;
    ihdr[12] = 0;

    unsigned char lenb[4], ctype[4], crcb[4];

    put_u32_be(lenb, 13u); fwrite(lenb, 1, 4, f);
    memcpy(ctype, "IHDR", 4); fwrite(ctype, 1, 4, f);
    fwrite(ihdr, 1, 13, f);
    unsigned char ihdr_crc_buf[17];
    memcpy(ihdr_crc_buf, ctype, 4);
    memcpy(ihdr_crc_buf + 4, ihdr, 13);
    put_u32_be(crcb, crc32_calc(ihdr_crc_buf, 17)); fwrite(crcb, 1, 4, f);

    put_u32_be(lenb, (uint32_t)zp); fwrite(lenb, 1, 4, f);
    memcpy(ctype, "IDAT", 4); fwrite(ctype, 1, 4, f);
    fwrite(z, 1, zp, f);
    unsigned char* idat_crc_buf = (unsigned char*)malloc(zp + 4);
    memcpy(idat_crc_buf, ctype, 4);
    memcpy(idat_crc_buf + 4, z, zp);
    put_u32_be(crcb, crc32_calc(idat_crc_buf, zp + 4)); fwrite(crcb, 1, 4, f);
    free(idat_crc_buf);

    put_u32_be(lenb, 0u); fwrite(lenb, 1, 4, f);
    memcpy(ctype, "IEND", 4); fwrite(ctype, 1, 4, f);
    put_u32_be(crcb, crc32_calc(ctype, 4)); fwrite(crcb, 1, 4, f);

    fclose(f);
    free(z);
    free(raw);
    return 0;
}

/* ---------------- Host-side substrate/pool helpers ---------------- */

static void zero_plane_struct(Plane* p) { memset(p, 0, sizeof(*p)); }
static void free_plane(Plane* p); /* forward decl */

static int alloc_plane(Plane* p, int with_nca) {
    zero_plane_struct(p);

    cudaError_t e;

#define ALLOC_OR_FAIL(expr) do { e = (expr); if (e != cudaSuccess) { free_plane(p); return -1; } } while(0)

    ALLOC_OR_FAIL(cudaMallocPitch(&p->pixel_prev, &p->pixel_pitch, WIDTH * sizeof(float4), HEIGHT));
    ALLOC_OR_FAIL(cudaMallocPitch(&p->pixel_curr, &p->pixel_pitch, WIDTH * sizeof(float4), HEIGHT));
    ALLOC_OR_FAIL(cudaMallocPitch(&p->pixel_next, &p->pixel_pitch, WIDTH * sizeof(float4), HEIGHT));

    ALLOC_OR_FAIL(cudaMallocPitch(&p->wave_prev, &p->wave_pitch, WIDTH * sizeof(float4), HEIGHT));
    ALLOC_OR_FAIL(cudaMallocPitch(&p->wave_curr, &p->wave_pitch, WIDTH * sizeof(float4), HEIGHT));
    ALLOC_OR_FAIL(cudaMallocPitch(&p->wave_next, &p->wave_pitch, WIDTH * sizeof(float4), HEIGHT));

    ALLOC_OR_FAIL(cudaMallocPitch(&p->delta_curr, &p->delta_pitch, WIDTH * sizeof(float4), HEIGHT));
    ALLOC_OR_FAIL(cudaMallocPitch(&p->delta_next, &p->delta_pitch, WIDTH * sizeof(float4), HEIGHT));
    ALLOC_OR_FAIL(cudaMallocPitch(&p->delta_mag, &p->scalar_pitch, WIDTH * sizeof(float), HEIGHT));

    ALLOC_OR_FAIL(cudaMallocPitch(&p->c2, &p->scalar_pitch, WIDTH * sizeof(float), HEIGHT));
    ALLOC_OR_FAIL(cudaMallocPitch(&p->damping, &p->scalar_pitch, WIDTH * sizeof(float), HEIGHT));
    ALLOC_OR_FAIL(cudaMallocPitch(&p->drive_gain, &p->scalar_pitch, WIDTH * sizeof(float), HEIGHT));
    ALLOC_OR_FAIL(cudaMallocPitch(&p->k_wave_to_px, &p->scalar_pitch, WIDTH * sizeof(float), HEIGHT));
    ALLOC_OR_FAIL(cudaMallocPitch(&p->feedback_gain, &p->scalar_pitch, WIDTH * sizeof(float), HEIGHT));

    ALLOC_OR_FAIL(cudaMalloc(&p->act_count, CELLS * sizeof(uint32_t)));
    ALLOC_OR_FAIL(cudaMalloc(&p->ewma_delta, CELLS * sizeof(float)));
    ALLOC_OR_FAIL(cudaMalloc(&p->dir_hist, CELLS * 8 * sizeof(float)));

    if (with_nca) {
        ALLOC_OR_FAIL(cudaMalloc(&p->ring, (size_t)HIST_TICKS * CELLS * 8 * sizeof(float)));
        ALLOC_OR_FAIL(cudaMalloc(&p->input,      CELLS * NCA_IN  * sizeof(float)));
        ALLOC_OR_FAIL(cudaMalloc(&p->hidden,     CELLS * NCA_H   * sizeof(float)));
        ALLOC_OR_FAIL(cudaMalloc(&p->output,     CELLS * NCA_OUT * sizeof(float)));
        ALLOC_OR_FAIL(cudaMalloc(&p->target,     CELLS * NCA_OUT * sizeof(float)));
        ALLOC_OR_FAIL(cudaMalloc(&p->prev_target,CELLS * NCA_OUT * sizeof(float)));
        ALLOC_OR_FAIL(cudaMalloc(&p->residual,   CELLS * NCA_OUT * sizeof(float)));
    }

    p->ring_head = 0;

    CHECK_CUDA(cudaMemset(p->act_count,  0, CELLS * sizeof(uint32_t)));
    CHECK_CUDA(cudaMemset(p->ewma_delta, 0, CELLS * sizeof(float)));
    CHECK_CUDA(cudaMemset(p->dir_hist,   0, CELLS * 8 * sizeof(float)));
    if (with_nca) {
        CHECK_CUDA(cudaMemset(p->ring,       0, (size_t)HIST_TICKS * CELLS * 8 * sizeof(float)));
        CHECK_CUDA(cudaMemset(p->hidden,     0, CELLS * NCA_H   * sizeof(float)));
        CHECK_CUDA(cudaMemset(p->output,     0, CELLS * NCA_OUT * sizeof(float)));
        CHECK_CUDA(cudaMemset(p->target,     0, CELLS * NCA_OUT * sizeof(float)));
        CHECK_CUDA(cudaMemset(p->prev_target,0, CELLS * NCA_OUT * sizeof(float)));
        CHECK_CUDA(cudaMemset(p->residual,   0, CELLS * NCA_OUT * sizeof(float)));
    }

#undef ALLOC_OR_FAIL

    dim3 b(BX, BY);
    dim3 g((WIDTH + BX - 1) / BX, (HEIGHT + BY - 1) / BY);
    init_params_kernel<<<g, b>>>(p->c2, p->scalar_pitch,
                                 p->damping, p->scalar_pitch,
                                 p->drive_gain, p->scalar_pitch,
                                 p->k_wave_to_px, p->scalar_pitch,
                                 p->feedback_gain, p->scalar_pitch,
                                 BASE_C2, BASE_DAMPING, BASE_DRIVE, BASE_KWTP, BASE_FEEDBACK);
    CHECK_CUDA(cudaGetLastError());
    return 0;
}

static void free_plane(Plane* p) {
    cudaFree(p->pixel_prev); cudaFree(p->pixel_curr); cudaFree(p->pixel_next);
    cudaFree(p->wave_prev); cudaFree(p->wave_curr); cudaFree(p->wave_next);
    cudaFree(p->delta_curr); cudaFree(p->delta_next); cudaFree(p->delta_mag);
    cudaFree(p->c2); cudaFree(p->damping); cudaFree(p->drive_gain); cudaFree(p->k_wave_to_px); cudaFree(p->feedback_gain);
    cudaFree(p->act_count); cudaFree(p->ewma_delta); cudaFree(p->dir_hist);
    cudaFree(p->ring); cudaFree(p->input); cudaFree(p->hidden); cudaFree(p->output); cudaFree(p->target); cudaFree(p->prev_target); cudaFree(p->residual);
    zero_plane_struct(p);
}

static void rotate_plane(Plane* p) {
    void* tmp;
    tmp = p->pixel_prev; p->pixel_prev = p->pixel_curr; p->pixel_curr = p->pixel_next; p->pixel_next = tmp;
    tmp = p->wave_prev; p->wave_prev = p->wave_curr; p->wave_curr = p->wave_next; p->wave_next = tmp;
    tmp = p->delta_curr; p->delta_curr = p->delta_next; p->delta_next = tmp;
}

static int pool_add_substrate(SubstratePool* pool, int id, int write_locked, int temporary, int parent, int seed_mode, uint32_t seed) {
    if (id < 0 || id >= MAX_SUBSTRATES) return -1;
    if (pool->s[id].active) return id;

    if (alloc_plane(&pool->s[id].p, !temporary) != 0) {
        return -1;
    }

    pool->s[id].id = id;
    pool->s[id].active = 1;
    pool->s[id].write_locked = write_locked;
    pool->s[id].temporary = temporary;
    pool->s[id].parent = parent;

    dim3 b(BX, BY);
    dim3 g((WIDTH + BX - 1) / BX, (HEIGHT + BY - 1) / BY);
    seed_kernel<<<g, b>>>(pool->s[id].p.pixel_curr, pool->s[id].p.pixel_pitch,
                          pool->s[id].p.wave_curr, pool->s[id].p.wave_pitch,
                          seed_mode, seed);
    seed_kernel<<<g, b>>>(pool->s[id].p.pixel_prev, pool->s[id].p.pixel_pitch,
                          pool->s[id].p.wave_prev, pool->s[id].p.wave_pitch,
                          seed_mode, seed);
    seed_kernel<<<g, b>>>(pool->s[id].p.pixel_next, pool->s[id].p.pixel_pitch,
                          pool->s[id].p.wave_next, pool->s[id].p.wave_pitch,
                          seed_mode, seed);
    CHECK_CUDA(cudaGetLastError());

    pool->active_count++;
    return id;
}

static void pool_remove_substrate(SubstratePool* pool, int id) {
    if (id < 0 || id >= MAX_SUBSTRATES) return;
    if (!pool->s[id].active) return;
    free_plane(&pool->s[id].p);
    memset(&pool->s[id], 0, sizeof(pool->s[id]));
    if (pool->active_count > 0) pool->active_count--;
}

static void apply_tunables_to_pool(SubstratePool* pool, const LiveTunables* t, cudaStream_t stream) {
    dim3 b(BX, BY);
    dim3 g((WIDTH + BX - 1) / BX, (HEIGHT + BY - 1) / BY);
    for (int sid = 0; sid < MAX_SUBSTRATES; ++sid) {
        if (!pool->s[sid].active) continue;
        Plane* p = &pool->s[sid].p;
        init_params_kernel<<<g, b, 0, stream>>>(p->c2, p->scalar_pitch,
                                                p->damping, p->scalar_pitch,
                                                p->drive_gain, p->scalar_pitch,
                                                p->k_wave_to_px, p->scalar_pitch,
                                                p->feedback_gain, p->scalar_pitch,
                                                t->c2, t->damping, t->drive, t->kwtp, t->feedback);
    }
    CHECK_CUDA(cudaGetLastError());
}

static void show_help(void) {
    fputs(ANSI_CLEAR, stdout);
    fprintf(stdout, ANSI_BOLD ANSI_CYAN
        "\n  ╔══════════════════════════════════════════════════════════════╗\n"
        "  ║               LIVE TUNING REFERENCE                         ║\n"
        "  ╚══════════════════════════════════════════════════════════════╝\n"
        ANSI_RESET "\n");
    fprintf(stdout,
        "  " ANSI_BOLD "Key   Parameter          Step         Range" ANSI_RESET "\n"
        "  ─────────────────────────────────────────────────────────\n"
        "  q/a   c2 (wave speed²)   ±0.005       0.01 – 0.49\n"
        "        Controls how fast waves propagate. Higher = faster\n"
        "        spreading, more energetic interference. CFL limit 0.5.\n\n"
        "  w/s   damping             ±0.0003      0.94 – 0.9999\n"
        "        Energy loss per tick. Lower = faster decay, higher =\n"
        "        longer ringing. Near 1.0 = almost lossless.\n\n"
        "  e/d   drive (color→wave)  ±0.0000005   0.0 – 0.00005\n"
        "        How strongly pixel RGB injects into wave channels.\n"
        "        Higher = louder color-driven oscillation.\n\n"
        "  r/f   kwtp (wave→pixel)   ±0.001       0.0 – 0.05\n"
        "        How strongly dominant wave channel biases pixel hue.\n"
        "        The bidirectional coupling with drive creates feedback.\n\n"
        "  t/g   feedback (NCA→wave) ±0.00001     0.0 – 0.001\n"
        "        NCA prediction residual injected back into wave field.\n"
        "        Higher = stronger self-correction loop.\n\n"
        "  y/h   gravity (color pull) ±0.00001    0.0 – 0.001\n"
        "        5×5 neighborhood color averaging strength.\n"
        "        Drives fractal growth — higher = faster aggregation.\n\n"
        "  u/j   lr (NCA learn rate) ±0.00002     0.000001 – 0.005\n"
        "        SGD learning rate for the NCA predictor.\n\n"
        "  " ANSI_BOLD "space" ANSI_RESET "  pause / resume\n"
        "  " ANSI_BOLD "n" ANSI_RESET "      single-step (while paused)\n"
        "  " ANSI_BOLD "x" ANSI_RESET "      quit\n"
        "  " ANSI_BOLD "?" ANSI_RESET "      this help screen\n\n"
        "  " ANSI_DIM "Press any key to return..." ANSI_RESET "\n");
    fflush(stdout);
    /* Block until a key is pressed */
    while (poll_key_nonblock() < 0) cli_sleep_ms(16);
}

static void process_live_controls(LiveTunables* t, int* params_dirty, int* request_quit) {
    for (;;) {
        int ch = poll_key_nonblock();
        if (ch < 0) break;

        if (ch == 0 || ch == 224) {
            int ext = poll_key_nonblock();
            if (ext < 0) break;
            continue;
        }

        if (ch == ' ') {
            t->paused = !t->paused;
            continue;
        }

        if (ch == '?') { show_help(); continue; }

        int c = tolower(ch);
        switch (c) {
            case 'x': *request_quit = 1; break;
            case 'n': t->step_once = 1; break;

            case 'q': t->c2 += 0.005f; *params_dirty = 1; break;
            case 'a': t->c2 -= 0.005f; *params_dirty = 1; break;

            case 'w': t->damping += 0.0003f; *params_dirty = 1; break;
            case 's': t->damping -= 0.0003f; *params_dirty = 1; break;

            case 'e': t->drive += 0.0000005f; *params_dirty = 1; break;
            case 'd': t->drive -= 0.0000005f; *params_dirty = 1; break;

            case 'r': t->kwtp += 0.001f; *params_dirty = 1; break;
            case 'f': t->kwtp -= 0.001f; *params_dirty = 1; break;

            case 't': t->feedback += 0.00001f; *params_dirty = 1; break;
            case 'g': t->feedback -= 0.00001f; *params_dirty = 1; break;

            case 'y': t->gravity += 0.0001f; break;
            case 'h': t->gravity -= 0.0001f; break;

            case 'u': t->lr += 0.00002f; break;
            case 'j': t->lr -= 0.00002f; break;

            case 'i': t->triad += 0.0001f; break;
            case 'k': t->triad -= 0.0001f; break;
            default: break;
        }

        if (t->c2 < 0.01f) t->c2 = 0.01f;
        if (t->c2 > 0.49f) t->c2 = 0.49f;
        if (t->damping < 0.94f) t->damping = 0.94f;
        if (t->damping > 0.9999f) t->damping = 0.9999f;
        if (t->drive < 0.0f) t->drive = 0.0f;
        if (t->drive > 0.00005f) t->drive = 0.00005f;
        if (t->kwtp < 0.0f) t->kwtp = 0.0f;
        if (t->kwtp > 0.08f) t->kwtp = 0.08f;
        if (t->feedback < 0.0f) t->feedback = 0.0f;
        if (t->feedback > 0.001f) t->feedback = 0.001f;
        if (t->gravity < 0.0f) t->gravity = 0.0f;
        if (t->gravity > 0.02f) t->gravity = 0.02f;
        if (t->lr < 0.000001f) t->lr = 0.000001f;
        if (t->lr > 0.01f) t->lr = 0.01f;
        if (t->triad < 0.0f) t->triad = 0.0f;
        if (t->triad > 0.5f) t->triad = 0.5f;
    }
}

static int parse_args(int argc, char** argv, HostOptions* opt) {
    opt->ticks = 1000;
    opt->device = 0;
    opt->substrates = DEFAULT_SUBSTRATES;
    opt->seed_mode = 0;
    opt->interactive = 1;
    opt->snap_every = 0;
    opt->snap_dir[0] = '\0';
    opt->resume_path[0] = '\0';
    opt->quantum_log[0] = '\0';

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--ticks") && i + 1 < argc) opt->ticks = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--device") && i + 1 < argc) opt->device = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--substrates") && i + 1 < argc) opt->substrates = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-interactive")) opt->interactive = 0;
        else if (!strcmp(argv[i], "--snap-every") && i + 1 < argc) opt->snap_every = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--snap-dir") && i + 1 < argc) {
            strncpy(opt->snap_dir, argv[++i], sizeof(opt->snap_dir) - 1);
            opt->snap_dir[sizeof(opt->snap_dir) - 1] = '\0';
        }
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) {
            const char* s = argv[++i];
            if (!strcmp(s, "impulse")) opt->seed_mode = 1;
            else if (!strcmp(s, "sparse")) opt->seed_mode = 2;
            else if (!strcmp(s, "quantum") && i + 1 < argc) {
                opt->seed_mode = 3;
                strncpy(opt->quantum_log, argv[++i], sizeof(opt->quantum_log) - 1);
                opt->quantum_log[sizeof(opt->quantum_log) - 1] = '\0';
            } else opt->seed_mode = 0;
        } else if (!strcmp(argv[i], "--resume") && i + 1 < argc) {
            strncpy(opt->resume_path, argv[++i], sizeof(opt->resume_path) - 1);
            opt->resume_path[sizeof(opt->resume_path) - 1] = '\0';
        }
    }

    if (opt->substrates < 2) opt->substrates = 2;
    if (opt->substrates > MAX_SUBSTRATES) opt->substrates = MAX_SUBSTRATES;
    return 0;
}

static void write_nca_weights(const char* path, const float* h_w, int n) {
    FILE* f = fopen(path, "wb");
    if (!f) return;
    fwrite(h_w, sizeof(float), (size_t)n, f);
    fclose(f);
}

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t width;
    uint32_t height;
    uint32_t active_mask;
    uint64_t tick;
} SnapshotHeader;

static int save_snapshot(const char* path, const SubstratePool* pool, uint64_t tick) {
    FILE* f = fopen(path, "wb");
    if (!f) return -1;

    SnapshotHeader h;
    h.magic = 0x4346504Cu; /* CFPL */
    h.version = 1;
    h.width = WIDTH;
    h.height = HEIGHT;
    h.active_mask = 0;
    for (int i = 0; i < MAX_SUBSTRATES; ++i) if (pool->s[i].active) h.active_mask |= (1u << i);
    h.tick = tick;
    fwrite(&h, sizeof(h), 1, f);

    float4* tmp_px = (float4*)malloc(CELLS * sizeof(float4));
    float4* tmp_wv = (float4*)malloc(CELLS * sizeof(float4));
    if (!tmp_px || !tmp_wv) {
        fclose(f);
        free(tmp_px); free(tmp_wv);
        return -1;
    }

    for (int i = 0; i < MAX_SUBSTRATES; ++i) {
        if (!pool->s[i].active) continue;
        const Plane* p = &pool->s[i].p;
        CHECK_CUDA(cudaMemcpy2D(tmp_px, WIDTH * sizeof(float4), p->pixel_curr, p->pixel_pitch,
                                WIDTH * sizeof(float4), HEIGHT, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy2D(tmp_wv, WIDTH * sizeof(float4), p->wave_curr, p->wave_pitch,
                                WIDTH * sizeof(float4), HEIGHT, cudaMemcpyDeviceToHost));
        fwrite(&i, sizeof(int), 1, f);
        fwrite(tmp_px, sizeof(float4), CELLS, f);
        fwrite(tmp_wv, sizeof(float4), CELLS, f);
    }

    free(tmp_px);
    free(tmp_wv);
    fclose(f);
    return 0;
}

static int load_snapshot(const char* path, SubstratePool* pool, uint64_t* tick_out) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;

    SnapshotHeader h;
    if (fread(&h, sizeof(h), 1, f) != 1) { fclose(f); return -1; }
    if (h.magic != 0x4346504Cu || h.width != WIDTH || h.height != HEIGHT) { fclose(f); return -1; }

    float4* tmp_px = (float4*)malloc(CELLS * sizeof(float4));
    float4* tmp_wv = (float4*)malloc(CELLS * sizeof(float4));
    if (!tmp_px || !tmp_wv) { fclose(f); free(tmp_px); free(tmp_wv); return -1; }

    for (int i = 0; i < MAX_SUBSTRATES; ++i) {
        if (!(h.active_mask & (1u << i))) continue;
        int sid;
        if (fread(&sid, sizeof(int), 1, f) != 1) break;
        if (!pool->s[sid].active) continue;
        if (fread(tmp_px, sizeof(float4), CELLS, f) != CELLS) break;
        if (fread(tmp_wv, sizeof(float4), CELLS, f) != CELLS) break;

        Plane* p = &pool->s[sid].p;
        CHECK_CUDA(cudaMemcpy2D(p->pixel_curr, p->pixel_pitch, tmp_px, WIDTH * sizeof(float4),
                                WIDTH * sizeof(float4), HEIGHT, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy2D(p->pixel_prev, p->pixel_pitch, tmp_px, WIDTH * sizeof(float4),
                                WIDTH * sizeof(float4), HEIGHT, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy2D(p->pixel_next, p->pixel_pitch, tmp_px, WIDTH * sizeof(float4),
                                WIDTH * sizeof(float4), HEIGHT, cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaMemcpy2D(p->wave_curr, p->wave_pitch, tmp_wv, WIDTH * sizeof(float4),
                                WIDTH * sizeof(float4), HEIGHT, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy2D(p->wave_prev, p->wave_pitch, tmp_wv, WIDTH * sizeof(float4),
                                WIDTH * sizeof(float4), HEIGHT, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy2D(p->wave_next, p->wave_pitch, tmp_wv, WIDTH * sizeof(float4),
                                WIDTH * sizeof(float4), HEIGHT, cudaMemcpyHostToDevice));
    }

    *tick_out = h.tick;
    free(tmp_px);
    free(tmp_wv);
    fclose(f);
    return 0;
}

static void snapshot_png(const Substrate* s, uint64_t tick, const char* snap_dir) {
    char path_px[512];
    char path_dm[512];
    if (snap_dir && snap_dir[0]) {
        snprintf(path_px, sizeof(path_px), "%s/pixel_%06llu.png", snap_dir, (unsigned long long)tick);
        snprintf(path_dm, sizeof(path_dm), "%s/delta_%06llu.png", snap_dir, (unsigned long long)tick);
    } else {
        snprintf(path_px, sizeof(path_px), "pixel_%06llu.png", (unsigned long long)tick);
        snprintf(path_dm, sizeof(path_dm), "delta_%06llu.png", (unsigned long long)tick);
    }

    float4* px = (float4*)malloc(CELLS * sizeof(float4));
    float* dm = (float*)malloc(CELLS * sizeof(float));
    unsigned char* rgba = (unsigned char*)malloc(CELLS * 4);
    unsigned char* gray = (unsigned char*)malloc(CELLS * 4);
    if (!px || !dm || !rgba || !gray) {
        free(px); free(dm); free(rgba); free(gray);
        return;
    }

    CHECK_CUDA(cudaMemcpy2D(px, WIDTH * sizeof(float4), s->p.pixel_curr, s->p.pixel_pitch,
                            WIDTH * sizeof(float4), HEIGHT, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy2D(dm, WIDTH * sizeof(float), s->p.delta_mag, s->p.scalar_pitch,
                            WIDTH * sizeof(float), HEIGHT, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < CELLS; ++i) {
        float4 p = px[i];
        rgba[i * 4 + 0] = (unsigned char)(fminf(1.f, fmaxf(0.f, p.x)) * 255.f);
        rgba[i * 4 + 1] = (unsigned char)(fminf(1.f, fmaxf(0.f, p.y)) * 255.f);
        rgba[i * 4 + 2] = (unsigned char)(fminf(1.f, fmaxf(0.f, p.z)) * 255.f);
        rgba[i * 4 + 3] = (unsigned char)(fminf(1.f, fmaxf(0.f, p.w)) * 255.f);

        float m = fminf(1.f, dm[i]);
        unsigned char g = (unsigned char)(m * 255.f);
        gray[i * 4 + 0] = g;
        gray[i * 4 + 1] = g;
        gray[i * 4 + 2] = g;
        gray[i * 4 + 3] = 255;
    }

    write_png_rgba8(path_px, rgba, WIDTH, HEIGHT);
    write_png_rgba8(path_dm, gray, WIDTH, HEIGHT);

    free(px); free(dm); free(rgba); free(gray);
}

static int edge_exists(const EntEdge* edges, int n, uint16_t src_sub, uint16_t dst_sub, uint32_t src_idx, uint32_t dst_idx) {
    uint32_t sx = src_idx % WIDTH, sy = src_idx / WIDTH;
    uint32_t dx = dst_idx % WIDTH, dy = dst_idx / WIDTH;
    for (int i = 0; i < n; ++i) {
        if (edges[i].src_sub == src_sub && edges[i].dst_sub == dst_sub &&
            edges[i].src_x == sx && edges[i].src_y == sy &&
            edges[i].dst_x == dx && edges[i].dst_y == dy) {
            return 1;
        }
    }
    return 0;
}

static void add_edge(EntEdge** edges, int* n, int* cap,
                     uint16_t src_sub, uint16_t dst_sub,
                     uint32_t src_idx, uint32_t dst_idx,
                     float strength) {
    if (src_sub >= dst_sub) return;
    if (edge_exists(*edges, *n, src_sub, dst_sub, src_idx, dst_idx)) return;

    if (*n + 1 > *cap) {
        int ncap = (*cap == 0) ? 4096 : (*cap * 2);
        EntEdge* ne = (EntEdge*)realloc(*edges, (size_t)ncap * sizeof(EntEdge));
        if (!ne) return;
        *edges = ne;
        *cap = ncap;
    }

    EntEdge e;
    e.src_sub = src_sub;
    e.dst_sub = dst_sub;
    e.src_x = src_idx % WIDTH;
    e.src_y = src_idx / WIDTH;
    e.dst_x = dst_idx % WIDTH;
    e.dst_y = dst_idx / WIDTH;
    e.strength = strength;
    (*edges)[(*n)++] = e;
}

static void remove_edges_for_substrate(EntEdge** edges, int* n, int sid) {
    int write = 0;
    for (int i = 0; i < *n; ++i) {
        if ((int)(*edges)[i].src_sub == sid || (int)(*edges)[i].dst_sub == sid) continue;
        (*edges)[write++] = (*edges)[i];
    }
    *n = write;
}

static void init_sparse_edges(EntEdge** edges, int* n, int* cap, int active_substrates, uint32_t seed) {
    uint32_t s = seed;
    int samples = (int)(CELLS * EDGE_DENSITY);
    for (int a = 0; a < active_substrates; ++a) {
        for (int b = a + 1; b < active_substrates; ++b) {
            for (int i = 0; i < samples; ++i) {
                s ^= s << 13; s ^= s >> 17; s ^= s << 5;
                uint32_t src = s % (uint32_t)CELLS;
                s ^= s << 13; s ^= s >> 17; s ^= s << 5;
                uint32_t dst = s % (uint32_t)CELLS;
                add_edge(edges, n, cap, (uint16_t)a, (uint16_t)b, src, dst, 0.08f);
            }
        }
    }
}

static void inherit_edge_subset(EntEdge** edges, int* n, int* cap, int parent, int child, uint32_t seed) {
    uint32_t s = seed;
    int orig = *n;
    for (int i = 0; i < orig; ++i) {
        const EntEdge* e = &(*edges)[i];
        if ((int)e->src_sub != parent && (int)e->dst_sub != parent) continue;
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        if ((s & 3u) != 0u) continue; /* 25% subset */

        uint16_t src = e->src_sub;
        uint16_t dst = e->dst_sub;
        if (src == (uint16_t)parent) src = (uint16_t)child;
        if (dst == (uint16_t)parent) dst = (uint16_t)child;

        if (src == dst) continue;
        if (src > dst) {
            uint16_t t = src; src = dst; dst = t;
        }

        uint32_t src_idx = e->src_y * WIDTH + e->src_x;
        uint32_t dst_idx = e->dst_y * WIDTH + e->dst_x;
        add_edge(edges, n, cap, src, dst, src_idx, dst_idx, e->strength);
    }
}

int main(int argc, char** argv) {
    enable_ansi();
    HostOptions opt;
    parse_args(argc, argv, &opt);

    CHECK_CUDA(cudaSetDevice(opt.device));

    /* Build 256-bin color→frequency constant memory table */
    init_freq_table();

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, opt.device));

    DashState dash;
    memset(&dash, 0, sizeof(dash));
    dash.ticks_goal = opt.ticks;
    dash.interactive = opt.interactive;
    snprintf(dash.device_name, sizeof(dash.device_name), "%s (sm_%d%d)",
             prop.name, prop.major, prop.minor);

    LiveTunables tune;
    tune.c2 = BASE_C2;
    tune.damping = BASE_DAMPING;
    tune.drive = BASE_DRIVE;
    tune.kwtp = BASE_KWTP;
    tune.feedback = BASE_FEEDBACK;
    tune.gravity = BASE_GRAVITY;
    tune.lr = BASE_LR;
    tune.triad = 0.06f;   /* wider default — clearly visible complement attraction/repulsion */
    tune.paused = 0;
    tune.step_once = 0;

    /* Load quantum bytes before allocating substrates so we can use them for seeding */
    uint8_t* h_qbytes = NULL;
    size_t   h_qlen   = 0;
    uint8_t* d_qbytes = NULL;
    if (opt.seed_mode == 3 && opt.quantum_log[0]) {
        h_qbytes = load_quantum_bytes(opt.quantum_log, &h_qlen);
        if (h_qbytes && h_qlen > 0) {
            CHECK_CUDA(cudaMalloc(&d_qbytes, h_qlen));
            CHECK_CUDA(cudaMemcpy(d_qbytes, h_qbytes, h_qlen, cudaMemcpyHostToDevice));
            free(h_qbytes); h_qbytes = NULL;
            fprintf(stderr, "[quantum] %zu bytes on GPU\n", h_qlen);
        } else {
            fprintf(stderr, "[quantum] log empty or unreadable; falling back to xorshift\n");
            opt.seed_mode = 0;
        }
    }

    SubstratePool pool;
    memset(&pool, 0, sizeof(pool));

    /* Use seed_mode=0 (xorshift) for the initial kernel, then overwrite with quantum if available */
    int base_seed_mode = (opt.seed_mode == 3) ? 0 : opt.seed_mode;
    for (int i = 0; i < opt.substrates; ++i) {
        int locked = (i == 0) ? 1 : 0;
        uint32_t seed = 0x12345678u + (uint32_t)i * 0x9E3779B9u;
        if (pool_add_substrate(&pool, i, locked, 0, (i == 0 ? -1 : 0), base_seed_mode, seed) < 0) {
            fprintf(stderr, "unable to allocate substrate %d; stopping at %d\n", i, pool.active_count);
            break;
        }
    }
    if (pool.active_count < 2) {
        fprintf(stderr, "need at least 2 substrates\n");
        return 1;
    }

    /* Overwrite pixel fields with quantum bytes if available */
    if (opt.seed_mode == 3 && d_qbytes && h_qlen > 0) {
        dim3 qb(BX, BY);
        dim3 qg((WIDTH + BX - 1) / BX, (HEIGHT + BY - 1) / BY);
        for (int i = 0; i < MAX_SUBSTRATES; ++i) {
            if (!pool.s[i].active) continue;
            Plane* p = &pool.s[i].p;
            /* Offset bytes per substrate so each gets a unique slice */
            size_t offset = (size_t)i * CELLS * 3;
            const uint8_t* qptr = d_qbytes + (offset < h_qlen ? offset : 0);
            size_t qavail = (offset < h_qlen) ? (h_qlen - offset) : h_qlen; /* wrap if needed */
            quantum_seed_kernel<<<qg, qb>>>(p->pixel_curr, p->pixel_pitch,
                                            p->wave_curr,  p->wave_pitch,
                                            qptr, qavail);
            quantum_seed_kernel<<<qg, qb>>>(p->pixel_prev, p->pixel_pitch,
                                            p->wave_prev,  p->wave_pitch,
                                            qptr, qavail);
            quantum_seed_kernel<<<qg, qb>>>(p->pixel_next, p->pixel_pitch,
                                            p->wave_next,  p->wave_pitch,
                                            qptr, qavail);
        }
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        cudaFree(d_qbytes); d_qbytes = NULL;
        fprintf(stderr, "[quantum] pixel fields seeded from quantum entropy\n");
    }

    if (opt.resume_path[0]) {
        uint64_t rtick = 0;
        if (load_snapshot(opt.resume_path, &pool, &rtick) == 0) {
            fprintf(stderr, "resumed from %s at tick=%llu\n", opt.resume_path, (unsigned long long)rtick);
        } else {
            fprintf(stderr, "resume failed for %s; starting fresh\n", opt.resume_path);
        }
    }

    EntEdge* h_edges = NULL;
    int edge_count = 0;
    int edge_cap = 0;
    init_sparse_edges(&h_edges, &edge_count, &edge_cap, pool.active_count, 0xBEEFCAFEu);

    EntEdge* d_edges = NULL;
    int d_edges_capacity = 0;
    int d_edges_uploaded = -1;
    int edge_overflow_warned = 0;
    DevSrcView* d_src_views = NULL;
    DevDstView* d_dst_views = NULL;
    CHECK_CUDA(cudaMalloc(&d_src_views, MAX_SUBSTRATES * sizeof(DevSrcView)));
    CHECK_CUDA(cudaMalloc(&d_dst_views, MAX_SUBSTRATES * sizeof(DevDstView)));

    PairCandidate* d_candidates = NULL;
    uint32_t* d_candidate_count = NULL;
    CHECK_CUDA(cudaMalloc(&d_candidates, MAX_CANDIDATES * sizeof(PairCandidate)));
    CHECK_CUDA(cudaMalloc(&d_candidate_count, sizeof(uint32_t)));

    float* d_pair_corr = NULL;
    CHECK_CUDA(cudaMalloc(&d_pair_corr, sizeof(float)));

    NaturalEntRecord* natural_log = NULL;
    int natural_n = 0;
    int natural_cap = 0;

    FILE* f_act = fopen("activations.csv", "a");
    FILE* f_energy = fopen("energy.csv", "a");
    FILE* f_de = fopen("delta_energy.csv", "a");
    FILE* f_nca = fopen("nca_training.csv", "a");
    if (!f_act || !f_energy || !f_de || !f_nca) {
        fprintf(stderr, "failed opening csv outputs\n");
        return 1;
    }

    if (ftell(f_act) == 0) fprintf(f_act, "tick,active\n");
    if (ftell(f_energy) == 0) fprintf(f_energy, "tick,wave_energy\n");
    if (ftell(f_de) == 0) fprintf(f_de, "tick,delta_energy\n");
    if (ftell(f_nca) == 0) fprintf(f_nca, "tick,adam_step,nca_loss,nca_residual,grad_norm\n");

    float* nca_w = (float*)malloc(NCA_PARAMS * sizeof(float));
    float* nca_g = (float*)malloc(NCA_PARAMS * sizeof(float));
    if (!nca_w || !nca_g) return 1;
    memset(nca_g, 0, NCA_PARAMS * sizeof(float));

    /* He (Kaiming) initialization for ReLU layers.
       For a uniform distribution U[-a, a], variance = a²/3.  To achieve He's
       target variance of 2/fan_in (for ReLU), we need a = sqrt(6/fan_in).
       This differs from the normal-distribution formulation (std = sqrt(2/fan_in))
       but is mathematically equivalent.
       W1: fan_in = NCA_IN  → a = sqrt(6/NCA_IN) ≈ 0.21
       W2: fan_in = NCA_H   → a = sqrt(6/NCA_H)  ≈ 0.43
       b1: small positive constant (0.01) to break dead-ReLU symmetry at init
       b2: zero */
    float w1_scale = sqrtf(6.0f / (float)NCA_IN);
    float w2_scale = sqrtf(6.0f / (float)NCA_H);
    uint32_t rs = 0xCAFED00Du;
    for (int i = 0; i < NCA_PARAMS; ++i) {
        rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5;
        float rnd = ((float)(rs & 0xFFFFu) / 65535.0f) - 0.5f;  /* [-0.5, 0.5) */
        if (i < NCA_W1) {
            nca_w[i] = rnd * 2.0f * w1_scale;
        } else if (i < NCA_W1 + NCA_B1) {
            nca_w[i] = 0.01f;   /* small positive bias — keeps ReLUs alive at init */
        } else if (i < NCA_W1 + NCA_B1 + NCA_W2) {
            nca_w[i] = rnd * 2.0f * w2_scale;
        } else {
            nca_w[i] = 0.0f;    /* output biases zero */
        }
    }

    float* d_w = NULL;
    float* d_g = NULL;
    float* d_m = NULL;   /* Adam first moment  (momentum) */
    float* d_v = NULL;   /* Adam second moment (RMS estimate) */
    int    adam_t = 0;   /* Adam step counter for bias correction */
    CHECK_CUDA(cudaMalloc(&d_w, NCA_PARAMS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_g, NCA_PARAMS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_m, NCA_PARAMS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_v, NCA_PARAMS * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_w, nca_w, NCA_PARAMS * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_g, 0, NCA_PARAMS * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_m, 0, NCA_PARAMS * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_v, 0, NCA_PARAMS * sizeof(float)));

    unsigned long long* d_active = NULL;
    float* d_wave_energy = NULL;
    float* d_delta_energy = NULL;
    float* d_nca_loss = NULL;
    float* d_nca_residual = NULL;
    float* d_gnorm = NULL;
    CHECK_CUDA(cudaMalloc(&d_active,       sizeof(unsigned long long)));
    CHECK_CUDA(cudaMalloc(&d_wave_energy,  sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_delta_energy, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_nca_loss,     sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_nca_residual, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_gnorm,        sizeof(float)));

    /* XOR broadcast buffers */
    float4* d_wv_lap = NULL;   /* pre-computed wave Laplacian field */
    float*  d_xor_buf = NULL;  /* trigger color slots */
    int*    d_xor_count = NULL;
    CHECK_CUDA(cudaMalloc(&d_wv_lap,    CELLS * sizeof(float4)));
    CHECK_CUDA(cudaMalloc(&d_xor_buf,   MAX_XOR_SLOTS * 5 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_xor_count, sizeof(int)));

    cudaStream_t stream0, stream2;
    CHECK_CUDA(cudaStreamCreate(&stream0));
    CHECK_CUDA(cudaStreamCreate(&stream2));

    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));
    CHECK_CUDA(cudaEventRecord(t0, stream0));

    dim3 block(BX, BY);
    dim3 grid((WIDTH + BX - 1) / BX, (HEIGHT + BY - 1) / BY);

    if (edge_cap > 0) {
        CHECK_CUDA(cudaMalloc(&d_edges, (size_t)edge_cap * sizeof(EntEdge)));
        d_edges_capacity = edge_cap;
    }

    /* Prime one-step-lag NCA I/O buffers before entering the training loop. */
    for (int sid = 0; sid < MAX_SUBSTRATES; ++sid) {
        if (!pool.s[sid].active || pool.s[sid].temporary) continue;
        Plane* p = &pool.s[sid].p;
        ring_write_kernel<<<grid, block, 0, stream0>>>(
            p->pixel_curr, p->pixel_pitch,
            p->pixel_prev, p->pixel_pitch,
            p->ring, p->ring_head);
        p->ring_head = (p->ring_head + 1) % HIST_TICKS;
        nca_build_io_kernel<<<grid, block, 0, stream0>>>(
            p->pixel_curr, p->pixel_pitch,
            p->pixel_prev, p->pixel_pitch,
            p->ring, p->ring_head,
            p->input, p->target, p->prev_target);
    }
    CHECK_CUDA(cudaGetLastError());

    int no_new_ent_ticks = 0;
    int ticks_executed = 0;
    int request_quit = 0;

    for (int tick = 1; tick <= opt.ticks; ++tick) {
        int params_dirty = 0;
        if (opt.interactive) {
            process_live_controls(&tune, &params_dirty, &request_quit);
            if (request_quit) break;
            if (params_dirty) {
                apply_tunables_to_pool(&pool, &tune, stream0);
                fprintf(stderr, "TUNE c2=%.4f damp=%.5f drive=%.7g kwtp=%.4f fb=%.6f grav=%.7f lr=%.6f\n",
                        tune.c2, tune.damping, tune.drive, tune.kwtp, tune.feedback, tune.gravity, tune.lr);
                fflush(stderr);
            }
            if (tune.paused && !tune.step_once) {
                dash.paused = 1;
                dash.tune_c2 = tune.c2;
                dash.tune_damping = tune.damping;
                dash.tune_drive = tune.drive;
                dash.tune_kwtp = tune.kwtp;
                dash.tune_fb = tune.feedback;
                dash.tune_gravity = tune.gravity;
                dash.tune_lr = tune.lr;
                dash.tune_triad = tune.triad;
                draw_dashboard(&dash);
                cli_sleep_ms(16);
                tick -= 1;
                continue;
            }
        }
        tune.step_once = 0;

        /* Physics tick on all active substrates */
        for (int sid = 0; sid < MAX_SUBSTRATES; ++sid) {
            if (!pool.s[sid].active) continue;
            Plane* p = &pool.s[sid].p;
            physics_tick_kernel<<<grid, block, 0, stream0>>>(
                p->pixel_curr, p->pixel_pitch,
                p->pixel_prev, p->pixel_pitch,
                p->wave_curr, p->wave_pitch,
                p->wave_prev, p->wave_pitch,
                p->pixel_next, p->pixel_pitch,
                p->wave_next, p->wave_pitch,
                p->delta_next, p->delta_pitch,
                p->delta_mag, p->scalar_pitch,
                p->c2, p->scalar_pitch,
                p->damping, p->scalar_pitch,
                p->drive_gain, p->scalar_pitch,
                p->k_wave_to_px, p->scalar_pitch);
        }
        CHECK_CUDA(cudaGetLastError());

        /* Pixel chromatic gravity — clusters similar hues directly on canvas */
        if (tune.gravity != 0.0f) {
            for (int sid = 0; sid < MAX_SUBSTRATES; ++sid) {
                if (!pool.s[sid].active) continue;
                Plane* p = &pool.s[sid].p;
                pixel_gravity_kernel<<<grid, block, 0, stream0>>>(
                    p->pixel_curr, p->pixel_pitch,
                    p->pixel_next, p->pixel_pitch,
                    WIDTH, HEIGHT, tune.gravity * 10.0f);
            }
            CHECK_CUDA(cudaGetLastError());
        }

        /* Collide (destructive) interference — wave zero-crossings flip pixel
           toward its complement.  This is the entropy mechanism that keeps the
           system from settling into stable attractors.  Fires whenever two waves
           collide and cancel, disrupting whatever cluster structure is present. */
        for (int sid = 0; sid < MAX_SUBSTRATES; ++sid) {
            if (!pool.s[sid].active) continue;
            Plane* p = &pool.s[sid].p;
            collide_interference_kernel<<<grid, block, 0, stream0>>>(
                p->wave_curr, p->wave_pitch,
                p->wave_prev, p->wave_pitch,
                p->pixel_curr, p->pixel_pitch,
                p->pixel_next, p->pixel_pitch,
                WIDTH, HEIGHT, tune.kwtp * COLLIDE_STRENGTH_MULT);
        }
        CHECK_CUDA(cudaGetLastError());

        /* Complementary neighbor force — Mexican hat on hue wheel.
           Exact opposite attracts, one-shade-off repels. */
        if (tune.triad != 0.0f) {
            for (int sid = 0; sid < MAX_SUBSTRATES; ++sid) {
                if (!pool.s[sid].active) continue;
                Plane* p = &pool.s[sid].p;
                complementary_neighbor_kernel<<<grid, block, 0, stream0>>>(
                    p->pixel_curr, p->pixel_pitch,
                    p->pixel_next, p->pixel_pitch,
                    WIDTH, HEIGHT, tune.triad);
            }
            CHECK_CUDA(cudaGetLastError());
        }

        /* XOR broadcast — Laplacian → collect triggers → broadcast to canvas */
        if (tune.kwtp > 0.0f) {
            for (int sid = 0; sid < MAX_SUBSTRATES; ++sid) {
                if (!pool.s[sid].active) continue;
                Plane* p = &pool.s[sid].p;

                /* Compute wave Laplacian into flat buffer */
                wave_laplacian_kernel<<<grid, block, 0, stream0>>>(
                    p->wave_curr, p->wave_pitch,
                    d_wv_lap, WIDTH, HEIGHT);

                /* Reset XOR slot counter */
                CHECK_CUDA(cudaMemsetAsync(d_xor_count, 0, sizeof(int), stream0));

                /* Pass 1: collect trigger colors at crossing points */
                xor_collect_kernel<<<grid, block, 0, stream0>>>(
                    p->pixel_curr, p->pixel_pitch,
                    d_wv_lap, d_xor_buf, d_xor_count,
                    tune.kwtp, WIDTH, HEIGHT);

                /* Pass 2: broadcast complement to all matching pixels */
                int h_xor_count = 0;
                CHECK_CUDA(cudaMemcpy(&h_xor_count, d_xor_count, sizeof(int), cudaMemcpyDeviceToHost));
                if (h_xor_count > 0) {
                    xor_broadcast_kernel<<<grid, block, 0, stream0>>>(
                        p->pixel_curr, p->pixel_pitch,
                        p->pixel_next, p->pixel_pitch,
                        d_xor_buf, h_xor_count,
                        0.04f, WIDTH, HEIGHT);
                }
            }
            CHECK_CUDA(cudaGetLastError());
        }

        /* Entanglement sync */
        if (edge_count > 0 && d_edges_capacity > 0) {
            int edge_count_gpu = edge_count;
            if (edge_count_gpu > d_edges_capacity) {
                edge_count_gpu = d_edges_capacity;
                if (!edge_overflow_warned) {
                    fprintf(stderr, "warning: edge_count (%d) exceeded GPU capacity (%d), clamping\n", edge_count, d_edges_capacity);
                    edge_overflow_warned = 1;
                }
            }
            if (edge_count_gpu != d_edges_uploaded) {
                CHECK_CUDA(cudaMemcpyAsync(d_edges, h_edges, (size_t)edge_count_gpu * sizeof(EntEdge), cudaMemcpyHostToDevice, stream0));
                d_edges_uploaded = edge_count_gpu;
            }

            DevSrcView h_src[MAX_SUBSTRATES];
            DevDstView h_dst[MAX_SUBSTRATES];
            memset(h_src, 0, sizeof(h_src));
            memset(h_dst, 0, sizeof(h_dst));
            for (int sid = 0; sid < MAX_SUBSTRATES; ++sid) {
                if (!pool.s[sid].active) continue;
                h_src[sid].px_next = pool.s[sid].p.pixel_next;
                h_src[sid].wv_next = pool.s[sid].p.wave_next;
                h_src[sid].px_pitch = pool.s[sid].p.pixel_pitch;
                h_src[sid].wv_pitch = pool.s[sid].p.wave_pitch;

                h_dst[sid].px_next = pool.s[sid].p.pixel_next;
                h_dst[sid].wv_next = pool.s[sid].p.wave_next;
                h_dst[sid].px_pitch = pool.s[sid].p.pixel_pitch;
                h_dst[sid].wv_pitch = pool.s[sid].p.wave_pitch;
            }
            CHECK_CUDA(cudaMemcpyAsync(d_src_views, h_src, sizeof(h_src), cudaMemcpyHostToDevice, stream0));
            CHECK_CUDA(cudaMemcpyAsync(d_dst_views, h_dst, sizeof(h_dst), cudaMemcpyHostToDevice, stream0));

            int eb = 256;
            int eg = (edge_count_gpu + eb - 1) / eb;
            entangle_sync_kernel<<<eg, eb, 0, stream0>>>(d_edges, edge_count_gpu, d_src_views, d_dst_views);
            CHECK_CUDA(cudaGetLastError());
        }

        /* Rotate buffers */
        for (int sid = 0; sid < MAX_SUBSTRATES; ++sid) {
            if (!pool.s[sid].active) continue;
            rotate_plane(&pool.s[sid].p);
        }

        for (int sid = 0; sid < MAX_SUBSTRATES; ++sid) {
            if (!pool.s[sid].active) continue;
            color_gravity_kernel<<<grid, block, 0, stream0>>>(
                pool.s[sid].p.pixel_curr, pool.s[sid].p.pixel_pitch,
                pool.s[sid].p.wave_curr, pool.s[sid].p.wave_pitch,
                WIDTH, HEIGHT, tune.gravity, 2, 2.0f);
        }

        /* d_g is zeroed inside adam_step_kernel at the end of each update step.
           No separate memset needed — the Adam kernel handles it, and the initial
           cudaMemset at allocation primes the very first accumulation window. */

        /* Comb + NCA forward/backward every NCA_FWD_PERIOD ticks */
        if ((tick % NCA_FWD_PERIOD) == 0) for (int sid = 0; sid < MAX_SUBSTRATES; ++sid) {
            if (!pool.s[sid].active) continue;
            Plane* p = &pool.s[sid].p;

            comb_update_kernel<<<grid, block, 0, stream0>>>(
                p->delta_mag, p->scalar_pitch,
                p->wave_curr, p->wave_pitch,
                p->act_count, p->ewma_delta, p->dir_hist);

            /* Temporary substrates have no NCA buffers — skip */
            if (pool.s[sid].temporary) continue;

            nca_build_io_kernel<<<grid, block, 0, stream0>>>(
                p->pixel_curr, p->pixel_pitch,
                p->pixel_prev, p->pixel_pitch,
                p->ring, p->ring_head,
                p->input, p->target, p->prev_target);

            int cb = 128;
            int cg = ((int)CELLS + cb - 1) / cb;
            const float* w1 = d_w;
            const float* b1 = d_w + NCA_W1;
            const float* w2 = d_w + NCA_W1 + NCA_B1;
            const float* b2 = d_w + NCA_W1 + NCA_B1 + NCA_W2;

            float* gw1 = d_g;
            float* gb1 = d_g + NCA_W1;
            float* gw2 = d_g + NCA_W1 + NCA_B1;
            float* gb2 = d_g + NCA_W1 + NCA_B1 + NCA_W2;

            nca_forward_kernel<<<cg, cb, 0, stream0>>>(p->input, w1, b1, w2, b2, p->hidden, p->output);

            if (!pool.s[sid].write_locked) {
                nca_backward_feedback_kernel<<<grid, block, 0, stream0>>>(
                    p->input, p->hidden, p->output, p->target,
                    w2,
                    gw1, gb1, gw2, gb2,
                    p->residual,
                    p->wave_curr, p->wave_pitch,
                    p->feedback_gain, p->scalar_pitch);
            }

            ring_write_kernel<<<grid, block, 0, stream0>>>(
                p->pixel_curr, p->pixel_pitch,
                p->pixel_prev, p->pixel_pitch,
                p->ring, p->ring_head);
            p->ring_head = (p->ring_head + 1) % HIST_TICKS;
        }

        /* Adam weight update: fires every NCA_UPDATE_PERIOD ticks.
           By this point NCA_UPDATE_PERIOD/NCA_FWD_PERIOD backward passes have
           been accumulated into d_g, giving a larger effective batch. */
        if ((tick % NCA_UPDATE_PERIOD) == 0) {
            int pb = 256;
            int pg = (NCA_PARAMS + pb - 1) / pb;

            /* Global gradient norm for clipping */
            #define GRAD_CLIP 1.0f
            CHECK_CUDA(cudaMemsetAsync(d_gnorm, 0, sizeof(float), stream0));
            /* Normalise by total accumulated samples:
               NCA_ACCUMULATIONS backward passes × CELLS samples each */
            float n_acc = (float)NCA_ACCUMULATIONS * (float)CELLS;
            grad_norm_kernel<<<pg, pb, 0, stream0>>>(d_g, NCA_PARAMS, 1.0f / n_acc, d_gnorm);
            CHECK_CUDA(cudaStreamSynchronize(stream0));
            float h_gnorm = 0.0f;
            CHECK_CUDA(cudaMemcpy(&h_gnorm, d_gnorm, sizeof(float), cudaMemcpyDeviceToHost));
            float gnorm = sqrtf(h_gnorm);
            float clip_scale = (gnorm > GRAD_CLIP) ? (GRAD_CLIP / gnorm) : 1.0f;

            /* Adam step with bias correction */
            adam_t++;
            float bc1 = 1.0f / (1.0f - powf(ADAM_BETA1, (float)adam_t));
            float bc2 = 1.0f / (1.0f - powf(ADAM_BETA2, (float)adam_t));
            adam_step_kernel<<<pg, pb, 0, stream0>>>(d_w, d_g, d_m, d_v, NCA_PARAMS,
                                                      tune.lr, 1.0f / n_acc, clip_scale,
                                                      ADAM_BETA1, ADAM_BETA2, ADAM_EPS,
                                                      bc1, bc2);
            dash.grad_norm = gnorm;
        }

        /* One experiment per 128 ticks on stream2 */
        if ((tick % EXPERIMENT_PERIOD) == 0) {
            int base = 1;
            if (pool.s[base].active) {
                CHECK_CUDA(cudaMemsetAsync(d_candidate_count, 0, sizeof(uint32_t), stream2));
                candidate_pairs_kernel<<<grid, block, 0, stream2>>>(
                    pool.s[base].p.act_count,
                    pool.s[base].p.ewma_delta,
                    d_candidates,
                    d_candidate_count,
                    MAX_CANDIDATES);
                CHECK_CUDA(cudaGetLastError());

                CHECK_CUDA(cudaStreamSynchronize(stream2));
                uint32_t h_count = 0;
                CHECK_CUDA(cudaMemcpy(&h_count, d_candidate_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
                if (h_count > MAX_CANDIDATES) h_count = MAX_CANDIDATES;

                if (h_count > 0) {
                    PairCandidate chosen;
                    CHECK_CUDA(cudaMemcpy(&chosen, d_candidates, sizeof(PairCandidate), cudaMemcpyDeviceToHost));

                    int exp_sid = -1;
                    for (int i = 2; i < MAX_SUBSTRATES; ++i) {
                        if (!pool.s[i].active) { exp_sid = i; break; }
                    }

                    if (exp_sid >= 0) {
                        uint32_t seed = 0xA0000000u ^ (uint32_t)tick;
                        if (pool_add_substrate(&pool, exp_sid, 0, 1, base, opt.seed_mode, seed) >= 0) {
                            inherit_edge_subset(&h_edges, &edge_count, &edge_cap, base, exp_sid, (uint32_t)tick);

                            CHECK_CUDA(cudaMemcpy2D(pool.s[exp_sid].p.pixel_curr, pool.s[exp_sid].p.pixel_pitch,
                                                    pool.s[base].p.pixel_curr, pool.s[base].p.pixel_pitch,
                                                    WIDTH * sizeof(float4), HEIGHT, cudaMemcpyDeviceToDevice));
                            CHECK_CUDA(cudaMemcpy2D(pool.s[exp_sid].p.pixel_prev, pool.s[exp_sid].p.pixel_pitch,
                                                    pool.s[base].p.pixel_prev, pool.s[base].p.pixel_pitch,
                                                    WIDTH * sizeof(float4), HEIGHT, cudaMemcpyDeviceToDevice));
                            CHECK_CUDA(cudaMemcpy2D(pool.s[exp_sid].p.wave_curr, pool.s[exp_sid].p.wave_pitch,
                                                    pool.s[base].p.wave_curr, pool.s[base].p.wave_pitch,
                                                    WIDTH * sizeof(float4), HEIGHT, cudaMemcpyDeviceToDevice));
                            CHECK_CUDA(cudaMemcpy2D(pool.s[exp_sid].p.wave_prev, pool.s[exp_sid].p.wave_pitch,
                                                    pool.s[base].p.wave_prev, pool.s[base].p.wave_pitch,
                                                    WIDTH * sizeof(float4), HEIGHT, cudaMemcpyDeviceToDevice));

                            pair_corr_kernel<<<1, 1, 0, stream2>>>(pool.s[base].p.delta_curr, pool.s[base].p.delta_pitch,
                                                                   chosen.a, chosen.b, d_pair_corr);
                            CHECK_CUDA(cudaStreamSynchronize(stream2));
                            float baseline = 0.0f;
                            CHECK_CUDA(cudaMemcpy(&baseline, d_pair_corr, sizeof(float), cudaMemcpyDeviceToHost));

                            force_pair_average_kernel<<<1, 1, 0, stream2>>>(
                                pool.s[exp_sid].p.pixel_curr, pool.s[exp_sid].p.pixel_pitch,
                                pool.s[exp_sid].p.wave_curr, pool.s[exp_sid].p.wave_pitch,
                                chosen.a, chosen.b);

                            for (int t = 0; t < EXPERIMENT_TICKS; ++t) {
                                Plane* p = &pool.s[exp_sid].p;
                                physics_tick_kernel<<<grid, block, 0, stream2>>>(
                                    p->pixel_curr, p->pixel_pitch,
                                    p->pixel_prev, p->pixel_pitch,
                                    p->wave_curr, p->wave_pitch,
                                    p->wave_prev, p->wave_pitch,
                                    p->pixel_next, p->pixel_pitch,
                                    p->wave_next, p->wave_pitch,
                                    p->delta_next, p->delta_pitch,
                                    p->delta_mag, p->scalar_pitch,
                                    p->c2, p->scalar_pitch,
                                    p->damping, p->scalar_pitch,
                                    p->drive_gain, p->scalar_pitch,
                                    p->k_wave_to_px, p->scalar_pitch);
                                rotate_plane(p);
                            }

                            pair_corr_kernel<<<1, 1, 0, stream2>>>(pool.s[exp_sid].p.delta_curr, pool.s[exp_sid].p.delta_pitch,
                                                                   chosen.a, chosen.b, d_pair_corr);
                            CHECK_CUDA(cudaStreamSynchronize(stream2));
                            float experiment = 0.0f;
                            CHECK_CUDA(cudaMemcpy(&experiment, d_pair_corr, sizeof(float), cudaMemcpyDeviceToHost));

                            if (experiment > baseline) {
                                if (natural_n + 1 > natural_cap) {
                                    int ncap = (natural_cap == 0) ? 1024 : (natural_cap * 2);
                                    NaturalEntRecord* nr = (NaturalEntRecord*)realloc(natural_log, (size_t)ncap * sizeof(NaturalEntRecord));
                                    if (nr) { natural_log = nr; natural_cap = ncap; }
                                }
                                if (natural_n < natural_cap) {
                                    natural_log[natural_n].tick = (uint64_t)tick;
                                    natural_log[natural_n].a = chosen.a;
                                    natural_log[natural_n].b = chosen.b;
                                    natural_log[natural_n].baseline = baseline;
                                    natural_log[natural_n].experiment = experiment;
                                    natural_n++;
                                }

                                add_edge(&h_edges, &edge_count, &edge_cap,
                                         0, 1, chosen.a, chosen.b, 0.12f);
                                no_new_ent_ticks = 0;
                            }

                            remove_edges_for_substrate(&h_edges, &edge_count, exp_sid);
                            d_edges_uploaded = -1; /* force re-upload next tick */
                            pool_remove_substrate(&pool, exp_sid);
                        }
                    }
                }
            }
        }

        /* Periodic comb candidate dump */
        if ((tick % COMB_PERIOD) == 0) {
            CHECK_CUDA(cudaMemset(d_candidate_count, 0, sizeof(uint32_t)));
            candidate_pairs_kernel<<<grid, block>>>(
                pool.s[1].p.act_count,
                pool.s[1].p.ewma_delta,
                d_candidates,
                d_candidate_count,
                MAX_CANDIDATES);
            CHECK_CUDA(cudaGetLastError());

            uint32_t h_count = 0;
            CHECK_CUDA(cudaMemcpy(&h_count, d_candidate_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            if (h_count > MAX_CANDIDATES) h_count = MAX_CANDIDATES;
            PairCandidate* h_pairs = (PairCandidate*)malloc((size_t)h_count * sizeof(PairCandidate));
            if (h_pairs) {
                CHECK_CUDA(cudaMemcpy(h_pairs, d_candidates, (size_t)h_count * sizeof(PairCandidate), cudaMemcpyDeviceToHost));
                char name[128];
                snprintf(name, sizeof(name), "relations_%d.bin", tick);
                FILE* rf = fopen(name, "wb");
                if (rf) {
                    fwrite(&h_count, sizeof(uint32_t), 1, rf);
                    fwrite(h_pairs, sizeof(PairCandidate), h_count, rf);
                    fclose(rf);
                }
                free(h_pairs);
            }

            CHECK_CUDA(cudaMemcpy(nca_w, d_w, NCA_PARAMS * sizeof(float), cudaMemcpyDeviceToHost));
            char nname[128];
            snprintf(nname, sizeof(nname), "nca_weights_%d.bin", tick);
            write_nca_weights(nname, nca_w, NCA_PARAMS);
        }

        /* Stats + observability */
        CHECK_CUDA(cudaMemsetAsync(d_active,       0, sizeof(unsigned long long), stream0));
        CHECK_CUDA(cudaMemsetAsync(d_wave_energy,  0, sizeof(float), stream0));
        CHECK_CUDA(cudaMemsetAsync(d_delta_energy, 0, sizeof(float), stream0));
        CHECK_CUDA(cudaMemsetAsync(d_nca_loss,     0, sizeof(float), stream0));
        CHECK_CUDA(cudaMemsetAsync(d_nca_residual, 0, sizeof(float), stream0));

        for (int sid = 0; sid < MAX_SUBSTRATES; ++sid) {
            if (!pool.s[sid].active) continue;
            stats_kernel<<<grid, block, 0, stream0>>>(
                pool.s[sid].p.delta_mag, pool.s[sid].p.scalar_pitch,
                pool.s[sid].p.wave_curr, pool.s[sid].p.wave_pitch,
                d_active, d_wave_energy, d_delta_energy);
            /* NCA learning stats — only on non-temporary substrates */
            if (!pool.s[sid].temporary && pool.s[sid].p.output && pool.s[sid].p.target) {
                int lb = 256, lg = ((int)CELLS + lb - 1) / lb;
                nca_learn_stats_kernel<<<lg, lb, 0, stream0>>>(
                    pool.s[sid].p.output, pool.s[sid].p.target, pool.s[sid].p.residual,
                    d_nca_loss, d_nca_residual);
            }
        }

        CHECK_CUDA(cudaStreamSynchronize(stream0));
        unsigned long long h_active = 0;
        float h_we = 0.f, h_de = 0.f, h_nca_loss = 0.f, h_nca_res = 0.f;
        CHECK_CUDA(cudaMemcpy(&h_active,   d_active,       sizeof(h_active),   cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(&h_we,       d_wave_energy,  sizeof(h_we),       cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(&h_de,       d_delta_energy, sizeof(h_de),       cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(&h_nca_loss, d_nca_loss,     sizeof(h_nca_loss), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(&h_nca_res,  d_nca_residual, sizeof(h_nca_res),  cudaMemcpyDeviceToHost));
        /* Normalize by cell count */
        h_nca_loss /= (float)CELLS;
        h_nca_res  /= (float)CELLS;

        fprintf(f_act, "%d,%llu\n", tick, (unsigned long long)h_active);
        fprintf(f_energy, "%d,%.9g\n", tick, h_we);
        fprintf(f_de, "%d,%.9g\n", tick, h_de);
        /* Write NCA training record on Adam update ticks so the CSV reflects
           actual optimizer steps rather than every simulation tick. */
        if ((tick % NCA_UPDATE_PERIOD) == 0) {
            fprintf(f_nca, "%d,%d,%.9g,%.9g,%.6g\n",
                    tick, adam_t, h_nca_loss, h_nca_res, dash.grad_norm);
        }

        /* Update dashboard state */
        dash.tick          = tick;
        dash.active_cells  = h_active;
        dash.wave_energy   = h_we;
        dash.delta_energy  = h_de;
        dash.edge_count    = edge_count;
        dash.natural_count = natural_n;
        dash.nca_loss_prev = dash.nca_loss;
        dash.nca_loss      = h_nca_loss;
        dash.nca_residual  = h_nca_res;
        /* grad_norm updated only on NCA ticks — value persists in dash */
        dash.tune_c2       = tune.c2;
        dash.tune_damping  = tune.damping;
        dash.tune_drive    = tune.drive;
        dash.tune_kwtp     = tune.kwtp;
        dash.tune_fb       = tune.feedback;
        dash.tune_gravity  = tune.gravity;
        dash.tune_lr       = tune.lr;
        dash.tune_triad    = tune.triad;
        dash.paused        = tune.paused;
        dash.alerts        = 0;
        if (h_active == 0ull) dash.alerts |= 1;
        if (h_we > 1.0e10)   dash.alerts |= 2;

        /* Count snapshot PNGs */
        {
            static int snap_count = 0;
            int period = (opt.snap_every > 0) ? opt.snap_every : SNAPSHOT_PERIOD;
            if ((tick % period) == 0) snap_count++;
            dash.snapshots = snap_count;
        }

        /* Rolling ms/tick using clock() */
        {
            static clock_t last_clock = 0;
            static int      last_tick  = 0;
            clock_t now = clock();
            if (last_tick > 0 && tick > last_tick) {
                double elapsed_ms = (double)(now - last_clock) * 1000.0 / CLOCKS_PER_SEC;
                int    dtick      = tick - last_tick;
                dash.tick_ms = (float)(elapsed_ms / dtick);
            }
            last_clock = now;
            last_tick  = tick;
        }

        draw_dashboard(&dash);

        no_new_ent_ticks += 1;
        if (no_new_ent_ticks >= 100000) {
            no_new_ent_ticks = 0;
        }

        {
            int period = (opt.snap_every > 0) ? opt.snap_every : SNAPSHOT_PERIOD;
            if ((tick % period) == 0) {
                snapshot_png(&pool.s[1], (uint64_t)tick, opt.snap_dir);
                fflush(f_act); fflush(f_energy); fflush(f_de); fflush(f_nca);
            }
        }

        ticks_executed = tick;
    }

    CHECK_CUDA(cudaEventRecord(t1, stream0));
    CHECK_CUDA(cudaEventSynchronize(t1));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));
    fprintf(stderr, "ticks=%d elapsed_ms=%.3f\n", opt.ticks, ms);

    if (natural_n > 0) {
        FILE* nf = fopen("natural_entanglements.bin", "ab");
        if (nf) {
            fwrite(natural_log, sizeof(NaturalEntRecord), (size_t)natural_n, nf);
            fclose(nf);
        }
    }

    save_snapshot("snapshot_final.bin", &pool, (uint64_t)opt.ticks);

    fclose(f_act); fclose(f_energy); fclose(f_de); fclose(f_nca);

    free(h_edges);
    free(natural_log);
    free(nca_w);
    free(nca_g);

    cudaFree(d_src_views);
    cudaFree(d_dst_views);
    cudaFree(d_edges);
    cudaFree(d_candidates);
    cudaFree(d_candidate_count);
    cudaFree(d_pair_corr);
    cudaFree(d_w);
    cudaFree(d_g);
    cudaFree(d_m);
    cudaFree(d_v);
    cudaFree(d_active);
    cudaFree(d_wave_energy);
    cudaFree(d_delta_energy);
    cudaFree(d_nca_loss);
    cudaFree(d_nca_residual);
    cudaFree(d_gnorm);
    cudaFree(d_wv_lap);
    cudaFree(d_xor_buf);
    cudaFree(d_xor_count);

    for (int i = 0; i < MAX_SUBSTRATES; ++i) {
        if (pool.s[i].active) pool_remove_substrate(&pool, i);
    }

    CHECK_CUDA(cudaEventDestroy(t0));
    CHECK_CUDA(cudaEventDestroy(t1));
    CHECK_CUDA(cudaStreamDestroy(stream0));
    CHECK_CUDA(cudaStreamDestroy(stream2));

    return 0;
}
