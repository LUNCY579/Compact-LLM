#ifndef PREFIX_LIBRARY_OBSERVER_H
#define PREFIX_LIBRARY_OBSERVER_H

/* Source-only observer candidate. No native qualification or execution permit.
 * Public ABI 1 is x64, little-endian, naturally aligned (no packing pragma).
 * Strings are counted UTF-16 units, never normalized/truncated/null-scanned.
 * Future binary/toolchain/options/callback code and unregister barrier require
 * separate qualification before any Windows registration or DLL load.
 */
#include <stdint.h>
#include <stddef.h>

#define PLO_ABI_VERSION 1u
#define PLO_ROW_CAPACITY 4096u
#define PLO_FULL_CAPACITY 1024u
#define PLO_BASE_CAPACITY 256u
#define PLO_SUCCESS 0
#define PLO_BAD_ARGUMENT (-1)
#define PLO_BAD_STATE (-2)
#define PLO_BUSY (-3)
#define PLO_POISONED (-4)
#define PLO_STATUS_UNSET ((int32_t)0x80000000u)

enum plo_phase { PLO_FRESH=0, PLO_ARMED=1, PLO_REGISTERED=2,
    PLO_STOPPING=3, PLO_DRAINING=4, PLO_SEALED=5, PLO_START_FAILED=6 };
enum plo_fault {
    PLO_FAULT_OVERFLOW=1u, PLO_FAULT_BAD_INPUT=2u, PLO_FAULT_FULL_LIMIT=4u,
    PLO_FAULT_BASE_LIMIT=8u, PLO_FAULT_REASON=16u, PLO_FAULT_REGISTER=32u,
    PLO_FAULT_UNREGISTER=64u, PLO_FAULT_LATE_CALLBACK=128u,
    PLO_FAULT_COUNTER_WRAP=256u, PLO_FAULT_RESTART=512u,
    PLO_FAULT_INFLIGHT=1024u
};

#if defined(_MSC_VER)
typedef volatile long PLO_ATOMIC;
#else
#include <stdatomic.h>
#if ATOMIC_INT_LOCK_FREE != 2
#error "Observer requires always-lock-free 32-bit atomic instructions"
#endif
typedef _Atomic uint32_t PLO_ATOMIC;
#endif

typedef struct plo_row {
    uint64_t sequence;
    uint64_t base_address;
    uint32_t reason;
    uint32_t image_size;
    uint32_t full_units;
    uint32_t base_units;
    uint32_t flags;
    PLO_ATOMIC ready;
    uint16_t full_name[PLO_FULL_CAPACITY];
    uint16_t base_name[PLO_BASE_CAPACITY];
} PLO_ROW;

typedef struct plo_snapshot {
    uint32_t abi_version, row_bytes, snapshot_bytes;
    uint32_t row_capacity, full_capacity, base_capacity;
    uint32_t phase, faults, attempted, reserved, published_prefix, inflight;
    uint32_t cookie_live;
    int32_t register_status, unregister_status;
    uint32_t drained, complete, dropped;
} PLO_SNAPSHOT;

typedef struct plo_unicode_view {
    uint16_t length_bytes, maximum_bytes;
    const uint16_t *buffer;
} PLO_UNICODE_VIEW;
typedef struct plo_event_input {
    uint32_t reason, reserved_flags;
    uintptr_t base_address;
    uint32_t image_size;
    PLO_UNICODE_VIEW full_name, base_name;
} PLO_EVENT_INPUT;

/* Fixed-width atomic storage: MSVC x64 intrinsic or lock-free C11 uint32.
 * Core state/rows are owned and initialized outside the callback and remain
 * alive through unregister, drain and process/DLL lifetime qualification.
 */
typedef struct plo_core {
    PLO_ROW *rows;
    uint32_t capacity;
    PLO_ATOMIC phase, faults, attempted, reserved, inflight, cookie_live;
    PLO_ATOMIC register_status, unregister_status, drained, dropped;
} PLO_CORE;

int32_t plo_core_init(PLO_CORE *, PLO_ROW *, uint32_t capacity);
int32_t plo_core_record(PLO_CORE *, const PLO_EVENT_INPUT *);
int32_t plo_core_registered(PLO_CORE *, int32_t status, uintptr_t cookie);
int32_t plo_core_stop_begin(PLO_CORE *);
int32_t plo_core_unregistered(PLO_CORE *, int32_t status);
int32_t plo_core_snapshot(PLO_CORE *, PLO_SNAPSHOT *);
int32_t plo_core_copy(PLO_CORE *, uint32_t first, uint32_t requested,
    PLO_ROW *out, uint32_t out_bytes, uint32_t *copied);
int32_t plo_core_drain(PLO_CORE *);

#if defined(_MSC_VER)
#define PLO_NTAPI __stdcall
#if defined(PLO_BUILD_DLL)
#define PLO_EXPORT __declspec(dllexport)
#else
#define PLO_EXPORT
#endif
#else
#define PLO_NTAPI
#define PLO_EXPORT
#endif

typedef struct plo_ldr_data {
    uint32_t flags;
    const PLO_UNICODE_VIEW *full_name, *base_name;
    void *dll_base;
    uint32_t image_size;
} PLO_LDR_DATA;
typedef void (PLO_NTAPI *PLO_NOTIFY)(uint32_t, const PLO_LDR_DATA *, void *);
typedef int32_t (PLO_NTAPI *PLO_REGISTER)(uint32_t, PLO_NOTIFY, void *, void **);
typedef int32_t (PLO_NTAPI *PLO_UNREGISTER)(void *);

/* Called outside loader notifications. Function addresses are resolved and
 * source/binary-qualified by the caller; this module never resolves DLLs.
 */
PLO_EXPORT uint32_t plobs_abi_version(void);
PLO_EXPORT int32_t plobs_start(PLO_REGISTER, PLO_UNREGISTER);
PLO_EXPORT int32_t plobs_snapshot(PLO_SNAPSHOT *, uint32_t out_bytes);
PLO_EXPORT int32_t plobs_copy_rows(uint32_t first, uint32_t requested,
    PLO_ROW *, uint32_t out_bytes, uint32_t *copied);
PLO_EXPORT int32_t plobs_stop(PLO_SNAPSHOT *, uint32_t out_bytes);

#endif
