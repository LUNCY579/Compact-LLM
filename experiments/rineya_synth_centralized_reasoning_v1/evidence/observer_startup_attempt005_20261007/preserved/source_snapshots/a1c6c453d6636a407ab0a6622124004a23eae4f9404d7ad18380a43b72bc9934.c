/* Microsoft source references:
 * https://learn.microsoft.com/en-us/windows/win32/devnotes/ldrdllnotification
 * https://learn.microsoft.com/en-us/windows/win32/devnotes/ldrregisterdllnotification
 * https://learn.microsoft.com/en-us/windows/win32/devnotes/ldrunregisterdllnotification
 * https://learn.microsoft.com/en-us/windows/win32/api/ntdef/ns-ntdef-_unicode_string
 * Notification callback may call only this module. No allocation, logging,
 * lock/wait, CRT copy, clock, string conversion or Python occurs on that path.
 * Atomic intrinsics and volatile counted copies must be confirmed in the
 * exact future DLL disassembly; this source/portable test is not that proof.
 */
#include "prefix_library_observer.h"

#if defined(_MSC_VER)
#if !defined(_M_X64)
#error "This candidate's native MSVC atomic/ABI route is x64 only"
#endif
#include <intrin.h>
#pragma intrinsic(_InterlockedCompareExchange)
#pragma intrinsic(_InterlockedExchange)
#pragma intrinsic(_InterlockedExchangeAdd)
#pragma intrinsic(_InterlockedOr)
static uint32_t a_load(PLO_ATOMIC *p) { return (uint32_t)_InterlockedCompareExchange(p,0,0); }
static void a_store(PLO_ATOMIC *p,uint32_t v) { (void)_InterlockedExchange(p,(long)v); }
static uint32_t a_add(PLO_ATOMIC *p,uint32_t v) { return (uint32_t)_InterlockedExchangeAdd(p,(long)v); }
static void a_or(PLO_ATOMIC *p,uint32_t v) { (void)_InterlockedOr(p,(long)v); }
#else
static uint32_t a_load(PLO_ATOMIC *p) { return atomic_load_explicit(p,memory_order_seq_cst); }
static void a_store(PLO_ATOMIC *p,uint32_t v) { atomic_store_explicit(p,v,memory_order_seq_cst); }
static uint32_t a_add(PLO_ATOMIC *p,uint32_t v) { return atomic_fetch_add_explicit(p,v,memory_order_seq_cst); }
static void a_or(PLO_ATOMIC *p,uint32_t v) { (void)atomic_fetch_or_explicit(p,v,memory_order_seq_cst); }
#endif

typedef char row_size_is_2600[(sizeof(PLO_ROW)==2600u)?1:-1];
typedef char snapshot_size_is_72[(sizeof(PLO_SNAPSHOT)==72u)?1:-1];
typedef char full_offset_is_40[(offsetof(PLO_ROW,full_name)==40u)?1:-1];
typedef char base_offset_is_2088[(offsetof(PLO_ROW,base_name)==2088u)?1:-1];
typedef char unicode_x64_size_is_16[(sizeof(PLO_UNICODE_VIEW)==16u)?1:-1];
typedef char unicode_buffer_offset_is_8[(offsetof(PLO_UNICODE_VIEW,buffer)==8u)?1:-1];
typedef char ldr_x64_size_is_40[(sizeof(PLO_LDR_DATA)==40u)?1:-1];
typedef char ldr_base_offset_is_24[(offsetof(PLO_LDR_DATA,dll_base)==24u)?1:-1];
typedef char ldr_image_offset_is_32[(offsetof(PLO_LDR_DATA,image_size)==32u)?1:-1];

static void fault(PLO_CORE *c,uint32_t value) { a_or(&c->faults,value); }
static void dropped(PLO_CORE *c) {
    if(a_add(&c->dropped,1u)==UINT32_MAX) fault(c,PLO_FAULT_COUNTER_WRAP);
}
static uint32_t prefix(PLO_CORE *c) {
    uint32_t n=0u, limit=a_load(&c->reserved);
    if(limit>c->capacity) limit=c->capacity;
    while(n<limit && a_load((PLO_ATOMIC *)&c->rows[n].ready)==1u) ++n;
    return n;
}
static uint32_t copy_name(volatile uint16_t *out,const PLO_UNICODE_VIEW *view,
    uint32_t capacity,uint32_t limit_fault,uint32_t *units) {
    uint32_t i,n;
    *units=0u;
    if(view==0 || view->buffer==0 || view->length_bytes==0u ||
       (view->length_bytes&1u)!=0u || view->length_bytes>view->maximum_bytes)
        return PLO_FAULT_BAD_INPUT;
    n=(uint32_t)view->length_bytes/2u;
    if(n>capacity) return limit_fault; /* NEVER copy a truncated prefix. */
    for(i=0u;i<n;++i) out[i]=((const volatile uint16_t *)view->buffer)[i];
    *units=n;
    return 0u;
}

int32_t plo_core_init(PLO_CORE *c,PLO_ROW *rows,uint32_t capacity) {
    uint32_t i,j;volatile unsigned char *bytes;
    if(c==0 || rows==0 || capacity==0u || capacity>PLO_ROW_CAPACITY) return PLO_BAD_ARGUMENT;
    c->rows=rows;c->capacity=capacity;
    a_store(&c->phase,PLO_ARMED);a_store(&c->faults,0u);a_store(&c->attempted,0u);
    a_store(&c->reserved,0u);a_store(&c->inflight,0u);a_store(&c->cookie_live,0u);
    a_store(&c->register_status,(uint32_t)PLO_STATUS_UNSET);
    a_store(&c->unregister_status,(uint32_t)PLO_STATUS_UNSET);
    a_store(&c->drained,0u);a_store(&c->dropped,0u);
    for(i=0u;i<capacity;++i) {
        bytes=(volatile unsigned char *)&rows[i];
        for(j=0u;j<(uint32_t)sizeof(PLO_ROW);++j)bytes[j]=0u;
        a_store(&rows[i].ready,0u);
    }
    return PLO_SUCCESS;
}

/* Callback core: finite scalar work + bounded volatile copies only. */
int32_t plo_core_record(PLO_CORE *c,const PLO_EVENT_INPUT *event) {
    uint32_t p,index,flags=0u;PLO_ROW *row;
    if(c==0 || c->rows==0) return PLO_BAD_ARGUMENT;
    if(a_add(&c->attempted,1u)==UINT32_MAX) fault(c,PLO_FAULT_COUNTER_WRAP);
    if(a_add(&c->inflight,1u)==UINT32_MAX) fault(c,PLO_FAULT_COUNTER_WRAP);
    p=a_load(&c->phase);
    if(p!=PLO_ARMED && p!=PLO_REGISTERED && p!=PLO_STOPPING) {
        fault(c,PLO_FAULT_LATE_CALLBACK);dropped(c);
        (void)a_add(&c->inflight,UINT32_MAX);return PLO_BAD_STATE;
    }
    if((a_load(&c->faults)&(PLO_FAULT_OVERFLOW|PLO_FAULT_COUNTER_WRAP))!=0u) {
        dropped(c);(void)a_add(&c->inflight,UINT32_MAX);return PLO_POISONED;
    }
    index=a_add(&c->reserved,1u);
    if(index>=c->capacity) {
        if(index==UINT32_MAX)fault(c,PLO_FAULT_COUNTER_WRAP);
        fault(c,PLO_FAULT_OVERFLOW);dropped(c);
        (void)a_add(&c->inflight,UINT32_MAX);return PLO_POISONED;
    }
    row=&c->rows[index];row->sequence=(uint64_t)index+1u;
    row->reason=event==0?0u:event->reason;
    row->image_size=event==0?0u:event->image_size;
    row->base_address=event==0?0u:(uint64_t)event->base_address;
    row->full_units=0u;row->base_units=0u;
    if(event==0) flags=PLO_FAULT_BAD_INPUT;
    else {
        if(event->reason!=1u && event->reason!=2u) flags|=PLO_FAULT_REASON;
        if(event->reserved_flags!=0u || event->base_address==0u) flags|=PLO_FAULT_BAD_INPUT;
        flags|=copy_name(row->full_name,&event->full_name,PLO_FULL_CAPACITY,PLO_FAULT_FULL_LIMIT,&row->full_units);
        flags|=copy_name(row->base_name,&event->base_name,PLO_BASE_CAPACITY,PLO_FAULT_BASE_LIMIT,&row->base_units);
    }
    row->flags=flags;if(flags!=0u) fault(c,flags);
    a_store((PLO_ATOMIC *)&row->ready,1u); /* Publish after all scalar/name writes. */
    (void)a_add(&c->inflight,UINT32_MAX);
    return flags==0u?PLO_SUCCESS:PLO_POISONED;
}

int32_t plo_core_registered(PLO_CORE *c,int32_t status,uintptr_t cookie) {
    if(c==0 || a_load(&c->phase)!=PLO_ARMED) return PLO_BAD_STATE;
    a_store(&c->register_status,(uint32_t)status);
    if(status!=0 || cookie==0u) {
        fault(c,PLO_FAULT_REGISTER);a_store(&c->cookie_live,cookie!=0u);
        a_store(&c->phase,cookie==0u?PLO_START_FAILED:PLO_STOPPING);return PLO_POISONED;
    }
    a_store(&c->cookie_live,1u);a_store(&c->phase,PLO_REGISTERED);return PLO_SUCCESS;
}
int32_t plo_core_stop_begin(PLO_CORE *c) {
    uint32_t p;if(c==0) return PLO_BAD_ARGUMENT;p=a_load(&c->phase);
    if(p!=PLO_REGISTERED && p!=PLO_STOPPING) return PLO_BAD_STATE;
    a_store(&c->phase,PLO_STOPPING);return PLO_SUCCESS;
}
int32_t plo_core_unregistered(PLO_CORE *c,int32_t status) {
    if(c==0 || a_load(&c->phase)!=PLO_STOPPING) return PLO_BAD_STATE;
    a_store(&c->unregister_status,(uint32_t)status);
    if(status!=0) {fault(c,PLO_FAULT_UNREGISTER);return PLO_POISONED;}
    a_store(&c->cookie_live,0u);a_store(&c->phase,PLO_DRAINING);return PLO_SUCCESS;
}
int32_t plo_core_drain(PLO_CORE *c) {
    uint32_t count,p;if(c==0) return PLO_BAD_ARGUMENT;p=a_load(&c->phase);
    if(p!=PLO_DRAINING && p!=PLO_START_FAILED) return PLO_BAD_STATE;
    if(a_load(&c->cookie_live)!=0u || a_load(&c->inflight)!=0u) return PLO_BUSY;
    count=a_load(&c->reserved);
    if(prefix(c)!=(count<c->capacity?count:c->capacity)) return PLO_BUSY;
    a_store(&c->drained,1u);a_store(&c->phase,PLO_SEALED);
    return a_load(&c->faults)==0u?PLO_SUCCESS:PLO_POISONED;
}
int32_t plo_core_snapshot(PLO_CORE *c,PLO_SNAPSHOT *out) {
    if(c==0 || out==0 || c->rows==0) return PLO_BAD_ARGUMENT;
    out->abi_version=PLO_ABI_VERSION;out->row_bytes=(uint32_t)sizeof(PLO_ROW);
    out->snapshot_bytes=(uint32_t)sizeof(PLO_SNAPSHOT);out->row_capacity=c->capacity;
    out->full_capacity=PLO_FULL_CAPACITY;out->base_capacity=PLO_BASE_CAPACITY;
    out->phase=a_load(&c->phase);out->faults=a_load(&c->faults);
    out->attempted=a_load(&c->attempted);out->reserved=a_load(&c->reserved);
    out->published_prefix=prefix(c);out->inflight=a_load(&c->inflight);
    out->cookie_live=a_load(&c->cookie_live);out->register_status=(int32_t)a_load(&c->register_status);
    out->unregister_status=(int32_t)a_load(&c->unregister_status);out->drained=a_load(&c->drained);
    out->dropped=a_load(&c->dropped);
    out->complete=(out->phase==PLO_SEALED && out->faults==0u && out->inflight==0u &&
        out->cookie_live==0u && out->drained==1u && out->register_status==0 &&
        out->unregister_status==0 && out->reserved==out->published_prefix);
    return PLO_SUCCESS;
}
int32_t plo_core_copy(PLO_CORE *c,uint32_t first,uint32_t requested,PLO_ROW *out,
    uint32_t out_bytes,uint32_t *copied) {
    uint32_t n,i,j,available;volatile unsigned char *dst;const volatile unsigned char *src;
    if(c==0 || out==0 || copied==0 || requested>PLO_ROW_CAPACITY ||
       requested>out_bytes/(uint32_t)sizeof(PLO_ROW)) return PLO_BAD_ARGUMENT;
    *copied=0u;available=prefix(c);if(first>available) return PLO_BAD_ARGUMENT;
    n=available-first;if(n>requested)n=requested;
    for(i=0u;i<n;++i) {
        dst=(volatile unsigned char *)&out[i];src=(const volatile unsigned char *)&c->rows[first+i];
        for(j=0u;j<(uint32_t)sizeof(PLO_ROW);++j)dst[j]=src[j];
    }
    *copied=n;return PLO_SUCCESS;
}

#if !defined(PLO_CORE_ONLY)
static PLO_CORE global_core;
static PLO_ROW global_rows[PLO_ROW_CAPACITY];
static PLO_UNREGISTER unregister_function;
static void *registration_cookie;
static uint32_t ever_started;

static void PLO_NTAPI notification(uint32_t reason,const PLO_LDR_DATA *data,void *context) {
    PLO_EVENT_INPUT event;PLO_CORE *core=(PLO_CORE *)context;
    if(data==0) {(void)plo_core_record(core,0);return;}
    event.reason=reason;event.reserved_flags=data->flags;event.base_address=(uintptr_t)data->dll_base;
    event.image_size=data->image_size;
    event.full_name.length_bytes=data->full_name==0?0u:data->full_name->length_bytes;
    event.full_name.maximum_bytes=data->full_name==0?0u:data->full_name->maximum_bytes;
    event.full_name.buffer=data->full_name==0?0:data->full_name->buffer;
    event.base_name.length_bytes=data->base_name==0?0u:data->base_name->length_bytes;
    event.base_name.maximum_bytes=data->base_name==0?0u:data->base_name->maximum_bytes;
    event.base_name.buffer=data->base_name==0?0:data->base_name->buffer;
    (void)plo_core_record(core,&event);
}
uint32_t plobs_abi_version(void) {return PLO_ABI_VERSION;}
int32_t plobs_start(PLO_REGISTER reg,PLO_UNREGISTER unreg) {
    int32_t status,result;
    if(reg==0 || unreg==0) return PLO_BAD_ARGUMENT;
    if(ever_started!=0u) {fault(&global_core,PLO_FAULT_RESTART);return PLO_BAD_STATE;}
    ever_started=1u;(void)plo_core_init(&global_core,global_rows,PLO_ROW_CAPACITY);
    unregister_function=unreg;registration_cookie=0;
    status=reg(0u,notification,&global_core,&registration_cookie);
    result=plo_core_registered(&global_core,status,(uintptr_t)registration_cookie);
    return result;
}
int32_t plobs_snapshot(PLO_SNAPSHOT *out,uint32_t bytes) {
    if(bytes!=(uint32_t)sizeof(PLO_SNAPSHOT))return PLO_BAD_ARGUMENT;
    return plo_core_snapshot(&global_core,out);
}
int32_t plobs_copy_rows(uint32_t first,uint32_t requested,PLO_ROW *out,uint32_t bytes,uint32_t *copied) {
    return plo_core_copy(&global_core,first,requested,out,bytes,copied);
}
int32_t plobs_stop(PLO_SNAPSHOT *out,uint32_t bytes) {
    int32_t result,status;uint32_t phase;
    if(out==0 || bytes!=(uint32_t)sizeof(PLO_SNAPSHOT))return PLO_BAD_ARGUMENT;
    phase=a_load(&global_core.phase);
    if(phase==PLO_SEALED) {
        (void)plo_core_snapshot(&global_core,out);return out->complete?PLO_SUCCESS:PLO_POISONED;
    }
    if(phase==PLO_DRAINING || phase==PLO_START_FAILED) {
        result=plo_core_drain(&global_core);(void)plo_core_snapshot(&global_core,out);return result;
    }
    result=plo_core_stop_begin(&global_core);
    if(result==PLO_SUCCESS) {
        status=unregister_function(registration_cookie);
        result=plo_core_unregistered(&global_core,status);
        if(status==0)registration_cookie=0;
    } else if(a_load(&global_core.phase)!=PLO_START_FAILED)return result;
    if(a_load(&global_core.phase)==PLO_DRAINING || a_load(&global_core.phase)==PLO_START_FAILED)
        result=plo_core_drain(&global_core);
    (void)plo_core_snapshot(&global_core,out);return result;
}
#endif
