/* Portable CPU mock-event tests. Compile with PLO_CORE_ONLY. No Windows API,
 * observer registration, DLL load, process observation or model/GPU work.
 * printf belongs to the harness and is never called by the callback core.
 */
#include "prefix_library_observer.h"
#include <stdio.h>

static PLO_CORE core;
static PLO_ROW rows[8], copied_rows[8];
static uint16_t full[PLO_FULL_CAPACITY+1], base[PLO_BASE_CAPACITY+1];
static uint32_t checks,cases,failures;
#define CHECK(v) do {++checks;if(!(v)){++failures;printf("FAIL case %u line %u\n",cases,(unsigned)__LINE__);}} while(0)
static PLO_EVENT_INPUT event(uint32_t reason) {
    PLO_EVENT_INPUT e;
    e.reason=reason;e.reserved_flags=0;e.base_address=(uintptr_t)0x12345000u;e.image_size=4096u;
    e.full_name.length_bytes=6;e.full_name.maximum_bytes=6;e.full_name.buffer=full;
    e.base_name.length_bytes=4;e.base_name.maximum_bytes=4;e.base_name.buffer=base;
    return e;
}
static void fresh(uint32_t capacity) {
    ++cases;CHECK(plo_core_init(&core,rows,capacity)==PLO_SUCCESS);
    full[0]='C';full[1]=':';full[2]='x';base[0]='x';base[1]='y';
}
static PLO_SNAPSHOT snapshot(void) {
    PLO_SNAPSHOT s;CHECK(plo_core_snapshot(&core,&s)==PLO_SUCCESS);return s;
}
static void registered(void) {CHECK(plo_core_registered(&core,0,(uintptr_t)1)==PLO_SUCCESS);}
static void stopped(void) {
    CHECK(plo_core_stop_begin(&core)==PLO_SUCCESS);
    CHECK(plo_core_unregistered(&core,0)==PLO_SUCCESS);
}

int main(void) {
    PLO_EVENT_INPUT e;PLO_SNAPSHOT s;uint32_t count,i;
    fresh(4);e=event(1);registered();CHECK(plo_core_record(&core,&e)==PLO_SUCCESS);
    e=event(2);CHECK(plo_core_record(&core,&e)==PLO_SUCCESS);stopped();
    CHECK(plo_core_drain(&core)==PLO_SUCCESS);s=snapshot();
    CHECK(s.complete==1 && s.drained==1 && s.cookie_live==0 && s.faults==0);
    CHECK(s.attempted==2 && s.reserved==2 && s.published_prefix==2 && s.inflight==0);
    CHECK(plo_core_copy(&core,0,2,copied_rows,sizeof(copied_rows),&count)==PLO_SUCCESS && count==2);
    CHECK(copied_rows[0].sequence==1 && copied_rows[1].sequence==2);
    CHECK(copied_rows[0].reason==1 && copied_rows[1].reason==2);
    CHECK(copied_rows[0].base_address==0x12345000u && copied_rows[0].image_size==4096);
    CHECK(copied_rows[0].full_units==3 && copied_rows[0].base_units==2);
    CHECK(copied_rows[0].full_name[0]=='C' && copied_rows[0].base_name[1]=='y');

    fresh(1);e=event(1);CHECK(plo_core_record(&core,&e)==PLO_SUCCESS);
    full[0]='Z';base[0]='Z';CHECK(rows[0].full_name[0]=='C' && rows[0].base_name[0]=='x');

    fresh(1);e=event(1);CHECK(plo_core_record(&core,&e)==PLO_SUCCESS);
    CHECK(plo_core_record(&core,&e)==PLO_POISONED);s=snapshot();
    CHECK((s.faults&PLO_FAULT_OVERFLOW)!=0 && s.published_prefix==1 && s.dropped==1);
    CHECK(plo_core_record(&core,&e)==PLO_POISONED);CHECK(rows[0].sequence==1);
    registered();stopped();CHECK(plo_core_drain(&core)==PLO_POISONED);CHECK(snapshot().complete==0);

    fresh(1);for(i=0;i<PLO_FULL_CAPACITY+1;++i)full[i]=(uint16_t)('a'+i%20);
    e=event(1);e.full_name.length_bytes=(uint16_t)((PLO_FULL_CAPACITY+1)*2);
    e.full_name.maximum_bytes=e.full_name.length_bytes;
    CHECK(plo_core_record(&core,&e)==PLO_POISONED);s=snapshot();
    CHECK((s.faults&PLO_FAULT_FULL_LIMIT)!=0 && rows[0].full_units==0);

    fresh(1);e=event(1);e.base_name.length_bytes=(uint16_t)((PLO_BASE_CAPACITY+1)*2);
    e.base_name.maximum_bytes=e.base_name.length_bytes;
    CHECK(plo_core_record(&core,&e)==PLO_POISONED);CHECK((snapshot().faults&PLO_FAULT_BASE_LIMIT)!=0);

    fresh(1);e=event(1);e.full_name.length_bytes=5;
    CHECK(plo_core_record(&core,&e)==PLO_POISONED);CHECK((snapshot().faults&PLO_FAULT_BAD_INPUT)!=0);

    fresh(1);e=event(1);e.full_name.maximum_bytes=2;
    CHECK(plo_core_record(&core,&e)==PLO_POISONED);CHECK(rows[0].full_units==0);

    fresh(1);e=event(1);e.full_name.buffer=0;
    CHECK(plo_core_record(&core,&e)==PLO_POISONED);CHECK(rows[0].ready==1);

    fresh(1);e=event(99);CHECK(plo_core_record(&core,&e)==PLO_POISONED);
    CHECK(rows[0].reason==99 && (snapshot().faults&PLO_FAULT_REASON)!=0);

    fresh(1);e=event(1);e.reserved_flags=1;
    CHECK(plo_core_record(&core,&e)==PLO_POISONED);CHECK((snapshot().faults&PLO_FAULT_BAD_INPUT)!=0);

    fresh(1);CHECK(plo_core_record(&core,0)==PLO_POISONED);CHECK(rows[0].sequence==1 && rows[0].reason==0);

    fresh(1);e=event(1);e.full_name.length_bytes=(uint16_t)(PLO_FULL_CAPACITY*2);
    e.full_name.maximum_bytes=e.full_name.length_bytes;e.base_name.length_bytes=(uint16_t)(PLO_BASE_CAPACITY*2);
    e.base_name.maximum_bytes=e.base_name.length_bytes;
    for(i=0;i<PLO_FULL_CAPACITY;++i)full[i]=(uint16_t)(0x800+i);
    for(i=0;i<PLO_BASE_CAPACITY;++i)base[i]=(uint16_t)(0xD800+i); /* preserve raw UTF16 units */
    CHECK(plo_core_record(&core,&e)==PLO_SUCCESS);
    CHECK(rows[0].full_units==PLO_FULL_CAPACITY && rows[0].full_name[PLO_FULL_CAPACITY-1]==full[PLO_FULL_CAPACITY-1]);
    CHECK(rows[0].base_units==PLO_BASE_CAPACITY && rows[0].base_name[PLO_BASE_CAPACITY-1]==base[PLO_BASE_CAPACITY-1]);

    fresh(1);CHECK(plo_core_registered(&core,-7,0)==PLO_POISONED);
    CHECK(plo_core_drain(&core)==PLO_POISONED);s=snapshot();CHECK(s.cookie_live==0 && s.register_status==-7 && s.complete==0);

    fresh(1);CHECK(plo_core_registered(&core,0,0)==PLO_POISONED);CHECK((snapshot().faults&PLO_FAULT_REGISTER)!=0);

    fresh(1);registered();CHECK(plo_core_stop_begin(&core)==PLO_SUCCESS);
    CHECK(plo_core_unregistered(&core,-9)==PLO_POISONED);s=snapshot();
    CHECK(s.cookie_live==1 && s.phase==PLO_STOPPING && s.unregister_status==-9);
    CHECK(plo_core_drain(&core)==PLO_BAD_STATE);
    CHECK(plo_core_unregistered(&core,0)==PLO_SUCCESS);CHECK(plo_core_drain(&core)==PLO_POISONED);
    CHECK(snapshot().complete==0); /* retry cleans but cannot erase prior loss */

    fresh(1);registered();core.inflight=1;stopped();CHECK(plo_core_drain(&core)==PLO_BUSY);
    core.inflight=0;CHECK(plo_core_drain(&core)==PLO_SUCCESS);CHECK(snapshot().complete==1);

    fresh(2);registered();core.reserved=2;rows[0].ready=0;rows[1].ready=1;stopped();
    CHECK(snapshot().published_prefix==0);CHECK(plo_core_drain(&core)==PLO_BUSY);
    rows[0].ready=1;CHECK(plo_core_drain(&core)==PLO_SUCCESS);CHECK(snapshot().published_prefix==2);

    fresh(1);e=event(1);registered();stopped();CHECK(plo_core_drain(&core)==PLO_SUCCESS);
    CHECK(plo_core_record(&core,&e)==PLO_BAD_STATE);s=snapshot();
    CHECK((s.faults&PLO_FAULT_LATE_CALLBACK)!=0 && s.complete==0 && s.dropped==1);

    fresh(1);e=event(1);CHECK(plo_core_record(&core,&e)==PLO_SUCCESS);
    CHECK(plo_core_copy(&core,2,1,copied_rows,sizeof(copied_rows),&count)==PLO_BAD_ARGUMENT);
    CHECK(plo_core_copy(&core,0,1,copied_rows,1,&count)==PLO_BAD_ARGUMENT);
    CHECK(plo_core_copy(&core,1,1,copied_rows,sizeof(copied_rows),&count)==PLO_SUCCESS && count==0);

    fresh(1);e=event(1);core.reserved=(PLO_ATOMIC)UINT32_MAX;
    CHECK(plo_core_record(&core,&e)==PLO_POISONED);s=snapshot();
    CHECK((s.faults&PLO_FAULT_COUNTER_WRAP)!=0 && s.published_prefix==0);
    CHECK(plo_core_record(&core,&e)==PLO_POISONED && rows[0].ready==0);

    CHECK(sizeof(PLO_ROW)==2600 && sizeof(PLO_SNAPSHOT)==72);
    CHECK(offsetof(PLO_ROW,full_name)==40 && offsetof(PLO_ROW,base_name)==2088);
    printf("%s portable CPU core: %u cases, %u checks, %u failures; no Windows registration/DLL load\n",
        failures==0?"PASS":"FAIL",cases,checks,failures);
    return failures==0?0:1;
}
