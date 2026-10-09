"""Bounded original Python traceback metadata, evidence only; no file/OS APIs.

Only builtin traceback/frame/code filename, function and line descriptors are
accessed. No exception strings, repr, args, locals, cause/context or tensor data.
Supplied pins are joined literally; this helper never verifies their file bytes.
"""
import hashlib,json
DEFAULT_LIMITS={'maximum_frames':16,'maximum_filename_chars':512,'maximum_function_chars':128,'maximum_serialized_bytes':16384}
KIND='BOUNDED_ORIGINAL_EXCEPTION_PYTHON_FRAMES_V1'
_TRACEBACK=BaseException.__dict__['__traceback__']
def require(value,message):
    if not value:raise ValueError(message)
def canonical(value):return json.dumps(value,ensure_ascii=True,sort_keys=True,separators=(',',':'),allow_nan=False)
def seal(value):
    value=dict(value);value['sha256']=hashlib.sha256(canonical(value).encode('ascii')).hexdigest();return value
def _pins(source,runtime):
    require(type(source) is dict and type(runtime) is dict,'plain supplied registered pin dictionaries')
    require(len(source)+len(runtime)<=4096,'bounded supplied registered pin count')
    joined={}
    for domain,entries in (('source',source),('runtime',runtime)):
        for filename,pin in entries.items():
            require(type(filename) is str and 0<len(filename)<=4096 and type(pin) is str and len(pin)==64
                and all(c in '0123456789abcdef' for c in pin),'typed supplied literal filename/SHA256 pin')
            if filename in joined:require(joined[filename][0]==pin,'overlap registered pin conflict')
            else:joined[filename]=[pin,[]]
            joined[filename][1].append(domain)
    return joined
def _limits(value):
    result=dict(DEFAULT_LIMITS) if value is None else value
    require(type(result) is dict and set(result)==set(DEFAULT_LIMITS)
        and all(type(n) is int for n in result.values()),'exact integer diagnostic limit fields')
    require(1<=result['maximum_frames']<=16 and 1<=result['maximum_filename_chars']<=512
        and 1<=result['maximum_function_chars']<=128 and 1024<=result['maximum_serialized_bytes']<=16384,'fixed conservative diagnostic bounds')
    return dict(result)
def capture_exception_frames(error,source_pins,runtime_pins,*,limits=None):
    require(isinstance(error,BaseException),'original exception object required')
    bound=_limits(limits);registered=_pins(source_pins,runtime_pins)
    # Direct builtin descriptor bypasses subclass __getattribute__/properties.
    tb=_TRACEBACK.__get__(error,BaseException)
    packet={'schema_version':1,'kind':KIND,'scope':'PYTHON_REPORTING_BOUNDARY_EVIDENCE_ONLY_UNQUALIFIED',
        'status':'NO_TRACEBACK' if tb is None else 'CAPTURED','frames':[],'frames_retained':0,
        'truncated':False,'truncation_reason':None,'limits':bound,'pins_are_supplied_not_byte_verified':True,
        'exception_preserved':True,'locals_args_repr_contexts_tensors_read':False,
        'cause_established':False,'native_qualification_established':False,'scoring_authorized':False}
    while tb is not None and len(packet['frames'])<bound['maximum_frames']:
        code=tb.tb_frame.f_code;filename=code.co_filename;function=code.co_name;line=tb.tb_lineno
        # CodeType.replace can retain str subclasses. Never invoke their len/
        # hash/equality hooks while processing diagnostic metadata.
        names_plain=type(filename) is str and type(function) is str
        pin=registered.get(filename) if names_plain and len(filename)<=4096 else None
        if pin is None:
            frame={'filename':'<unregistered>'[:bound['maximum_filename_chars']],'function':'<unregistered>'[:bound['maximum_function_chars']],'line':line,
                'registration_domains':[],'registered_sha256':None,'registration_status':'unregistered_redacted'}
        else:
            names_ok=len(filename)<=bound['maximum_filename_chars'] and len(function)<=bound['maximum_function_chars']
            frame={'filename':filename if len(filename)<=bound['maximum_filename_chars'] else '<name-limit>'[:bound['maximum_filename_chars']],
                'function':function if len(function)<=bound['maximum_function_chars'] else '<name-limit>'[:bound['maximum_function_chars']],
                'line':line,'registration_domains':list(pin[1]),'registered_sha256':pin[0],
                'registration_status':'registered' if names_ok else 'name_limit_redacted'}
        if type(line) is not int or not(1<=line<=2147483647):frame['line']=None;frame['registration_status']='invalid_line_redacted'
        packet['frames'].append(frame);packet['frames_retained']=len(packet['frames'])
        candidate=seal(packet)
        if len(canonical(candidate).encode('ascii'))>bound['maximum_serialized_bytes']:
            packet['frames'].pop();packet['frames_retained']=len(packet['frames']);packet['status']='TRUNCATED_SIZE_LIMIT'
            packet['truncated']=True;packet['truncation_reason']='serialized_size';break
        tb=tb.tb_next
    if tb is not None and not packet['truncated']:
        packet['status']='TRUNCATED_FRAME_LIMIT';packet['truncated']=True;packet['truncation_reason']='frame_limit'
    result=seal(packet)
    # Conservative small fallback is metadata only, never the primary exception.
    if len(canonical(result).encode('ascii'))>bound['maximum_serialized_bytes']:
        packet.update(status='TRUNCATED_SIZE_LIMIT',frames=[],frames_retained=0,truncated=True,truncation_reason='serialized_size')
        result=seal(packet)
    require(len(canonical(result).encode('ascii'))<=bound['maximum_serialized_bytes'],'bounded diagnostic packet serialization')
    return result
