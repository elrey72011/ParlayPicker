"""Test-only suspended launch into an unnamed, non-breakaway Windows job."""
import ctypes
from ctypes import wintypes as W
import os
import subprocess
import time


class WindowsChild:
    def __init__(self, command):
        import _winapi
        import msvcrt
        self.api=_winapi;self.command=command;self.process=self.job=None
        self.stdout=self.stderr=None;readers=[];writers=[];thread=None
        self.kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        k=self.kernel
        signatures={
            'CreateJobObjectW':([ctypes.c_void_p,W.LPCWSTR],W.HANDLE),
            'SetInformationJobObject':([W.HANDLE,ctypes.c_int,ctypes.c_void_p,W.DWORD],W.BOOL),
            'AssignProcessToJobObject':([W.HANDLE,W.HANDLE],W.BOOL),
            'TerminateJobObject':([W.HANDLE,W.UINT],W.BOOL),
            'QueryInformationJobObject':([W.HANDLE,ctypes.c_int,ctypes.c_void_p,W.DWORD,ctypes.c_void_p],W.BOOL),
            'ResumeThread':([W.HANDLE],W.DWORD),
            'CloseHandle':([W.HANDLE],W.BOOL),
            'OpenProcess':([W.DWORD,W.BOOL,W.DWORD],W.HANDLE),
        }
        for name,(args,result) in signatures.items():
            getattr(k,name).argtypes=args;getattr(k,name).restype=result
        class Basic(ctypes.Structure):
            _fields_=[('user',ctypes.c_longlong),('job_user',ctypes.c_longlong),('flags',W.DWORD),
                ('minimum',ctypes.c_size_t),('maximum',ctypes.c_size_t),('active',W.DWORD),
                ('affinity',ctypes.c_size_t),('priority',W.DWORD),('scheduling',W.DWORD)]
        class IO(ctypes.Structure):
            _fields_=[(n,ctypes.c_ulonglong) for n in ('read','write','other','read_bytes','write_bytes','other_bytes')]
        class Limits(ctypes.Structure):
            _fields_=[('basic',Basic),('io',IO),('process_memory',ctypes.c_size_t),
                ('job_memory',ctypes.c_size_t),('peak_process',ctypes.c_size_t),('peak_job',ctypes.c_size_t)]
        try:
            self.job=k.CreateJobObjectW(None,None)
            if not self.job:raise ctypes.WinError(ctypes.get_last_error())
            limits=Limits();limits.basic.flags=0x2000  # KILL_ON_JOB_CLOSE, no breakaway.
            if not k.SetInformationJobObject(self.job,9,ctypes.byref(limits),ctypes.sizeof(limits)):
                raise ctypes.WinError(ctypes.get_last_error())
            for _ in range(2):
                r,w=os.pipe();readers.append(r);writers.append(w);os.set_inheritable(w,True)
            with open(os.devnull,'rb') as stdin:
                handles=[msvcrt.get_osfhandle(stdin.fileno()),*[msvcrt.get_osfhandle(w) for w in writers]]
                os.set_handle_inheritable(handles[0],True)
                si=subprocess.STARTUPINFO();si.dwFlags|=_winapi.STARTF_USESTDHANDLES
                si.hStdInput,si.hStdOutput,si.hStdError=handles;si.lpAttributeList={'handle_list':handles}
                self.process,thread,self.pid,_=_winapi.CreateProcess(command[0],subprocess.list2cmdline(command),
                    None,None,True,subprocess.CREATE_NO_WINDOW|4,None,None,si)  # CREATE_SUSPENDED.
            if not k.AssignProcessToJobObject(self.job,self.process):raise ctypes.WinError(ctypes.get_last_error())
            if k.ResumeThread(thread)==0xffffffff:raise ctypes.WinError(ctypes.get_last_error())
            self.stdout=os.fdopen(readers[0],'rb',buffering=0);readers.pop(0)
            self.stderr=os.fdopen(readers[0],'rb',buffering=0);readers.pop(0)
        except BaseException:
            if self.process:
                _winapi.TerminateProcess(self.process,1);_winapi.WaitForSingleObject(self.process,5000)
            self.close()
            raise
        finally:
            if thread:_winapi.CloseHandle(thread)
            for fd in readers+writers:os.close(fd)

    def wait(self, timeout):
        if self.api.WaitForSingleObject(self.process,max(0,int(timeout*1000)))==258:
            raise subprocess.TimeoutExpired(self.command,timeout)
        return self.api.GetExitCodeProcess(self.process)

    def members(self):
        class Accounting(ctypes.Structure):
            _fields_=[(n,ctypes.c_longlong) for n in ('user','kernel','period_user','period_kernel')]+[(n,W.DWORD) for n in ('faults','total','active','terminated')]
        value=Accounting()
        if not self.kernel.QueryInformationJobObject(self.job,1,ctypes.byref(value),ctypes.sizeof(value),None):
            raise ctypes.WinError(ctypes.get_last_error())
        return value.active

    def cleanup(self):
        before=self.members()
        # Active-process accounting can reach zero before descendant handles signal.
        # Retain synchronization handles while identities still belong to this job.
        buffer=ctypes.create_string_buffer(8+512*ctypes.sizeof(ctypes.c_size_t));held=[]
        try:
            if not self.kernel.QueryInformationJobObject(self.job,3,buffer,len(buffer),None):
                raise ctypes.WinError(ctypes.get_last_error())
            count=W.DWORD.from_buffer(buffer,4).value
            if count>512:raise RuntimeError('OWNED_JOB_PROCESS_LIMIT')
            ids=(ctypes.c_size_t*count).from_buffer(buffer,8)
            for pid in ids:
                handle=self.kernel.OpenProcess(0x100000,False,pid)
                if handle:held.append(handle)
                elif ctypes.get_last_error()!=87:raise ctypes.WinError(ctypes.get_last_error())
            if not self.kernel.TerminateJobObject(self.job,1):raise ctypes.WinError(ctypes.get_last_error())
            deadline=time.monotonic()+5
            for handle in held:
                if self.api.WaitForSingleObject(handle,max(0,int((deadline-time.monotonic())*1000)))!=0:
                    raise RuntimeError('OWNED_JOB_DESCENDANT_NOT_REAPED')
            self.wait(max(0,deadline-time.monotonic()))
            after=self.members()
            if after:raise RuntimeError('OWNED_JOB_NOT_EMPTY')
            return dict(kind='windows_job',active_before=before,active_after=after,owned_process_handles_signaled=len(held))
        finally:
            for handle in held:self.kernel.CloseHandle(handle)

    def close(self):
        if self.job:self.kernel.CloseHandle(self.job);self.job=None
        if self.process:self.api.CloseHandle(self.process);self.process=None
        for stream in (self.stdout,self.stderr):
            if stream is not None and not stream.closed:stream.close()
