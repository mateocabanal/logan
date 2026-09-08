#![allow(dead_code)]

use std::ffi::{CStr, CString};
use std::marker::PhantomData;
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;
use std::rc::Rc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::{AneError, Result};

pub(crate) type Id = *mut c_void;
pub(crate) type Class = *mut c_void;
pub(crate) type Sel = *mut c_void;
pub(crate) type IOSurfaceRef = *mut c_void;

const RTLD_NOW: c_int = 0x2;
const RTLD_LOCAL: c_int = 0x4;
const IOSURFACE_LOCK_READ_ONLY: u32 = 0x1;
const APPLE_NEURAL_ENGINE: &str =
    "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine";
const ANE_COMPILER: &str = "/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler";

static INIT: OnceLock<std::result::Result<(), String>> = OnceLock::new();
static COMPILER_LOADED: AtomicBool = AtomicBool::new(false);

unsafe extern "C" {
    fn dlopen(path: *const c_char, mode: c_int) -> *mut c_void;
    fn dlerror() -> *const c_char;
}

#[link(name = "objc")]
unsafe extern "C" {
    fn objc_getClass(name: *const c_char) -> Class;
    fn sel_registerName(name: *const c_char) -> Sel;
    fn objc_msgSend();
    fn objc_retain(value: Id) -> Id;
    fn objc_release(value: Id);
    fn objc_autoreleasePoolPush() -> *mut c_void;
    fn objc_autoreleasePoolPop(pool: *mut c_void);
    fn class_getInstanceMethod(cls: Class, sel: Sel) -> *mut c_void;
    fn class_getClassMethod(cls: Class, sel: Sel) -> *mut c_void;
    fn method_getTypeEncoding(method: *mut c_void) -> *const c_char;
}

#[link(name = "Foundation", kind = "framework")]
unsafe extern "C" {
    fn NSTemporaryDirectory() -> Id;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFRelease(value: *const c_void);
}

#[link(name = "IOSurface", kind = "framework")]
unsafe extern "C" {
    static kIOSurfaceWidth: Id;
    static kIOSurfaceHeight: Id;
    static kIOSurfaceBytesPerElement: Id;
    static kIOSurfaceBytesPerRow: Id;
    static kIOSurfaceAllocSize: Id;
    static kIOSurfacePixelFormat: Id;

    fn IOSurfaceCreate(properties: *const c_void) -> IOSurfaceRef;
    fn IOSurfaceGetAllocSize(surface: IOSurfaceRef) -> usize;
    fn IOSurfaceGetID(surface: IOSurfaceRef) -> u32;
    fn IOSurfaceGetBaseAddress(surface: IOSurfaceRef) -> *mut c_void;
    fn IOSurfaceLock(surface: IOSurfaceRef, options: u32, seed: *mut u32) -> i32;
    fn IOSurfaceUnlock(surface: IOSurfaceRef, options: u32, seed: *mut u32) -> i32;
}

pub(crate) fn ensure_frameworks() -> Result<()> {
    let result = INIT.get_or_init(|| unsafe {
        let ane = CString::new(APPLE_NEURAL_ENGINE).unwrap();
        if dlopen(ane.as_ptr(), RTLD_NOW | RTLD_LOCAL).is_null() {
            return Err(dlerror_string());
        }
        let compiler = CString::new(ANE_COMPILER).unwrap();
        if !dlopen(compiler.as_ptr(), RTLD_NOW | RTLD_LOCAL).is_null() {
            COMPILER_LOADED.store(true, Ordering::Relaxed);
        }
        Ok(())
    });
    result.clone().map_err(|message| AneError::FrameworkLoad {
        path: APPLE_NEURAL_ENGINE,
        message,
    })
}

pub(crate) fn compiler_framework_loaded() -> bool {
    COMPILER_LOADED.load(Ordering::Relaxed)
}

unsafe fn dlerror_string() -> String {
    let p = unsafe { dlerror() };
    if p.is_null() {
        "unknown dlopen error".into()
    } else {
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }
}

pub(crate) fn class(name: &'static str) -> Result<Class> {
    let c = CString::new(name).unwrap();
    let cls = unsafe { objc_getClass(c.as_ptr()) };
    if cls.is_null() {
        Err(AneError::MissingClass(name))
    } else {
        Ok(cls)
    }
}

pub(crate) fn selector(name: &'static str) -> Sel {
    let c = CString::new(name).unwrap();
    unsafe { sel_registerName(c.as_ptr()) }
}

pub(crate) fn require_class_selector(
    class_name: &'static str,
    selector_name: &'static str,
) -> Result<()> {
    let cls = class(class_name)?;
    let method = unsafe { class_getClassMethod(cls, selector(selector_name)) };
    if method.is_null() {
        Err(AneError::MissingSelector {
            class: class_name,
            selector: selector_name,
        })
    } else {
        Ok(())
    }
}

pub(crate) fn require_instance_selector(
    class_name: &'static str,
    selector_name: &'static str,
) -> Result<()> {
    let cls = class(class_name)?;
    let method = unsafe { class_getInstanceMethod(cls, selector(selector_name)) };
    if method.is_null() {
        Err(AneError::MissingSelector {
            class: class_name,
            selector: selector_name,
        })
    } else {
        Ok(())
    }
}

fn method_encoding(method: *mut c_void) -> String {
    if method.is_null() {
        return String::new();
    }
    let encoding = unsafe { method_getTypeEncoding(method) };
    if encoding.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(encoding) }
            .to_string_lossy()
            .into_owned()
    }
}

pub(crate) fn require_class_selector_encoding(
    class_name: &'static str,
    selector_name: &'static str,
    expected: &'static str,
) -> Result<()> {
    let cls = class(class_name)?;
    let method = unsafe { class_getClassMethod(cls, selector(selector_name)) };
    if method.is_null() {
        return Err(AneError::MissingSelector {
            class: class_name,
            selector: selector_name,
        });
    }
    let actual = method_encoding(method);
    if actual != expected {
        return Err(AneError::AbiMismatch {
            class: class_name,
            selector: selector_name,
            expected,
            actual,
        });
    }
    Ok(())
}

pub(crate) fn require_instance_selector_encoding(
    class_name: &'static str,
    selector_name: &'static str,
    expected: &'static str,
) -> Result<()> {
    let cls = class(class_name)?;
    let method = unsafe { class_getInstanceMethod(cls, selector(selector_name)) };
    if method.is_null() {
        return Err(AneError::MissingSelector {
            class: class_name,
            selector: selector_name,
        });
    }
    let actual = method_encoding(method);
    if actual != expected {
        return Err(AneError::AbiMismatch {
            class: class_name,
            selector: selector_name,
            expected,
            actual,
        });
    }
    Ok(())
}

pub(crate) fn has_class_selector(class_name: &'static str, selector_name: &'static str) -> bool {
    let Ok(cls) = class(class_name) else {
        return false;
    };
    unsafe { !class_getClassMethod(cls, selector(selector_name)).is_null() }
}

pub(crate) fn has_instance_selector(class_name: &'static str, selector_name: &'static str) -> bool {
    let Ok(cls) = class(class_name) else {
        return false;
    };
    unsafe { !class_getInstanceMethod(cls, selector(selector_name)).is_null() }
}

pub(crate) struct AutoreleasePool(*mut c_void);

impl AutoreleasePool {
    pub(crate) fn new() -> Self {
        Self(unsafe { objc_autoreleasePoolPush() })
    }
}

impl Drop for AutoreleasePool {
    fn drop(&mut self) {
        unsafe { objc_autoreleasePoolPop(self.0) }
    }
}

pub(crate) struct Retained {
    ptr: Id,
    _not_send_sync: PhantomData<Rc<()>>,
}

impl Retained {
    pub(crate) fn new(ptr: Id, operation: &'static str) -> Result<Self> {
        if ptr.is_null() {
            return Err(AneError::NullResult(operation));
        }
        let ptr = unsafe { objc_retain(ptr) };
        Ok(Self {
            ptr,
            _not_send_sync: PhantomData,
        })
    }

    pub(crate) fn as_ptr(&self) -> Id {
        self.ptr
    }
}

impl Drop for Retained {
    fn drop(&mut self) {
        unsafe { objc_release(self.ptr) }
    }
}

#[inline]
pub(crate) unsafe fn msg0_id(receiver: Id, sel: Sel) -> Id {
    let f: unsafe extern "C" fn(Id, Sel) -> Id =
        unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { f(receiver, sel) }
}

#[inline]
pub(crate) unsafe fn msg0_u64(receiver: Id, sel: Sel) -> u64 {
    let f: unsafe extern "C" fn(Id, Sel) -> u64 =
        unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { f(receiver, sel) }
}

#[inline]
pub(crate) unsafe fn msg0_u32(receiver: Id, sel: Sel) -> u32 {
    let f: unsafe extern "C" fn(Id, Sel) -> u32 =
        unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { f(receiver, sel) }
}

#[inline]
pub(crate) unsafe fn msg0_i8(receiver: Id, sel: Sel) -> i8 {
    let f: unsafe extern "C" fn(Id, Sel) -> i8 =
        unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { f(receiver, sel) }
}

#[inline]
pub(crate) unsafe fn msg0_bool(receiver: Id, sel: Sel) -> bool {
    let f: unsafe extern "C" fn(Id, Sel) -> u8 =
        unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { f(receiver, sel) != 0 }
}

#[inline]
pub(crate) unsafe fn msg0_void(receiver: Id, sel: Sel) {
    let f: unsafe extern "C" fn(Id, Sel) =
        unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { f(receiver, sel) }
}

#[inline]
pub(crate) unsafe fn msg1_u32_void(receiver: Id, sel: Sel, value: u32) {
    let f: unsafe extern "C" fn(Id, Sel, u32) =
        unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { f(receiver, sel, value) }
}

#[inline]
pub(crate) unsafe fn msg1_i8_void(receiver: Id, sel: Sel, value: i8) {
    let f: unsafe extern "C" fn(Id, Sel, i8) =
        unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { f(receiver, sel, value) }
}

#[inline]
pub(crate) unsafe fn msg1_id(receiver: Id, sel: Sel, a: Id) -> Id {
    let f: unsafe extern "C" fn(Id, Sel, Id) -> Id =
        unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { f(receiver, sel, a) }
}

#[inline]
pub(crate) unsafe fn msg3_id(receiver: Id, sel: Sel, a: Id, b: Id, c: Id) -> Id {
    let f: unsafe extern "C" fn(Id, Sel, Id, Id, Id) -> Id =
        unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { f(receiver, sel, a, b, c) }
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn msg7_id(
    receiver: Id,
    sel: Sel,
    a: Id,
    b: Id,
    c: Id,
    d: Id,
    e: Id,
    f_arg: Id,
    g: Id,
) -> Id {
    let f: unsafe extern "C" fn(Id, Sel, Id, Id, Id, Id, Id, Id, Id) -> Id =
        unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { f(receiver, sel, a, b, c, d, e, f_arg, g) }
}

pub(crate) unsafe fn msg_compile(
    receiver: Id,
    sel: Sel,
    qos: u32,
    options: Id,
    error: *mut Id,
) -> bool {
    let f: unsafe extern "C" fn(Id, Sel, u32, Id, *mut Id) -> u8 =
        unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { f(receiver, sel, qos, options, error) != 0 }
}

pub(crate) unsafe fn msg_evaluate(
    receiver: Id,
    sel: Sel,
    qos: u32,
    options: Id,
    request: Id,
    error: *mut Id,
) -> bool {
    let f: unsafe extern "C" fn(Id, Sel, u32, Id, Id, *mut Id) -> u8 =
        unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { f(receiver, sel, qos, options, request, error) != 0 }
}

pub(crate) unsafe fn msg_unload(receiver: Id, sel: Sel, qos: u32, error: *mut Id) -> bool {
    let f: unsafe extern "C" fn(Id, Sel, u32, *mut Id) -> u8 =
        unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { f(receiver, sel, qos, error) != 0 }
}

pub(crate) unsafe fn msg_client_evaluate(
    receiver: Id,
    sel: Sel,
    model: Id,
    options: Id,
    request: Id,
    qos: u32,
    error: *mut Id,
) -> bool {
    let f: unsafe extern "C" fn(Id, Sel, Id, Id, Id, u32, *mut Id) -> u8 =
        unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { f(receiver, sel, model, options, request, qos, error) != 0 }
}

pub(crate) unsafe fn msg_client_map_request(
    receiver: Id,
    sel: Sel,
    model: Id,
    request: Id,
    cache_inference: bool,
    error: *mut Id,
) -> bool {
    let f: unsafe extern "C" fn(Id, Sel, Id, Id, u8, *mut Id) -> u8 =
        unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { f(receiver, sel, model, request, cache_inference as u8, error) != 0 }
}

pub(crate) unsafe fn msg_client_unmap_request(receiver: Id, sel: Sel, model: Id, request: Id) {
    let f: unsafe extern "C" fn(Id, Sel, Id, Id) =
        unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { f(receiver, sel, model, request) }
}

pub(crate) unsafe fn msg_map_request(
    receiver: Id,
    sel: Sel,
    request: Id,
    cache_inference: bool,
    error: *mut Id,
) -> bool {
    let f: unsafe extern "C" fn(Id, Sel, Id, u8, *mut Id) -> u8 =
        unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { f(receiver, sel, request, cache_inference as u8, error) != 0 }
}

pub(crate) unsafe fn msg_unmap_request(receiver: Id, sel: Sel, request: Id) {
    let f: unsafe extern "C" fn(Id, Sel, Id) =
        unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { f(receiver, sel, request) }
}

pub(crate) unsafe fn msg_map_mutable_weights(
    receiver: Id,
    sel: Sel,
    model: Id,
    procedure: Id,
    mapped: *mut *mut c_void,
    size: *mut u64,
    error: *mut Id,
) -> bool {
    let f: unsafe extern "C" fn(Id, Sel, Id, Id, *mut *mut c_void, *mut u64, *mut Id) -> u8 =
        unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { f(receiver, sel, model, procedure, mapped, size, error) != 0 }
}

pub(crate) unsafe fn msg_sync_mutable_weights(
    receiver: Id,
    sel: Sel,
    model: Id,
    procedure: Id,
    offset: u64,
    size: u64,
    error: *mut Id,
) -> bool {
    let f: unsafe extern "C" fn(Id, Sel, Id, Id, u64, u64, *mut Id) -> u8 =
        unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { f(receiver, sel, model, procedure, offset, size, error) != 0 }
}

pub(crate) unsafe fn msg_unmap_mutable_weights(
    receiver: Id,
    sel: Sel,
    model: Id,
    procedure: Id,
) -> bool {
    let f: unsafe extern "C" fn(Id, Sel, Id, Id) -> u8 =
        unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { f(receiver, sel, model, procedure) != 0 }
}

pub(crate) unsafe fn msg_data(class: Class, sel: Sel, bytes: *const u8, len: usize) -> Id {
    let f: unsafe extern "C" fn(Class, Sel, *const u8, usize) -> Id =
        unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { f(class, sel, bytes, len) }
}

pub(crate) unsafe fn msg_cstr(class: Class, sel: Sel, text: *const c_char) -> Id {
    let f: unsafe extern "C" fn(Class, Sel, *const c_char) -> Id =
        unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { f(class, sel, text) }
}

pub(crate) unsafe fn msg_u64_object(class: Class, sel: Sel, value: u64) -> Id {
    let f: unsafe extern "C" fn(Class, Sel, u64) -> Id =
        unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { f(class, sel, value) }
}

pub(crate) unsafe fn msg_objects_count(
    class: Class,
    sel: Sel,
    objects: *const Id,
    count: usize,
) -> Id {
    let f: unsafe extern "C" fn(Class, Sel, *const Id, usize) -> Id =
        unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { f(class, sel, objects, count) }
}

pub(crate) unsafe fn msg_dictionary(
    class: Class,
    sel: Sel,
    objects: *const Id,
    keys: *const Id,
    count: usize,
) -> Id {
    let f: unsafe extern "C" fn(Class, Sel, *const Id, *const Id, usize) -> Id =
        unsafe { std::mem::transmute(objc_msgSend as *const ()) };
    unsafe { f(class, sel, objects, keys, count) }
}

pub(crate) fn ns_string(text: &str) -> Result<Id> {
    let cls = class("NSString")?;
    let c =
        CString::new(text).map_err(|_| AneError::InvalidArgument("string contains NUL".into()))?;
    let value = unsafe { msg_cstr(cls, selector("stringWithUTF8String:"), c.as_ptr()) };
    if value.is_null() {
        Err(AneError::NullResult("NSString creation"))
    } else {
        Ok(value)
    }
}

pub(crate) fn ns_data(bytes: &[u8]) -> Result<Id> {
    let cls = class("NSData")?;
    let value = unsafe {
        msg_data(
            cls,
            selector("dataWithBytes:length:"),
            bytes.as_ptr(),
            bytes.len(),
        )
    };
    if value.is_null() {
        Err(AneError::NullResult("NSData creation"))
    } else {
        Ok(value)
    }
}

pub(crate) fn ns_number(value: u64) -> Result<Id> {
    let cls = class("NSNumber")?;
    let value = unsafe { msg_u64_object(cls, selector("numberWithUnsignedLongLong:"), value) };
    if value.is_null() {
        Err(AneError::NullResult("NSNumber creation"))
    } else {
        Ok(value)
    }
}

pub(crate) fn ns_array(values: &[Id]) -> Result<Id> {
    let cls = class("NSArray")?;
    let value = unsafe {
        msg_objects_count(
            cls,
            selector("arrayWithObjects:count:"),
            values.as_ptr(),
            values.len(),
        )
    };
    if value.is_null() {
        Err(AneError::NullResult("NSArray creation"))
    } else {
        Ok(value)
    }
}

pub(crate) fn ns_dictionary(pairs: &[(Id, Id)]) -> Result<Id> {
    let cls = class("NSDictionary")?;
    if pairs.is_empty() {
        let value = unsafe { msg0_id(cls, selector("dictionary")) };
        return if value.is_null() {
            Err(AneError::NullResult("NSDictionary creation"))
        } else {
            Ok(value)
        };
    }
    let mut keys = Vec::with_capacity(pairs.len());
    let mut values = Vec::with_capacity(pairs.len());
    for &(key, value) in pairs {
        keys.push(key);
        values.push(value);
    }
    let value = unsafe {
        msg_dictionary(
            cls,
            selector("dictionaryWithObjects:forKeys:count:"),
            values.as_ptr(),
            keys.as_ptr(),
            pairs.len(),
        )
    };
    if value.is_null() {
        Err(AneError::NullResult("NSDictionary creation"))
    } else {
        Ok(value)
    }
}

pub(crate) fn nsstring_to_string(value: Id) -> Option<String> {
    if value.is_null() {
        return None;
    }
    let ptr = unsafe {
        let f: unsafe extern "C" fn(Id, Sel) -> *const c_char =
            std::mem::transmute(objc_msgSend as *const ());
        f(value, selector("UTF8String"))
    };
    if ptr.is_null() {
        None
    } else {
        Some(
            unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned(),
        )
    }
}

pub(crate) fn object_description(value: Id) -> String {
    if value.is_null() {
        return "unknown private API error".into();
    }
    let description = unsafe { msg0_id(value, selector("description")) };
    nsstring_to_string(description).unwrap_or_else(|| "unknown private API error".into())
}

pub(crate) fn temporary_directory() -> Result<String> {
    let path = unsafe { NSTemporaryDirectory() };
    nsstring_to_string(path).ok_or(AneError::NullResult("NSTemporaryDirectory"))
}

pub(crate) fn create_surface(bytes: usize) -> Result<IOSurfaceRef> {
    if bytes == 0 {
        return Err(AneError::InvalidArgument(
            "IOSurface size must be non-zero".into(),
        ));
    }
    let _pool = AutoreleasePool::new();
    let width = ns_number(bytes as u64)?;
    let one = ns_number(1)?;
    let pixel_format = ns_number(0)?;
    let pairs = unsafe {
        [
            (kIOSurfaceWidth, width),
            (kIOSurfaceHeight, one),
            (kIOSurfaceBytesPerElement, one),
            (kIOSurfaceBytesPerRow, width),
            (kIOSurfaceAllocSize, width),
            (kIOSurfacePixelFormat, pixel_format),
        ]
    };
    let dict = ns_dictionary(&pairs)?;
    let surface = unsafe { IOSurfaceCreate(dict as *const c_void) };
    if surface.is_null() {
        Err(AneError::NullResult("IOSurfaceCreate"))
    } else {
        Ok(surface)
    }
}

pub(crate) fn surface_alloc_size(surface: IOSurfaceRef) -> usize {
    unsafe { IOSurfaceGetAllocSize(surface) }
}

pub(crate) fn surface_id(surface: IOSurfaceRef) -> u32 {
    unsafe { IOSurfaceGetID(surface) }
}

pub(crate) fn lock_surface(surface: IOSurfaceRef, read_only: bool) -> Result<*mut u8> {
    let options = if read_only {
        IOSURFACE_LOCK_READ_ONLY
    } else {
        0
    };
    let rc = unsafe { IOSurfaceLock(surface, options, ptr::null_mut()) };
    if rc != 0 {
        return Err(AneError::Surface {
            operation: "lock",
            code: rc,
        });
    }
    let ptr = unsafe { IOSurfaceGetBaseAddress(surface) } as *mut u8;
    if ptr.is_null() {
        let _ = unsafe { IOSurfaceUnlock(surface, options, ptr::null_mut()) };
        Err(AneError::NullResult("IOSurfaceGetBaseAddress"))
    } else {
        Ok(ptr)
    }
}

pub(crate) fn unlock_surface(surface: IOSurfaceRef, read_only: bool) {
    let options = if read_only {
        IOSURFACE_LOCK_READ_ONLY
    } else {
        0
    };
    let _ = unsafe { IOSurfaceUnlock(surface, options, ptr::null_mut()) };
}

pub(crate) fn release_surface(surface: IOSurfaceRef) {
    if !surface.is_null() {
        unsafe { CFRelease(surface as *const c_void) }
    }
}
