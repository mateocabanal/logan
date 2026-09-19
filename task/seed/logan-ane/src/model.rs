use std::marker::PhantomData;
use std::ops::Range;
use std::os::raw::c_void;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use crate::mil::MilProgram;
use crate::raw;
use crate::surface::AneSurface;
use crate::{AneError, Result};

#[link(name = "logan_ane_async", kind = "static")]
unsafe extern "C" {
    fn logan_ane_async_submit_signal(
        in_memory_model: *mut c_void,
        input_surfaces: *const *mut c_void,
        input_count: usize,
        output_surfaces: *const *mut c_void,
        output_count: usize,
        procedure_index: u64,
        shared_event: *mut c_void,
        signal_value: u64,
        qos: u32,
        direct_client: u8,
        error_buf: *mut i8,
        error_cap: usize,
    ) -> *mut c_void;
    fn logan_ane_async_finish(
        pending: *mut c_void,
        timeout_ms: u64,
        error_buf: *mut i8,
        error_cap: usize,
    ) -> i32;
    fn logan_ane_async_discard(pending: *mut c_void);
    fn logan_ane_async_channel_create(
        in_memory_model: *mut c_void,
        input_surfaces: *const *mut c_void,
        input_count: usize,
        output_surfaces: *const *mut c_void,
        output_count: usize,
        procedure_index: u64,
        shared_event: *mut c_void,
        wait_shared_event: *mut c_void,
        qos: u32,
        submit_mode: u8,
        premap: u8,
        error_buf: *mut i8,
        error_cap: usize,
    ) -> *mut c_void;
    fn logan_ane_async_channel_submit(
        channel: *mut c_void,
        wait_value: u64,
        signal_value: u64,
        error_buf: *mut i8,
        error_cap: usize,
    ) -> *mut c_void;
    fn logan_ane_async_channel_finish(
        pending: *mut c_void,
        timeout_ms: u64,
        error_buf: *mut i8,
        error_cap: usize,
    ) -> i32;
    fn logan_ane_async_channel_discard_pending(pending: *mut c_void);
    fn logan_ane_async_channel_free(channel: *mut c_void);
    fn logan_ane_probe_mutable_buffer(
        in_memory_model: *mut c_void,
        buffer_id: u64,
        size_out: *mut u64,
        error_buf: *mut i8,
        error_cap: usize,
    ) -> i32;
}

fn async_error(buf: &[u8]) -> String {
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).into_owned()
}

/// Owning split-phase ANE request. The native shim retains the in-memory
/// model, request, completion block, shared event, and IOSurface wrappers.
/// Dropping without `finish` drains completion before releasing them.
pub struct AnePending {
    raw: Option<std::ptr::NonNull<c_void>>,
    _not_send_sync: PhantomData<Rc<()>>,
}

impl AnePending {
    pub fn finish(mut self, timeout_ms: u64) -> Result<()> {
        let Some(raw) = self.raw.take() else {
            return Err(AneError::InvalidArgument("ANE pending already consumed".into()));
        };
        let mut error = [0u8; 512];
        let rc = unsafe {
            logan_ane_async_finish(
                raw.as_ptr(),
                timeout_ms,
                error.as_mut_ptr().cast(),
                error.len(),
            )
        };
        match rc {
            1 => Ok(()),
            0 => {
                // The native object intentionally remains retained on timeout;
                // releasing it while the callback may still run would be UAF.
                std::mem::forget(self);
                Err(AneError::ObjectiveC {
                    operation: "async evaluate timeout",
                    message: async_error(&error),
                })
            }
            _ => Err(AneError::ObjectiveC {
                operation: "async evaluate",
                message: async_error(&error),
            }),
        }
    }
}

impl Drop for AnePending {
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            unsafe { logan_ane_async_discard(raw.as_ptr()) };
        }
    }
}

/// Reusable per-layer ANE request/channel. IOSurface wrappers, shared-event
/// objects, request arrays and completion block are built once; each decode
/// only updates the monotonic event value and resubmits.
pub struct AneAsyncChannel {
    raw: std::ptr::NonNull<c_void>,
    _not_send_sync: PhantomData<Rc<()>>,
}

pub struct AneChannelPending {
    raw: Option<std::ptr::NonNull<c_void>>,
    _not_send_sync: PhantomData<Rc<()>>,
}

impl AneAsyncChannel {
    pub fn submit(&mut self, signal_value: u64) -> Result<AneChannelPending> {
        if signal_value == 0 {
            return Err(AneError::InvalidArgument("ANE channel signal value must be nonzero".into()));
        }
        let mut error = [0u8; 512];
        let raw = unsafe {
            logan_ane_async_channel_submit(
                self.raw.as_ptr(), 0, signal_value, error.as_mut_ptr().cast(), error.len(),
            )
        };
        let raw = std::ptr::NonNull::new(raw).ok_or_else(|| AneError::ObjectiveC {
            operation: "async channel submit",
            message: async_error(&error),
        })?;
        Ok(AneChannelPending { raw: Some(raw), _not_send_sync: PhantomData })
    }

    /// Submit a channel configured with a device-side wait event. ANE starts
    /// only after `wait_value` is signaled and signals `signal_value` on
    /// completion; both values can change per reusable submission.
    pub fn submit_after(&mut self, wait_value: u64, signal_value: u64) -> Result<AneChannelPending> {
        if wait_value == 0 || signal_value == 0 {
            return Err(AneError::InvalidArgument("ANE channel wait/signal values must be nonzero".into()));
        }
        let mut error = [0u8; 512];
        let raw = unsafe {
            logan_ane_async_channel_submit(
                self.raw.as_ptr(), wait_value, signal_value,
                error.as_mut_ptr().cast(), error.len(),
            )
        };
        let raw = std::ptr::NonNull::new(raw).ok_or_else(|| AneError::ObjectiveC {
            operation: "async channel submit-after", message: async_error(&error),
        })?;
        Ok(AneChannelPending { raw: Some(raw), _not_send_sync: PhantomData })
    }
}

impl Drop for AneAsyncChannel {
    fn drop(&mut self) {
        unsafe { logan_ane_async_channel_free(self.raw.as_ptr()) };
    }
}

impl AneChannelPending {
    pub fn finish(mut self, timeout_ms: u64) -> Result<()> {
        let Some(raw) = self.raw.take() else {
            return Err(AneError::InvalidArgument("ANE channel pending already consumed".into()));
        };
        let mut error = [0u8; 512];
        let rc = unsafe {
            logan_ane_async_channel_finish(raw.as_ptr(), timeout_ms, error.as_mut_ptr().cast(), error.len())
        };
        match rc {
            1 => Ok(()),
            0 => {
                std::mem::forget(self);
                Err(AneError::ObjectiveC {
                    operation: "async channel timeout",
                    message: async_error(&error),
                })
            }
            _ => Err(AneError::ObjectiveC {
                operation: "async channel evaluate",
                message: async_error(&error),
            }),
        }
    }
}

impl Drop for AneChannelPending {
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            unsafe { logan_ane_async_channel_discard_pending(raw.as_ptr()) };
        }
    }
}

const CLS_DESCRIPTOR: &str = "_ANEInMemoryModelDescriptor";
const CLS_MODEL: &str = "_ANEInMemoryModel";
const CLS_REQUEST: &str = "_ANERequest";
const CLS_SURFACE_OBJECT: &str = "_ANEIOSurfaceObject";
const CLS_DEVICE_INFO: &str = "_ANEDeviceInfo";
const CLS_CLIENT: &str = "_ANEClient";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AneQos(pub u32);

impl AneQos {
    /// QoS value used by the current private in-memory ANE path in CoreML-derived
    /// research implementations. It is intentionally overridable because the
    /// private ABI does not provide a stable public enum.
    pub const DEFAULT: Self = Self(21);
}

impl Default for AneQos {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[derive(Clone, Debug)]
pub struct CompileOptions {
    pub qos: AneQos,
    pub keep_temporary_files: bool,
    /// Optional Logan-owned persistent cache root. Each descriptor is stored
    /// under a deterministic hash directory beneath this root.
    pub cache_directory: Option<PathBuf>,
    /// Reuse Apple's private compiled-model cache when the descriptor already
    /// exists there. Falls back to a normal compile on a miss.
    pub reuse_compiled_model: bool,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            qos: AneQos::DEFAULT,
            keep_temporary_files: false,
            cache_directory: None,
            reuse_compiled_model: true,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct RuntimeCapabilities {
    pub in_memory_mil: bool,
    pub request_iosurfaces: bool,
    pub request_mapping: bool,
    pub client_request_mapping: bool,
    pub mutable_weights: bool,
    pub shared_events: bool,
    pub realtime_evaluation: bool,
    pub direct_client_evaluation: bool,
    pub ane_compiler_framework_loaded: bool,
}

#[derive(Clone, Debug, Default)]
pub struct AneDeviceInfo {
    pub has_ane: bool,
    pub ane_count: u32,
    pub core_count: u32,
    pub architecture: Option<String>,
    pub subtype: Option<String>,
    pub product_variant: Option<String>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ModelState {
    pub private_state: u64,
    pub program_handle: u64,
    pub intermediate_buffer_handle: u64,
    pub queue_depth: i8,
    pub perf_stats_mask: u32,
}

#[derive(Clone, Debug)]
pub struct AneRuntime {
    capabilities: RuntimeCapabilities,
    device: AneDeviceInfo,
}

impl AneRuntime {
    pub fn load() -> Result<Self> {
        raw::ensure_frameworks()?;
        raw::require_class_selector_encoding(
            CLS_DESCRIPTOR,
            "modelWithMILText:weights:optionsPlist:",
            "@40@0:8@16@24@32",
        )?;
        raw::require_class_selector_encoding(
            CLS_MODEL,
            "inMemoryModelWithDescriptor:",
            "@24@0:8@16",
        )?;
        raw::require_class_selector_encoding(
            CLS_REQUEST,
            "requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:",
            "@72@0:8@16@24@32@40@48@56@64",
        )?;
        raw::require_class_selector_encoding(
            CLS_SURFACE_OBJECT,
            "objectWithIOSurface:",
            "@24@0:8^{__IOSurface=}16",
        )?;
        for (selector, encoding) in [
            ("compileWithQoS:options:error:", "B36@0:8I16@20^@28"),
            ("loadWithQoS:options:error:", "B36@0:8I16@20^@28"),
            (
                "evaluateWithQoS:options:request:error:",
                "B44@0:8I16@20@28^@36",
            ),
            ("unloadWithQoS:error:", "B28@0:8I16^@20"),
            ("hexStringIdentifier", "@16@0:8"),
            ("model", "@16@0:8"),
            ("state", "Q16@0:8"),
            ("programHandle", "Q16@0:8"),
            ("intermediateBufferHandle", "Q16@0:8"),
            ("queueDepth", "c16@0:8"),
            ("perfStatsMask", "I16@0:8"),
        ] {
            raw::require_instance_selector_encoding(CLS_MODEL, selector, encoding)?;
        }

        let capabilities = RuntimeCapabilities {
            in_memory_mil: true,
            request_iosurfaces: true,
            request_mapping: raw::require_instance_selector_encoding(
                CLS_MODEL,
                "mapIOSurfacesWithRequest:cacheInference:error:",
                "B36@0:8@16B24^@28",
            )
            .is_ok()
                && raw::require_instance_selector_encoding(
                    CLS_MODEL,
                    "unmapIOSurfacesWithRequest:",
                    "v24@0:8@16",
                )
                .is_ok(),
            client_request_mapping: raw::require_instance_selector_encoding(
                CLS_CLIENT,
                "mapIOSurfacesWithModel:request:cacheInference:error:",
                "B44@0:8@16@24B32^@36",
            )
            .is_ok()
                && raw::require_instance_selector_encoding(
                    CLS_CLIENT,
                    "unmapIOSurfacesWithModel:request:",
                    "v32@0:8@16@24",
                )
                .is_ok(),
            mutable_weights: raw::require_instance_selector_encoding(
                CLS_CLIENT,
                "mapMutableWeightsForModel:andProcedure:mappedWeightsBuffer:size:error:",
                "B56@0:8@16@24^^v32^Q40^@48",
            )
            .is_ok()
                && raw::require_instance_selector_encoding(
                    CLS_CLIENT,
                    "syncMutableWeightsForModel:andProcedure:fromOffset:withSize:error:",
                    "B56@0:8@16@24Q32Q40^@48",
                )
                .is_ok()
                && raw::require_instance_selector_encoding(
                    CLS_CLIENT,
                    "unmapMutableWeightsForModel:andProcedure:",
                    "B32@0:8@16@24",
                )
                .is_ok(),
            shared_events: raw::require_class_selector_encoding(
                CLS_REQUEST,
                "requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:",
                "@80@0:8@16@24@32@40@48@56@64@72",
            )
            .is_ok(),
            realtime_evaluation: raw::require_instance_selector_encoding(
                CLS_CLIENT,
                "evaluateRealTimeWithModel:options:request:error:",
                "B48@0:8@16@24@32^@40",
            )
            .is_ok(),
            direct_client_evaluation: raw::require_instance_selector_encoding(
                CLS_CLIENT,
                "doEvaluateDirectWithModel:options:request:qos:error:",
                "B52@0:8@16@24@32I40^@44",
            )
            .is_ok(),
            ane_compiler_framework_loaded: raw::compiler_framework_loaded(),
        };

        Ok(Self {
            capabilities,
            device: read_device_info(),
        })
    }

    pub fn capabilities(&self) -> &RuntimeCapabilities {
        &self.capabilities
    }

    pub fn device_info(&self) -> &AneDeviceInfo {
        &self.device
    }

    pub fn compile(&self, program: &MilProgram, options: CompileOptions) -> Result<AneModel> {
        program.validate()?;
        compile_program(program, options)
    }

    /// Obtain the process-wide private ANE client used for lower-level calls
    /// that are not exposed on `_ANEInMemoryModel` itself.
    pub fn shared_client(&self) -> Result<AneClient> {
        AneClient::shared()
    }
}

fn read_device_info() -> AneDeviceInfo {
    let Ok(cls) = raw::class(CLS_DEVICE_INFO) else {
        return AneDeviceInfo::default();
    };
    let _pool = raw::AutoreleasePool::new();
    let has_ane = raw::require_class_selector_encoding(CLS_DEVICE_INFO, "hasANE", "B16@0:8")
        .is_ok()
        && unsafe { raw::msg0_bool(cls, raw::selector("hasANE")) };
    let core_count = if raw::require_class_selector_encoding(
        CLS_DEVICE_INFO,
        "numANECores",
        "I16@0:8",
    )
    .is_ok()
    {
        unsafe { raw::msg0_u32(cls, raw::selector("numANECores")) }
    } else {
        0
    };
    let ane_count =
        if raw::require_class_selector_encoding(CLS_DEVICE_INFO, "numANEs", "I16@0:8").is_ok() {
            unsafe { raw::msg0_u32(cls, raw::selector("numANEs")) }
        } else {
            0
        };
    let read_string = |selector_name: &'static str| {
        if raw::require_class_selector_encoding(CLS_DEVICE_INFO, selector_name, "@16@0:8").is_ok() {
            let value = unsafe { raw::msg0_id(cls, raw::selector(selector_name)) };
            raw::nsstring_to_string(value)
        } else {
            None
        }
    };
    AneDeviceInfo {
        has_ane,
        ane_count,
        core_count,
        architecture: read_string("aneArchitectureType"),
        subtype: read_string("aneSubType"),
        product_variant: read_string("aneSubTypeProductVariant"),
    }
}

/// Process-wide `_ANEClient` connection for private runtime operations that
/// sit below the `_ANEInMemoryModel` convenience lifecycle.
pub struct AneClient {
    object: raw::Retained,
    _not_send_sync: PhantomData<Rc<()>>,
}

impl AneClient {
    pub fn shared() -> Result<Self> {
        raw::ensure_frameworks()?;
        raw::require_class_selector_encoding(CLS_CLIENT, "sharedConnection", "@16@0:8")?;
        let _pool = raw::AutoreleasePool::new();
        let cls = raw::class(CLS_CLIENT)?;
        let object = unsafe { raw::msg0_id(cls, raw::selector("sharedConnection")) };
        Ok(Self {
            object: raw::Retained::new(object, "_ANEClient sharedConnection")?,
            _not_send_sync: PhantomData,
        })
    }

    fn underlying_model(model: &AneModel) -> Result<raw::Id> {
        let underlying = unsafe { raw::msg0_id(model.object.as_ptr(), raw::selector("model")) };
        if underlying.is_null() {
            Err(AneError::NullResult("_ANEInMemoryModel model"))
        } else {
            Ok(underlying)
        }
    }

    /// Reissue the private client load call for an already-loaded model as a
    /// residency hint. This is experimental: callers should use it only for
    /// hardware qualification and avoid repeated calls unless the runtime
    /// proves they are idempotent on the current OS.
    pub fn residency_hint(&self, model: &AneModel, qos: AneQos) -> Result<()> {
        raw::require_instance_selector_encoding(
            CLS_CLIENT,
            "loadModel:options:qos:error:",
            "B44@0:8@16@24I32^@36",
        )?;
        if !model.loaded {
            return Err(AneError::InvalidArgument(
                "model must be loaded before residency hint".into(),
            ));
        }
        let _pool = raw::AutoreleasePool::new();
        let underlying = Self::underlying_model(model)?;
        let options = raw::ns_dictionary(&[])?;
        let mut error: raw::Id = std::ptr::null_mut();
        let ok = unsafe {
            raw::msg_client_load(
                self.object.as_ptr(),
                raw::selector("loadModel:options:qos:error:"),
                underlying,
                options,
                qos.0,
                &mut error,
            )
        };
        if ok {
            Ok(())
        } else {
            Err(AneError::ObjectiveC {
                operation: "ANE residency hint",
                message: raw::object_description(error),
            })
        }
    }

    /// Evaluate through the process-wide `_ANEClient` instead of the
    /// `_ANEInMemoryModel` convenience method.
    ///
    /// This is useful when experimenting with the lower-level scheduling path
    /// while keeping Rust-owned model/request lifetimes intact.
    pub fn evaluate(&self, model: &AneModel, request: &AneRequest<'_>, qos: AneQos) -> Result<()> {
        raw::require_instance_selector_encoding(
            CLS_CLIENT,
            "evaluateWithModel:options:request:qos:error:",
            "B52@0:8@16@24@32I40^@44",
        )?;
        if !model.loaded {
            return Err(AneError::InvalidArgument(
                "model must be loaded before _ANEClient evaluation".into(),
            ));
        }
        let _pool = raw::AutoreleasePool::new();
        let underlying = Self::underlying_model(model)?;
        let options = raw::ns_dictionary(&[])?;
        let mut error: raw::Id = std::ptr::null_mut();
        let ok = unsafe {
            raw::msg_client_evaluate(
                self.object.as_ptr(),
                raw::selector("evaluateWithModel:options:request:qos:error:"),
                underlying,
                options,
                request.object.as_ptr(),
                qos.0,
                &mut error,
            )
        };
        if ok {
            Ok(())
        } else {
            Err(AneError::ObjectiveC {
                operation: "client evaluate",
                message: raw::object_description(error),
            })
        }
    }

    /// Use the private direct-evaluation selector that bypasses part of the
    /// `_ANEClient` convenience dispatch path.
    ///
    /// This selector is present on the validated host but remains private and
    /// version-fragile, so callers should capability-check
    /// `RuntimeCapabilities::direct_client_evaluation` first.
    pub fn evaluate_direct(
        &self,
        model: &AneModel,
        request: &AneRequest<'_>,
        qos: AneQos,
    ) -> Result<()> {
        raw::require_instance_selector_encoding(
            CLS_CLIENT,
            "doEvaluateDirectWithModel:options:request:qos:error:",
            "B52@0:8@16@24@32I40^@44",
        )?;
        if !model.loaded {
            return Err(AneError::InvalidArgument(
                "model must be loaded before direct _ANEClient evaluation".into(),
            ));
        }
        let _pool = raw::AutoreleasePool::new();
        let underlying = Self::underlying_model(model)?;
        let options = raw::ns_dictionary(&[])?;
        let mut error: raw::Id = std::ptr::null_mut();
        let ok = unsafe {
            raw::msg_client_evaluate(
                self.object.as_ptr(),
                raw::selector("doEvaluateDirectWithModel:options:request:qos:error:"),
                underlying,
                options,
                request.object.as_ptr(),
                qos.0,
                &mut error,
            )
        };
        if ok {
            Ok(())
        } else {
            Err(AneError::ObjectiveC {
                operation: "direct client evaluate",
                message: raw::object_description(error),
            })
        }
    }

    /// Pre-map a request's IOSurfaces through `_ANEClient` for experiments
    /// with cache-inference and repeated dispatch behavior.
    pub fn map_request(
        &self,
        model: &AneModel,
        request: &AneRequest<'_>,
        cache_inference: bool,
    ) -> Result<()> {
        raw::require_instance_selector_encoding(
            CLS_CLIENT,
            "mapIOSurfacesWithModel:request:cacheInference:error:",
            "B44@0:8@16@24B32^@36",
        )?;
        let _pool = raw::AutoreleasePool::new();
        let underlying = Self::underlying_model(model)?;
        let mut error: raw::Id = std::ptr::null_mut();
        let ok = unsafe {
            raw::msg_client_map_request(
                self.object.as_ptr(),
                raw::selector("mapIOSurfacesWithModel:request:cacheInference:error:"),
                underlying,
                request.object.as_ptr(),
                cache_inference,
                &mut error,
            )
        };
        if ok {
            Ok(())
        } else {
            Err(AneError::ObjectiveC {
                operation: "client map IOSurfaces",
                message: raw::object_description(error),
            })
        }
    }

    pub fn unmap_request(&self, model: &AneModel, request: &AneRequest<'_>) -> Result<()> {
        raw::require_instance_selector_encoding(
            CLS_CLIENT,
            "unmapIOSurfacesWithModel:request:",
            "v32@0:8@16@24",
        )?;
        let underlying = Self::underlying_model(model)?;
        unsafe {
            raw::msg_client_unmap_request(
                self.object.as_ptr(),
                raw::selector("unmapIOSurfacesWithModel:request:"),
                underlying,
                request.object.as_ptr(),
            )
        };
        Ok(())
    }

    /// Map the mutable-weight buffer for a private ANE model/procedure pair.
    ///
    /// This exposes the locally discovered `_ANEClient` selector directly
    /// without pretending that Apple's undocumented procedure-object contract
    /// is stable. Use `AneModel::as_raw_underlying_model()` for the first
    /// argument; the procedure object must correspond to the compiled program.
    ///
    /// # Safety
    ///
    /// `model` and `procedure` must be valid Objective-C objects accepted by
    /// the running OS's `_ANEClient`. The model must have been compiled with a
    /// mutable-weight procedure. Passing objects of the wrong private class can
    /// crash the process because this is an undocumented Objective-C ABI.
    pub unsafe fn map_mutable_weights_raw<'a>(
        &'a self,
        model: *mut c_void,
        procedure: *mut c_void,
    ) -> Result<MutableWeightMapping<'a>> {
        raw::require_instance_selector_encoding(
            CLS_CLIENT,
            "mapMutableWeightsForModel:andProcedure:mappedWeightsBuffer:size:error:",
            "B56@0:8@16@24^^v32^Q40^@48",
        )?;
        raw::require_instance_selector_encoding(
            CLS_CLIENT,
            "syncMutableWeightsForModel:andProcedure:fromOffset:withSize:error:",
            "B56@0:8@16@24Q32Q40^@48",
        )?;
        raw::require_instance_selector_encoding(
            CLS_CLIENT,
            "unmapMutableWeightsForModel:andProcedure:",
            "B32@0:8@16@24",
        )?;
        if model.is_null() || procedure.is_null() {
            return Err(AneError::InvalidArgument(
                "mutable-weight model/procedure objects must be non-null".into(),
            ));
        }

        let model_hold = raw::Retained::new(model, "mutable-weight model retain")?;
        let procedure_hold = raw::Retained::new(procedure, "mutable-weight procedure retain")?;
        let mut mapped = std::ptr::null_mut();
        let mut size = 0u64;
        let mut error: raw::Id = std::ptr::null_mut();
        let ok = unsafe {
            raw::msg_map_mutable_weights(
                self.object.as_ptr(),
                raw::selector(
                    "mapMutableWeightsForModel:andProcedure:mappedWeightsBuffer:size:error:",
                ),
                model,
                procedure,
                &mut mapped,
                &mut size,
                &mut error,
            )
        };
        if !ok {
            return Err(AneError::ObjectiveC {
                operation: "map mutable weights",
                message: raw::object_description(error),
            });
        }
        if mapped.is_null() || size == 0 {
            return Err(AneError::NullResult("mutable-weight mapping"));
        }
        let len = usize::try_from(size).map_err(|_| {
            AneError::InvalidArgument(format!("mutable-weight mapping is too large: {size} bytes"))
        })?;
        Ok(MutableWeightMapping {
            client: self,
            model: model_hold,
            procedure: procedure_hold,
            ptr: mapped.cast(),
            len,
        })
    }

    /// Borrow `_ANEClient *` for private selectors that `logan-ane` has not
    /// wrapped yet.
    ///
    /// # Safety
    ///
    /// The pointer is unowned and valid only while `self` lives.
    pub unsafe fn as_raw_object(&self) -> *mut c_void {
        self.object.as_ptr()
    }
}

/// CPU-visible private ANE mutable-weight mapping.
///
/// Creation is unsafe because Apple's model/procedure object contract is
/// private. Once successfully created, bounds and mapping lifetime are enforced
/// by this wrapper; call `sync` after modifying a range before ANE evaluation.
pub struct MutableWeightMapping<'a> {
    client: &'a AneClient,
    model: raw::Retained,
    procedure: raw::Retained,
    ptr: *mut u8,
    len: usize,
}

impl MutableWeightMapping<'_> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn sync(&self, range: Range<usize>) -> Result<()> {
        if range.start > range.end || range.end > self.len {
            return Err(AneError::InvalidArgument(format!(
                "mutable-weight sync range {:?} exceeds {} bytes",
                range, self.len
            )));
        }
        if range.is_empty() {
            return Ok(());
        }
        let mut error: raw::Id = std::ptr::null_mut();
        let ok = unsafe {
            raw::msg_sync_mutable_weights(
                self.client.object.as_ptr(),
                raw::selector("syncMutableWeightsForModel:andProcedure:fromOffset:withSize:error:"),
                self.model.as_ptr(),
                self.procedure.as_ptr(),
                range.start as u64,
                (range.end - range.start) as u64,
                &mut error,
            )
        };
        if ok {
            Ok(())
        } else {
            Err(AneError::ObjectiveC {
                operation: "sync mutable weights",
                message: raw::object_description(error),
            })
        }
    }
}

impl Drop for MutableWeightMapping<'_> {
    fn drop(&mut self) {
        unsafe {
            let _ = raw::msg_unmap_mutable_weights(
                self.client.object.as_ptr(),
                raw::selector("unmapMutableWeightsForModel:andProcedure:"),
                self.model.as_ptr(),
                self.procedure.as_ptr(),
            );
        }
    }
}

pub struct AneModel {
    object: raw::Retained,
    temp_dir: PathBuf,
    qos: AneQos,
    loaded: bool,
    native_cache_hit: bool,
    keep_temporary_files: bool,
    _not_send_sync: PhantomData<Rc<()>>,
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<()> {
    if !src.is_dir() {
        return Ok(());
    }
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let out = dst.join(entry.file_name());
        if ty.is_dir() {
            copy_dir_recursive(&entry.path(), &out)?;
        } else if ty.is_file() {
            std::fs::copy(entry.path(), out)?;
        }
    }
    Ok(())
}

fn private_local_model_path(object: raw::Id) -> Option<PathBuf> {
    if raw::require_instance_selector_encoding(CLS_MODEL, "localModelPath", "@16@0:8").is_err() {
        return None;
    }
    let value = unsafe { raw::msg0_id(object, raw::selector("localModelPath")) };
    raw::nsstring_to_string(value).map(PathBuf::from)
}

fn private_compiled_model_exists(object: raw::Id) -> bool {
    raw::require_instance_selector_encoding(CLS_MODEL, "compiledModelExists", "B16@0:8")
        .is_ok()
        && unsafe { raw::msg0_bool(object, raw::selector("compiledModelExists")) }
}

fn compile_program(program: &MilProgram, options: CompileOptions) -> Result<AneModel> {
    let _pool = raw::AutoreleasePool::new();
    let mil_data = raw::ns_data(program.text().as_bytes())?;

    // Private API quirk: pass an empty NSDictionary when there are no weights.
    // `nil` is not equivalent for some MIL programs.
    let mut weight_pairs = Vec::with_capacity(program.weights().len());
    for weight in program.weights() {
        let path = raw::ns_string(weight.path())?;
        let offset_key = raw::ns_string("offset")?;
        let data_key = raw::ns_string("data")?;
        let offset = raw::ns_number(weight.descriptor_file_offset())?;
        let data = raw::ns_data(weight.data())?;
        let descriptor = raw::ns_dictionary(&[(offset_key, offset), (data_key, data)])?;
        weight_pairs.push((path, descriptor));
    }
    let weights = raw::ns_dictionary(&weight_pairs)?;

    let desc_cls = raw::class(CLS_DESCRIPTOR)?;
    let descriptor = unsafe {
        raw::msg3_id(
            desc_cls,
            raw::selector("modelWithMILText:weights:optionsPlist:"),
            mil_data,
            weights,
            std::ptr::null_mut(),
        )
    };
    if descriptor.is_null() {
        return Err(AneError::NullResult("_ANEInMemoryModelDescriptor creation"));
    }

    let model_cls = raw::class(CLS_MODEL)?;
    let model_ptr = unsafe {
        raw::msg1_id(
            model_cls,
            raw::selector("inMemoryModelWithDescriptor:"),
            descriptor,
        )
    };
    let object = raw::Retained::new(model_ptr, "_ANEInMemoryModel creation")?;

    let hex = unsafe { raw::msg0_id(object.as_ptr(), raw::selector("hexStringIdentifier")) };
    let hex = raw::nsstring_to_string(hex).ok_or(AneError::NullResult("hexStringIdentifier"))?;
    if hex.contains('/') || hex.contains("..") {
        return Err(AneError::InvalidArgument(format!(
            "private model identifier is not path-safe: {hex:?}"
        )));
    }

    let persistent = options.cache_directory.as_ref().map(|root| root.join(&hex));
    let temp_dir = match persistent.clone() {
        Some(path) => path,
        None => PathBuf::from(raw::temporary_directory()?).join(&hex),
    };
    std::fs::create_dir_all(&temp_dir)?;
    let mil_path = temp_dir.join("model.mil");
    if !mil_path.is_file() {
        std::fs::write(&mil_path, program.text().as_bytes())?;
    }
    for weight in program.weights() {
        let relative = weight.path().strip_prefix("@model_path/").unwrap();
        let destination = temp_dir.join(relative);
        if destination.is_file() {
            continue;
        }
        if let Some(parent) = destination.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(destination, weight.data())?;
    }

    // If Logan has a mirrored copy of the private runtime's local compiled
    // files, restore it to the exact path the framework expects before asking
    // the daemon whether the descriptor is already compiled.
    let runtime_local = private_local_model_path(object.as_ptr());
    if let (Some(entry), Some(runtime_local)) = (persistent.as_ref(), runtime_local.as_ref()) {
        let mirrored = entry.join("compiled");
        if mirrored.is_dir() && !runtime_local.is_dir() {
            copy_dir_recursive(&mirrored, runtime_local)?;
        }
    }

    let cache_hit = options.reuse_compiled_model && private_compiled_model_exists(object.as_ptr());
    if !cache_hit {
        // _ANEInMemoryModel derives its source bundle path from the descriptor
        // under NSTemporaryDirectory. A Logan cache directory does not redirect
        // that private path. Stage compiler inputs there on a native cache miss;
        // otherwise compile fails with verifyBundleAtPath: invalid model.
        let runtime_inputs = PathBuf::from(raw::temporary_directory()?).join(&hex);
        std::fs::create_dir_all(&runtime_inputs)?;
        std::fs::write(runtime_inputs.join("model.mil"), program.text().as_bytes())?;
        for weight in program.weights() {
            let relative = weight.path().strip_prefix("@model_path/").unwrap();
            let destination = runtime_inputs.join(relative);
            if let Some(parent) = destination.parent() { std::fs::create_dir_all(parent)?; }
            std::fs::write(destination, weight.data())?;
        }
        let empty_options = raw::ns_dictionary(&[])?;
        let mut error: raw::Id = std::ptr::null_mut();
        let ok = unsafe {
            raw::msg_compile(
                object.as_ptr(),
                raw::selector("compileWithQoS:options:error:"),
                options.qos.0,
                empty_options,
                &mut error,
            )
        };
        if !ok {
            let message = raw::object_description(error);
            if !options.keep_temporary_files && persistent.is_none() {
                let _ = std::fs::remove_dir_all(&temp_dir);
            }
            return Err(AneError::ObjectiveC {
                operation: "compile",
                message,
            });
        }
    }

    // Mirror native files only when the private runtime exposes a real local
    // directory. On current macOS the daemon often keeps the compiled object
    // solely in its own cache, in which case Logan records the native hit but
    // does not pretend the MIL/weight mirror is itself a compiled artifact.
    if let (Some(entry), Some(runtime_local)) = (persistent.as_ref(), private_local_model_path(object.as_ptr())) {
        if runtime_local.is_dir() {
            let mirrored = entry.join("compiled");
            let staging = entry.join("compiled.tmp");
            let _ = std::fs::remove_dir_all(&staging);
            copy_dir_recursive(&runtime_local, &staging)?;
            let _ = std::fs::remove_dir_all(&mirrored);
            std::fs::rename(staging, mirrored)?;
        }
        std::fs::write(
            entry.join("cache.meta"),
            format!("descriptor={hex}\nnative_cache_hit={}\n", cache_hit),
        )?;
    }

    Ok(AneModel {
        object,
        temp_dir,
        qos: options.qos,
        loaded: false,
        native_cache_hit: cache_hit,
        keep_temporary_files: options.keep_temporary_files || persistent.is_some(),
        _not_send_sync: PhantomData,
    })
}

impl AneModel {
    pub fn load(&mut self) -> Result<()> {
        if self.loaded {
            return Ok(());
        }
        let _pool = raw::AutoreleasePool::new();
        let options = raw::ns_dictionary(&[])?;
        let mut error: raw::Id = std::ptr::null_mut();
        let ok = unsafe {
            raw::msg_compile(
                self.object.as_ptr(),
                raw::selector("loadWithQoS:options:error:"),
                self.qos.0,
                options,
                &mut error,
            )
        };
        if !ok {
            return Err(AneError::ObjectiveC {
                operation: "load",
                message: raw::object_description(error),
            });
        }
        self.loaded = true;
        Ok(())
    }

    pub fn unload(&mut self) -> Result<()> {
        if !self.loaded {
            return Ok(());
        }
        let _pool = raw::AutoreleasePool::new();
        let mut error: raw::Id = std::ptr::null_mut();
        let ok = unsafe {
            raw::msg_unload(
                self.object.as_ptr(),
                raw::selector("unloadWithQoS:error:"),
                self.qos.0,
                &mut error,
            )
        };
        if !ok {
            return Err(AneError::ObjectiveC {
                operation: "unload",
                message: raw::object_description(error),
            });
        }
        self.loaded = false;
        Ok(())
    }

    /// Whether this compilation reused the private runtime's compiled model,
    /// rather than merely reusing Logan's generated MIL/weight inputs.
    pub fn native_cache_hit(&self) -> bool { self.native_cache_hit }

    pub fn is_loaded(&self) -> bool {
        self.loaded
    }

    pub fn state(&self) -> ModelState {
        let _pool = raw::AutoreleasePool::new();
        ModelState {
            private_state: unsafe { raw::msg0_u64(self.object.as_ptr(), raw::selector("state")) },
            program_handle: unsafe {
                raw::msg0_u64(self.object.as_ptr(), raw::selector("programHandle"))
            },
            intermediate_buffer_handle: unsafe {
                raw::msg0_u64(
                    self.object.as_ptr(),
                    raw::selector("intermediateBufferHandle"),
                )
            },
            queue_depth: unsafe { raw::msg0_i8(self.object.as_ptr(), raw::selector("queueDepth")) },
            perf_stats_mask: unsafe {
                raw::msg0_u32(self.object.as_ptr(), raw::selector("perfStatsMask"))
            },
        }
    }

    pub fn temporary_directory(&self) -> &std::path::Path {
        &self.temp_dir
    }

    /// Query the private ANE daemon's compiled-model cache for this model.
    ///
    /// This is intentionally best-effort/private-ABI guarded. A `false`
    /// result means Logan should compile normally; it does not imply that the
    /// model is invalid.
    pub fn compiled_model_exists(&self) -> Result<bool> {
        raw::require_instance_selector_encoding(
            CLS_MODEL,
            "compiledModelExists",
            "B16@0:8",
        )?;
        Ok(unsafe { raw::msg0_bool(self.object.as_ptr(), raw::selector("compiledModelExists")) })
    }

    /// Return the private runtime's local model path when available.
    pub fn local_model_path(&self) -> Result<Option<String>> {
        raw::require_instance_selector_encoding(CLS_MODEL, "localModelPath", "@16@0:8")?;
        let value = unsafe { raw::msg0_id(self.object.as_ptr(), raw::selector("localModelPath")) };
        if value.is_null() {
            Ok(None)
        } else {
            Ok(raw::nsstring_to_string(value).or_else(|| Some(raw::object_description(value))))
        }
    }

    pub fn request<'a>(
        &self,
        inputs: &[&'a AneSurface],
        outputs: &[&'a AneSurface],
    ) -> Result<AneRequest<'a>> {
        AneRequest::new(inputs, outputs, 0)
    }

    pub fn evaluate(&self, request: &AneRequest<'_>) -> Result<()> {
        if !self.loaded {
            return Err(AneError::InvalidArgument(
                "model must be loaded before evaluate".into(),
            ));
        }
        let _pool = raw::AutoreleasePool::new();
        let options = raw::ns_dictionary(&[])?;
        let mut error: raw::Id = std::ptr::null_mut();
        let ok = unsafe {
            raw::msg_evaluate(
                self.object.as_ptr(),
                raw::selector("evaluateWithQoS:options:request:error:"),
                self.qos.0,
                options,
                request.object.as_ptr(),
                &mut error,
            )
        };
        if ok {
            Ok(())
        } else {
            Err(AneError::ObjectiveC {
                operation: "evaluate",
                message: raw::object_description(error),
            })
        }
    }

    /// Build a reusable async request around fixed input/output IOSurfaces and
    /// one Metal-owned shared event. The request may optionally be pre-mapped
    /// through the private runtime's cache-inference path.
    ///
    /// # Safety
    /// `shared_event` must remain valid for the lifetime of the returned
    /// channel. In Logan this is guaranteed by the paired MetalAneFence field.
    pub unsafe fn async_channel(
        &self,
        inputs: &[&AneSurface],
        outputs: &[&AneSurface],
        procedure_index: u64,
        shared_event: *mut c_void,
        submit_mode: u8,
        premap: bool,
    ) -> Result<AneAsyncChannel> {
        if !self.loaded && submit_mode != 2 {
            return Err(AneError::InvalidArgument("model must be loaded before async channel creation".into()));
        }
        if inputs.is_empty() || outputs.is_empty() || shared_event.is_null() {
            return Err(AneError::InvalidArgument("invalid reusable ANE channel".into()));
        }
        let input_raw: Vec<*mut c_void> = inputs.iter().map(|s| s.raw_surface().cast()).collect();
        let output_raw: Vec<*mut c_void> = outputs.iter().map(|s| s.raw_surface().cast()).collect();
        let mut error = [0u8; 512];
        let raw = unsafe {
            logan_ane_async_channel_create(
                self.object.as_ptr(),
                input_raw.as_ptr(), input_raw.len(),
                output_raw.as_ptr(), output_raw.len(),
                procedure_index, shared_event, std::ptr::null_mut(), self.qos.0,
                submit_mode, u8::from(premap),
                error.as_mut_ptr().cast(), error.len(),
            )
        };
        let raw = std::ptr::NonNull::new(raw).ok_or_else(|| AneError::ObjectiveC {
            operation: "async channel create",
            message: async_error(&error),
        })?;
        Ok(AneAsyncChannel { raw, _not_send_sync: PhantomData })
    }

    /// Build a reusable ANE request with both a device-side wait event and a
    /// completion signal event. The events may share the same backing as long
    /// as callers use strictly increasing values.
    pub unsafe fn async_channel_wait_signal(
        &self, inputs: &[&AneSurface], outputs: &[&AneSurface], procedure_index: u64,
        shared_event: *mut c_void, wait_shared_event: *mut c_void,
        submit_mode: u8, premap: bool,
    ) -> Result<AneAsyncChannel> {
        if !self.loaded && submit_mode != 2 {
            return Err(AneError::InvalidArgument("model must be loaded before async channel creation".into()));
        }
        if inputs.is_empty() || outputs.is_empty() || shared_event.is_null() || wait_shared_event.is_null() {
            return Err(AneError::InvalidArgument("invalid reusable ANE wait/signal channel".into()));
        }
        let input_raw: Vec<*mut c_void> = inputs.iter().map(|s| s.raw_surface().cast()).collect();
        let output_raw: Vec<*mut c_void> = outputs.iter().map(|s| s.raw_surface().cast()).collect();
        let mut error = [0u8; 512];
        let raw = unsafe {
            logan_ane_async_channel_create(
                self.object.as_ptr(), input_raw.as_ptr(), input_raw.len(),
                output_raw.as_ptr(), output_raw.len(), procedure_index,
                shared_event, wait_shared_event, self.qos.0, submit_mode, u8::from(premap),
                error.as_mut_ptr().cast(), error.len(),
            )
        };
        let raw = std::ptr::NonNull::new(raw).ok_or_else(|| AneError::ObjectiveC {
            operation: "async wait/signal channel create", message: async_error(&error),
        })?;
        Ok(AneAsyncChannel { raw, _not_send_sync: PhantomData })
    }

    /// Submit an ANE request that signals a Metal-owned shared event on
    /// completion. `evaluateWithQoS` returns after enqueue when shared events
    /// are present; the returned owner must live until completion.
    ///
    /// # Safety
    /// `shared_event` must be a live `IOSurfaceSharedEvent` compatible with the
    /// running private ANE ABI and remain valid until the returned pending is
    /// finished or dropped.
    pub unsafe fn evaluate_async_signal(
        &self,
        inputs: &[&AneSurface],
        outputs: &[&AneSurface],
        procedure_index: u64,
        shared_event: *mut c_void,
        signal_value: u64,
        direct_client: bool,
    ) -> Result<AnePending> {
        if !self.loaded {
            return Err(AneError::InvalidArgument(
                "model must be loaded before async evaluate".into(),
            ));
        }
        if inputs.is_empty() || outputs.is_empty() || shared_event.is_null() || signal_value == 0 {
            return Err(AneError::InvalidArgument("invalid async ANE request".into()));
        }
        let input_raw: Vec<*mut c_void> = inputs.iter().map(|s| s.raw_surface().cast()).collect();
        let output_raw: Vec<*mut c_void> = outputs.iter().map(|s| s.raw_surface().cast()).collect();
        let mut error = [0u8; 512];
        let raw = unsafe {
            logan_ane_async_submit_signal(
                self.object.as_ptr(),
                input_raw.as_ptr(),
                input_raw.len(),
                output_raw.as_ptr(),
                output_raw.len(),
                procedure_index,
                shared_event,
                signal_value,
                self.qos.0,
                u8::from(direct_client),
                error.as_mut_ptr().cast(),
                error.len(),
            )
        };
        let raw = std::ptr::NonNull::new(raw).ok_or_else(|| AneError::ObjectiveC {
            operation: "async evaluate submit",
            message: async_error(&error),
        })?;
        Ok(AnePending { raw: Some(raw), _not_send_sync: PhantomData })
    }

    /// Explicitly asks the private runtime to map request IOSurfaces ahead of
    /// evaluation. Most ordinary evaluations do not require this; it is useful
    /// for probing cache-inference behavior and lower-level scheduling.
    pub fn map_request(&self, request: &AneRequest<'_>, cache_inference: bool) -> Result<()> {
        raw::require_instance_selector_encoding(
            CLS_MODEL,
            "mapIOSurfacesWithRequest:cacheInference:error:",
            "B36@0:8@16B24^@28",
        )?;
        let _pool = raw::AutoreleasePool::new();
        let mut error: raw::Id = std::ptr::null_mut();
        let ok = unsafe {
            raw::msg_map_request(
                self.object.as_ptr(),
                raw::selector("mapIOSurfacesWithRequest:cacheInference:error:"),
                request.object.as_ptr(),
                cache_inference,
                &mut error,
            )
        };
        if ok {
            Ok(())
        } else {
            Err(AneError::ObjectiveC {
                operation: "map IOSurfaces",
                message: raw::object_description(error),
            })
        }
    }

    pub fn unmap_request(&self, request: &AneRequest<'_>) -> Result<()> {
        raw::require_instance_selector_encoding(
            CLS_MODEL,
            "unmapIOSurfacesWithRequest:",
            "v24@0:8@16",
        )?;
        unsafe {
            raw::msg_unmap_request(
                self.object.as_ptr(),
                raw::selector("unmapIOSurfacesWithRequest:"),
                request.object.as_ptr(),
            )
        };
        Ok(())
    }

    /// Probe one private mutable-weight buffer ID for the compiled `main`
    /// procedure. This does not modify model data: the native shim maps,
    /// reports the byte size, and immediately unmaps while containing ObjC
    /// exceptions. Intended only for ABI/runtime discovery.
    pub fn probe_mutable_weight_buffer(&self, buffer_id: u64) -> Result<usize> {
        let mut size = 0u64;
        let mut error = [0u8; 512];
        let rc = unsafe {
            logan_ane_probe_mutable_buffer(
                self.object.as_ptr(), buffer_id, &mut size,
                error.as_mut_ptr().cast(), error.len(),
            )
        };
        if rc == 1 {
            usize::try_from(size).map_err(|_| AneError::InvalidArgument(
                format!("mutable ANE buffer too large: {size}")))
        } else {
            Err(AneError::ObjectiveC {
                operation: "probe mutable weight buffer",
                message: async_error(&error),
            })
        }
    }

    /// Borrow the underlying private `_ANEModel *` held by this in-memory
    /// wrapper. This is the object expected by lower-level `_ANEClient` APIs.
    ///
    /// # Safety
    ///
    /// The pointer is unowned and only valid while `self` lives. Its concrete
    /// private class/semantics may change across macOS releases.
    pub unsafe fn as_raw_underlying_model(&self) -> *mut c_void {
        unsafe { raw::msg0_id(self.object.as_ptr(), raw::selector("model")) }
    }

    /// Borrow the private `_ANEInMemoryModel *` for experiments that are not
    /// yet wrapped by this crate.
    ///
    /// # Safety
    ///
    /// The pointer is unowned and only valid while `self` lives. Private ObjC
    /// selector signatures are version-fragile; callers must use the exact ABI
    /// for the running OS.
    pub unsafe fn as_raw_object(&self) -> *mut c_void {
        self.object.as_ptr()
    }
}

impl Drop for AneModel {
    fn drop(&mut self) {
        if self.loaded {
            let _ = self.unload();
        }
        if !self.keep_temporary_files {
            let _ = std::fs::remove_dir_all(&self.temp_dir);
        }
    }
}

pub struct AneRequest<'a> {
    object: raw::Retained,
    _surfaces: PhantomData<&'a AneSurface>,
    _not_send_sync: PhantomData<Rc<()>>,
}

impl<'a> AneRequest<'a> {
    pub fn new(
        inputs: &[&'a AneSurface],
        outputs: &[&'a AneSurface],
        procedure_index: u64,
    ) -> Result<Self> {
        if inputs.is_empty() || outputs.is_empty() {
            return Err(AneError::InvalidArgument(
                "ANE request requires at least one input and one output".into(),
            ));
        }
        let input_indices: Vec<u64> = (0..inputs.len() as u64).collect();
        let output_indices: Vec<u64> = (0..outputs.len() as u64).collect();
        Self::with_indices(
            inputs,
            &input_indices,
            outputs,
            &output_indices,
            procedure_index,
        )
    }

    pub fn with_indices(
        inputs: &[&'a AneSurface],
        input_indices: &[u64],
        outputs: &[&'a AneSurface],
        output_indices: &[u64],
        procedure_index: u64,
    ) -> Result<Self> {
        if inputs.len() != input_indices.len() || outputs.len() != output_indices.len() {
            return Err(AneError::InvalidArgument(
                "request surface/index array lengths do not match".into(),
            ));
        }
        raw::require_class_selector_encoding(
            CLS_SURFACE_OBJECT,
            "objectWithIOSurface:",
            "@24@0:8^{__IOSurface=}16",
        )?;
        raw::require_class_selector_encoding(
            CLS_REQUEST,
            "requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:",
            "@72@0:8@16@24@32@40@48@56@64",
        )?;
        let _pool = raw::AutoreleasePool::new();
        let wrapper_cls = raw::class(CLS_SURFACE_OBJECT)?;
        let mut input_wrappers = Vec::with_capacity(inputs.len());
        for surface in inputs {
            let wrapper = unsafe {
                raw::msg1_id(
                    wrapper_cls,
                    raw::selector("objectWithIOSurface:"),
                    surface.raw_surface(),
                )
            };
            if wrapper.is_null() {
                return Err(AneError::NullResult("_ANEIOSurfaceObject input wrapper"));
            }
            input_wrappers.push(wrapper);
        }
        let mut output_wrappers = Vec::with_capacity(outputs.len());
        for surface in outputs {
            let wrapper = unsafe {
                raw::msg1_id(
                    wrapper_cls,
                    raw::selector("objectWithIOSurface:"),
                    surface.raw_surface(),
                )
            };
            if wrapper.is_null() {
                return Err(AneError::NullResult("_ANEIOSurfaceObject output wrapper"));
            }
            output_wrappers.push(wrapper);
        }
        let input_objects = raw::ns_array(&input_wrappers)?;
        let output_objects = raw::ns_array(&output_wrappers)?;
        let input_numbers: Vec<raw::Id> = input_indices
            .iter()
            .map(|&v| raw::ns_number(v))
            .collect::<Result<_>>()?;
        let output_numbers: Vec<raw::Id> = output_indices
            .iter()
            .map(|&v| raw::ns_number(v))
            .collect::<Result<_>>()?;
        let input_index_array = raw::ns_array(&input_numbers)?;
        let output_index_array = raw::ns_array(&output_numbers)?;
        let procedure = raw::ns_number(procedure_index)?;
        let request_cls = raw::class(CLS_REQUEST)?;
        let request = unsafe {
            raw::msg7_id(
                request_cls,
                raw::selector(
                    "requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:",
                ),
                input_objects,
                input_index_array,
                output_objects,
                output_index_array,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                procedure,
            )
        };
        Ok(Self {
            object: raw::Retained::new(request, "_ANERequest creation")?,
            _surfaces: PhantomData,
            _not_send_sync: PhantomData,
        })
    }

    /// Borrow the private `_ANERequest *`.
    ///
    /// # Safety
    ///
    /// The pointer is unowned and valid only for the lifetime of this request.
    pub unsafe fn as_raw_object(&self) -> *mut c_void {
        self.object.as_ptr()
    }
}
