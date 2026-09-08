use std::marker::PhantomData;
use std::path::PathBuf;

use crate::{AneError, MilProgram, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct AneQos(pub u32);

#[derive(Clone, Debug, Default)]
pub struct CompileOptions {
    pub qos: AneQos,
    pub keep_temporary_files: bool,
    pub cache_directory: Option<PathBuf>,
    pub reuse_compiled_model: bool,
}

#[derive(Clone, Debug, Default)]
pub struct RuntimeCapabilities;

#[derive(Clone, Debug, Default)]
pub struct AneDeviceInfo;

#[derive(Clone, Copy, Debug, Default)]
pub struct ModelState;

#[derive(Clone, Debug, Default)]
pub struct AneRuntime;

impl AneRuntime {
    pub fn load() -> Result<Self> {
        Err(AneError::UnsupportedPlatform)
    }

    pub fn capabilities(&self) -> &RuntimeCapabilities {
        static CAPS: RuntimeCapabilities = RuntimeCapabilities;
        &CAPS
    }

    pub fn device_info(&self) -> &AneDeviceInfo {
        static INFO: AneDeviceInfo = AneDeviceInfo;
        &INFO
    }

    pub fn compile(&self, _program: &MilProgram, _options: CompileOptions) -> Result<AneModel> {
        Err(AneError::UnsupportedPlatform)
    }

    pub fn shared_client(&self) -> Result<AneClient> {
        Err(AneError::UnsupportedPlatform)
    }
}

pub struct AneSurface;
impl AneSurface {
    pub fn new(_bytes: usize) -> Result<Self> {
        Err(AneError::UnsupportedPlatform)
    }
}

pub struct SurfaceRead<'a>(PhantomData<&'a ()>);
pub struct SurfaceWrite<'a>(PhantomData<&'a mut ()>);
pub struct AneClient;
pub struct AneModel;
pub struct AneRequest<'a>(PhantomData<&'a ()>);
pub struct MutableWeightMapping<'a>(PhantomData<&'a ()>);
