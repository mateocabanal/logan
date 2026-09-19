//! Model-family and protocol selection for chat frontends.
//!
//! Selection is intentionally independent from model loading.  The existing
//! Qwen engine remains the default execution path; MiniCPM5 gets an explicit,
//! plain-data protocol adapter so callers cannot accidentally reuse Qwen
//! prefixes or state snapshots.

use crate::openai::ApiMessage;
use crate::protocol::minicpm5::{
    MiniCpm5Delta, MiniCpm5StreamParser, MiniCpm5TemplateOptions, ThinkingMode, ToolDefinition,
    MINICPM5_EOS_IDS,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelFamily {
    Qwen4,
    MiniCpm5,
}

impl ModelFamily {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Qwen4 => "qwen4",
            Self::MiniCpm5 => "minicpm5",
        }
    }

    pub fn eos_ids(self) -> &'static [u32] {
        match self {
            Self::Qwen4 => &[],
            Self::MiniCpm5 => &MINICPM5_EOS_IDS,
        }
    }
}

/// Select a family from an explicit model id.  Unknown ids retain the legacy
/// Qwen route for compatibility; MiniCPM5 is opt-in and never inferred from a
/// Qwen package's protocol markers.
pub fn select_model_family(model: Option<&str>) -> ModelFamily {
    let Some(model) = model else {
        return ModelFamily::Qwen4;
    };
    let normalized = model.to_ascii_lowercase();
    if normalized.contains("minicpm5") || normalized.contains("mini-cpm5") {
        ModelFamily::MiniCpm5
    } else {
        ModelFamily::Qwen4
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum PromptAdapter {
    Qwen,
    MiniCpm5(MiniCpm5TemplateOptions),
}

impl PromptAdapter {
    pub fn for_family(family: ModelFamily) -> Self {
        match family {
            ModelFamily::Qwen4 => Self::Qwen,
            ModelFamily::MiniCpm5 => Self::MiniCpm5(MiniCpm5TemplateOptions::default()),
        }
    }

    pub fn family(&self) -> ModelFamily {
        match self {
            Self::Qwen => ModelFamily::Qwen4,
            Self::MiniCpm5(_) => ModelFamily::MiniCpm5,
        }
    }
}

/// Public request-level hooks used by OpenAI and other frontends.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProtocolOptions {
    pub thinking: Option<bool>,
    pub tools: Vec<ToolDefinition>,
}

impl ProtocolOptions {
    pub fn mini_cpm5(&self) -> MiniCpm5TemplateOptions {
        MiniCpm5TemplateOptions {
            thinking: if self.thinking.unwrap_or(false) {
                ThinkingMode::On
            } else {
                ThinkingMode::Off
            },
            tools: self.tools.clone(),
        }
    }
}

pub fn prompt_adapter(family: ModelFamily, options: &ProtocolOptions) -> PromptAdapter {
    match family {
        ModelFamily::Qwen4 => PromptAdapter::Qwen,
        ModelFamily::MiniCpm5 => PromptAdapter::MiniCpm5(options.mini_cpm5()),
    }
}

pub fn render_prompt(
    adapter: &PromptAdapter,
    messages: &[ApiMessage],
    instructions: Option<&str>,
) -> Result<String, String> {
    match adapter {
        PromptAdapter::Qwen => crate::openai::render_qwen_prompt(messages, instructions),
        PromptAdapter::MiniCpm5(options) => {
            let mut messages = messages.to_vec();
            if let Some(instructions) = instructions.filter(|value| !value.trim().is_empty()) {
                messages.insert(
                    0,
                    ApiMessage {
                        role: "system".into(),
                        text: instructions.into(),
                    },
                );
            }
            crate::protocol::minicpm5::render_prompt(&messages, options)
        }
    }
}

pub struct StreamAdapter {
    family: ModelFamily,
    mini_cpm5: Option<MiniCpm5StreamParser>,
}

impl StreamAdapter {
    pub fn new(family: ModelFamily) -> Self {
        Self {
            family,
            mini_cpm5: (family == ModelFamily::MiniCpm5).then(MiniCpm5StreamParser::new),
        }
    }

    pub fn family(&self) -> ModelFamily {
        self.family
    }

    pub fn feed(&mut self, bytes: &[u8]) -> Result<Vec<MiniCpm5Delta>, String> {
        match self.mini_cpm5.as_mut() {
            Some(parser) => parser.feed(bytes).map_err(|error| error.to_string()),
            None => Ok(vec![MiniCpm5Delta::Text(
                String::from_utf8(bytes.to_vec()).map_err(|_| "stream is not UTF-8".to_string())?,
            )]),
        }
    }

    pub fn finish(&mut self) -> Result<Vec<MiniCpm5Delta>, String> {
        match self.mini_cpm5.as_mut() {
            Some(parser) => parser.finish().map_err(|error| error.to_string()),
            None => Ok(Vec::new()),
        }
    }
}

/// Dense execution lives in `engine` so it can share generation settings while
/// retaining a distinct handle from the Qwen worker. Re-exporting the adapter
/// here keeps family selection and execution discoverable together.
pub use crate::engine::DenseMiniCpm;
