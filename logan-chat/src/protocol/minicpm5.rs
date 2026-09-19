//! MiniCPM5 prompt and streaming protocol.
//!
//! This module deliberately contains no model or native-runtime state.  It is a
//! plain-data adapter that can be used by HTTP handlers and by an inference
//! worker without moving native objects between threads.

use std::collections::BTreeMap;
use std::fmt;

use serde_json::{Map, Value};

/// MiniCPM5 exposes both the ordinary EOS token and an end-of-turn token.
/// Callers may override these for a tokenizer package whose ids differ.
pub const MINICPM5_EOS_IDS: [u32; 2] = [2, 73440];

pub fn is_minicpm5_eos(token_id: u32) -> bool {
    MINICPM5_EOS_IDS.contains(&token_id)
}

pub type MiniCpm5Parser = MiniCpm5StreamParser;
pub type MiniCpm5Event = MiniCpm5Delta;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThinkingMode {
    Off,
    On,
}

impl Default for ThinkingMode {
    fn default() -> Self {
        Self::Off
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolDefinition {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Value,
}

impl ToolDefinition {
    pub fn new(name: impl Into<String>, parameters: Value) -> Self {
        Self {
            name: name.into(),
            description: None,
            parameters,
        }
    }

    pub fn from_openai(value: &Value) -> Result<Self, String> {
        let object = value
            .as_object()
            .ok_or_else(|| "tool definition must be an object".to_string())?;
        let function = object.get("function").unwrap_or(value);
        let function = function
            .as_object()
            .ok_or_else(|| "tool function must be an object".to_string())?;
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| !name.trim().is_empty())
            .ok_or_else(|| "tool function is missing a name".to_string())?;
        let parameters = function
            .get("parameters")
            .cloned()
            .unwrap_or_else(|| Value::Object(Map::new()));
        Ok(Self {
            name: name.to_string(),
            description: function
                .get("description")
                .and_then(Value::as_str)
                .map(str::to_string),
            parameters,
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MiniCpm5TemplateOptions {
    pub thinking: ThinkingMode,
    pub tools: Vec<ToolDefinition>,
}

/// Render MiniCPM5's ChatML-compatible prompt.  The assistant marker is
/// intentionally selected here, rather than borrowing Qwen's empty-think
/// prefix.
pub fn render_prompt(
    messages: &[crate::openai::ApiMessage],
    options: &MiniCpm5TemplateOptions,
) -> Result<String, String> {
    if messages.is_empty() {
        return Err("request has no input messages".into());
    }
    let mut out = String::new();
    for message in messages {
        let role = match message.role.as_str() {
            "developer" => "system",
            "system" | "user" | "assistant" => message.role.as_str(),
            role => return Err(format!("unsupported message role: {role}")),
        };
        push_message(&mut out, role, &message.text);
    }
    if !options.tools.is_empty() {
        let tool_text = render_tools(&options.tools)?;
        push_message(&mut out, "system", &tool_text);
    }
    out.push_str("<|im_start|>assistant\n");
    if options.thinking == ThinkingMode::On {
        out.push_str("<think>\n");
    }
    Ok(out)
}

/// Hook used by request adapters that need to render only a tool preamble.
pub fn render_tools(tools: &[ToolDefinition]) -> Result<String, String> {
    if tools.is_empty() {
        return Ok(String::new());
    }
    let mut out = String::from("Available tools:\n");
    for tool in tools {
        if tool.name.trim().is_empty() {
            return Err("tool name must not be empty".into());
        }
        out.push_str("- ");
        out.push_str(&tool.name);
        if let Some(description) = &tool.description {
            out.push_str(": ");
            out.push_str(description);
        }
        out.push('\n');
        out.push_str("  parameters: ");
        out.push_str(&serde_json::to_string(&tool.parameters).map_err(|e| e.to_string())?);
        out.push('\n');
    }
    out.push_str(
        "Call a tool with <function name=\"...\"><param name=\"...\">...</param></function>.\n",
    );
    Ok(out)
}

fn push_message(out: &mut String, role: &str, text: &str) {
    out.push_str("<|im_start|>");
    out.push_str(role);
    out.push('\n');
    out.push_str(text);
    out.push_str("<|im_end|>\n");
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MiniCpm5Delta {
    Text(String),
    Reasoning(String),
    ToolCallStarted { name: String },
    ToolArgument { name: String, value: String },
    ToolCallFinished(ToolCall),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MiniCpm5ParseError {
    Malformed(String),
    Truncated(String),
    InvalidUtf8,
    ExternalEntity,
}

impl fmt::Display for MiniCpm5ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Malformed(message) => write!(f, "malformed MiniCPM5 output: {message}"),
            Self::Truncated(message) => write!(f, "truncated MiniCPM5 output: {message}"),
            Self::InvalidUtf8 => f.write_str("MiniCPM5 output is not valid UTF-8"),
            Self::ExternalEntity => f.write_str("external XML entities are disabled"),
        }
    }
}

impl std::error::Error for MiniCpm5ParseError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Text,
    Reasoning,
    Function,
    Param,
}

/// Incremental, non-executing parser for MiniCPM5's XML function-call stream.
///
/// `feed` may receive arbitrary byte slices, including a split UTF-8 scalar or
/// a split XML delimiter.  Calls are emitted only after their closing element;
/// malformed and truncated calls never produce an executable completion.
pub struct MiniCpm5StreamParser {
    pending: Vec<u8>,
    state: State,
    function_name: Option<String>,
    arguments: BTreeMap<String, String>,
    param_name: Option<String>,
    finished: bool,
}

impl Default for MiniCpm5StreamParser {
    fn default() -> Self {
        Self::new()
    }
}

impl MiniCpm5StreamParser {
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
            state: State::Text,
            function_name: None,
            arguments: BTreeMap::new(),
            param_name: None,
            finished: false,
        }
    }

    pub fn feed(&mut self, bytes: &[u8]) -> Result<Vec<MiniCpm5Delta>, MiniCpm5ParseError> {
        if self.finished {
            return Err(MiniCpm5ParseError::Malformed("input after finish".into()));
        }
        self.pending.extend_from_slice(bytes);
        self.parse(false)
    }

    pub fn finish(&mut self) -> Result<Vec<MiniCpm5Delta>, MiniCpm5ParseError> {
        if self.finished {
            return Ok(Vec::new());
        }
        self.finished = true;
        let mut deltas = self.parse(true)?;
        if self.state != State::Text {
            return Err(MiniCpm5ParseError::Truncated(
                match self.state {
                    State::Reasoning => "reasoning block is not closed",
                    State::Function => "function call is not closed",
                    State::Param => "parameter is not closed",
                    State::Text => unreachable!(),
                }
                .into(),
            ));
        }
        if !self.pending.is_empty() {
            let text = take_utf8(&mut self.pending)?;
            if !text.is_empty() {
                deltas.push(MiniCpm5Delta::Text(text));
            }
        }
        Ok(deltas)
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    fn parse(&mut self, final_input: bool) -> Result<Vec<MiniCpm5Delta>, MiniCpm5ParseError> {
        let mut out = Vec::new();
        loop {
            if self.pending.is_empty() {
                break;
            }
            match self.state {
                State::Text | State::Reasoning => {
                    let reasoning = self.state == State::Reasoning;
                    let tag = if reasoning { "</think>" } else { "<think>" };
                    let alt = if reasoning {
                        "</reasoning>"
                    } else {
                        "<reasoning>"
                    };
                    let Some(mark) = find_byte(&self.pending, b'<') else {
                        if final_input {
                            let text = take_utf8(&mut self.pending)?;
                            if !text.is_empty() {
                                push_text(&mut out, reasoning, text);
                            }
                        } else if let Some(prefix_len) = safe_text_prefix(&self.pending) {
                            let bytes = self.pending.drain(..prefix_len).collect::<Vec<_>>();
                            let text = String::from_utf8(bytes)
                                .map_err(|_| MiniCpm5ParseError::InvalidUtf8)?;
                            if !text.is_empty() {
                                push_text(&mut out, reasoning, text);
                            }
                        }
                        break;
                    };
                    if mark > 0 {
                        let bytes = self.pending.drain(..mark).collect::<Vec<_>>();
                        let text = String::from_utf8(bytes)
                            .map_err(|_| MiniCpm5ParseError::InvalidUtf8)?;
                        push_text(&mut out, reasoning, text);
                        continue;
                    }
                    if reasoning {
                        if self.pending.starts_with(tag.as_bytes()) {
                            self.pending.drain(..tag.len());
                            self.state = State::Text;
                            continue;
                        }
                        if self.pending.starts_with(alt.as_bytes()) {
                            self.pending.drain(..alt.len());
                            self.state = State::Text;
                            continue;
                        }
                    } else {
                        if self.pending.starts_with(tag.as_bytes()) {
                            self.pending.drain(..tag.len());
                            self.state = State::Reasoning;
                            continue;
                        }
                        if self.pending.starts_with(alt.as_bytes()) {
                            self.pending.drain(..alt.len());
                            self.state = State::Reasoning;
                            continue;
                        }
                    }
                    if !reasoning && self.pending.starts_with(b"<!DOCTYPE")
                        || !reasoning && self.pending.starts_with(b"<!ENTITY")
                        || !reasoning && self.pending.starts_with(b"<?xml")
                    {
                        return Err(MiniCpm5ParseError::ExternalEntity);
                    }
                    if !reasoning
                        && (is_prefix_of(&self.pending, b"<!DOCTYPE")
                            || is_prefix_of(&self.pending, b"<!ENTITY")
                            || is_prefix_of(&self.pending, b"<?xml"))
                    {
                        break;
                    }
                    if !reasoning && self.pending.starts_with(b"<function") {
                        let Some(end) = find_bytes(&self.pending, b">") else {
                            if final_input {
                                return Err(MiniCpm5ParseError::Truncated(
                                    "function opening tag is not closed".into(),
                                ));
                            }
                            break;
                        };
                        let raw = self.pending.drain(..=end).collect::<Vec<_>>();
                        let name = parse_function_open(&raw)?;
                        self.function_name = Some(name.clone());
                        self.arguments.clear();
                        self.state = State::Function;
                        out.push(MiniCpm5Delta::ToolCallStarted { name });
                        continue;
                    }
                    let known_prefix = [tag.as_bytes(), alt.as_bytes(), b"<function"]
                        .iter()
                        .any(|prefix| is_prefix_of(&self.pending, prefix));
                    if known_prefix {
                        break;
                    }
                    // A literal '<' is ordinary text unless it starts a known
                    // protocol construct.  This keeps prose such as `a < b`
                    // streamable while malformed function markup is rejected.
                    let byte = self.pending.remove(0);
                    let text = String::from_utf8(vec![byte])
                        .map_err(|_| MiniCpm5ParseError::InvalidUtf8)?;
                    push_text(&mut out, reasoning, text);
                }
                State::Function => {
                    trim_ascii_prefix(&mut self.pending);
                    if self.pending.starts_with(b"</function>") {
                        self.pending.drain(.."</function>".len());
                        let name = self.function_name.take().ok_or_else(|| {
                            MiniCpm5ParseError::Malformed("function name missing".into())
                        })?;
                        out.push(MiniCpm5Delta::ToolCallFinished(ToolCall {
                            name,
                            arguments: std::mem::take(&mut self.arguments),
                        }));
                        self.state = State::Text;
                        continue;
                    }
                    if self.pending.starts_with(b"<param") {
                        let Some(end) = find_bytes(&self.pending, b">") else {
                            break;
                        };
                        let raw = self.pending.drain(..=end).collect::<Vec<_>>();
                        self.param_name = Some(parse_param_open(&raw)?);
                        self.state = State::Param;
                        continue;
                    }
                    if is_prefix_of(&self.pending, b"</function>")
                        || is_prefix_of(&self.pending, b"<param")
                    {
                        break;
                    }
                    return Err(MiniCpm5ParseError::Malformed(
                        "expected <param> or </function>".into(),
                    ));
                }
                State::Param => {
                    let close = b"</param>";
                    let Some(end) = find_bytes(&self.pending, close) else {
                        if final_input {
                            return Err(MiniCpm5ParseError::Truncated(
                                "parameter is not closed".into(),
                            ));
                        }
                        break;
                    };
                    let raw = self.pending.drain(..end).collect::<Vec<_>>();
                    self.pending.drain(..close.len());
                    let value = parse_param_value(&raw)?;
                    let name = self.param_name.take().ok_or_else(|| {
                        MiniCpm5ParseError::Malformed("parameter name missing".into())
                    })?;
                    if self.arguments.insert(name.clone(), value.clone()).is_some() {
                        return Err(MiniCpm5ParseError::Malformed(format!(
                            "duplicate parameter {name:?}"
                        )));
                    }
                    out.push(MiniCpm5Delta::ToolArgument { name, value });
                    self.state = State::Function;
                }
            }
        }
        Ok(out)
    }
}

fn push_text(out: &mut Vec<MiniCpm5Delta>, reasoning: bool, text: String) {
    if text.is_empty() {
        return;
    }
    let delta = if reasoning {
        MiniCpm5Delta::Reasoning(text)
    } else {
        MiniCpm5Delta::Text(text)
    };
    out.push(delta);
}

fn find_byte(bytes: &[u8], byte: u8) -> Option<usize> {
    bytes.iter().position(|candidate| *candidate == byte)
}

fn find_bytes(bytes: &[u8], needle: &[u8]) -> Option<usize> {
    bytes
        .windows(needle.len())
        .position(|window| window == needle)
}

fn is_prefix_of(bytes: &[u8], prefix: &[u8]) -> bool {
    bytes.len() < prefix.len() && prefix.starts_with(bytes)
}

fn safe_text_prefix(bytes: &[u8]) -> Option<usize> {
    match std::str::from_utf8(bytes) {
        Ok(_) => Some(bytes.len()),
        Err(error) => Some(error.valid_up_to()),
    }
}

fn take_utf8(bytes: &mut Vec<u8>) -> Result<String, MiniCpm5ParseError> {
    let out = std::mem::take(bytes);
    String::from_utf8(out).map_err(|_| MiniCpm5ParseError::InvalidUtf8)
}

fn trim_ascii_prefix(bytes: &mut Vec<u8>) {
    let count = bytes
        .iter()
        .take_while(|byte| byte.is_ascii_whitespace())
        .count();
    bytes.drain(..count);
}

fn parse_function_open(raw: &[u8]) -> Result<String, MiniCpm5ParseError> {
    let text = std::str::from_utf8(raw).map_err(|_| MiniCpm5ParseError::InvalidUtf8)?;
    let attrs = parse_tag(text, "function")?;
    attrs
        .get("name")
        .cloned()
        .filter(|name| !name.is_empty())
        .ok_or_else(|| MiniCpm5ParseError::Malformed("function name is missing".into()))
}

fn parse_param_open(raw: &[u8]) -> Result<String, MiniCpm5ParseError> {
    let text = std::str::from_utf8(raw).map_err(|_| MiniCpm5ParseError::InvalidUtf8)?;
    let attrs = parse_tag(text, "param")?;
    attrs
        .get("name")
        .cloned()
        .filter(|name| !name.is_empty())
        .ok_or_else(|| MiniCpm5ParseError::Malformed("parameter name is missing".into()))
}

fn parse_tag(text: &str, expected: &str) -> Result<BTreeMap<String, String>, MiniCpm5ParseError> {
    if text.contains("<!DOCTYPE") || text.contains("<!ENTITY") || text.contains("<?xml") {
        return Err(MiniCpm5ParseError::ExternalEntity);
    }
    let inner = text
        .strip_prefix('<')
        .and_then(|value| value.strip_suffix('>'))
        .ok_or_else(|| MiniCpm5ParseError::Malformed("unterminated tag".into()))?;
    let inner = inner.strip_suffix('/').unwrap_or(inner).trim();
    let (name, rest) = inner.split_once(char::is_whitespace).unwrap_or((inner, ""));
    if name != expected {
        return Err(MiniCpm5ParseError::Malformed(format!(
            "expected <{expected}>"
        )));
    }
    let mut attrs = BTreeMap::new();
    let mut rest = rest.trim();
    while !rest.is_empty() {
        let key_end = rest
            .find(|character: char| character.is_ascii_whitespace() || character == '=')
            .unwrap_or(rest.len());
        let key = &rest[..key_end];
        if key.is_empty() {
            return Err(MiniCpm5ParseError::Malformed(
                "attribute name missing".into(),
            ));
        }
        rest = rest[key_end..].trim_start();
        if !rest.starts_with('=') {
            return Err(MiniCpm5ParseError::Malformed("attribute must use =".into()));
        }
        rest = rest[1..].trim_start();
        let quote = rest
            .as_bytes()
            .first()
            .copied()
            .ok_or_else(|| MiniCpm5ParseError::Malformed("attribute value missing".into()))?
            as char;
        if quote != '"' && quote != '\'' {
            return Err(MiniCpm5ParseError::Malformed(
                "attribute values must be quoted".into(),
            ));
        }
        rest = &rest[1..];
        let end = rest
            .find(quote)
            .ok_or_else(|| MiniCpm5ParseError::Malformed("unterminated attribute".into()))?;
        let value = decode_entities(&rest[..end])?;
        rest = rest[end + 1..].trim_start();
        if attrs.insert(key.to_string(), value).is_some() {
            return Err(MiniCpm5ParseError::Malformed("duplicate attribute".into()));
        }
    }
    Ok(attrs)
}

fn parse_param_value(raw: &[u8]) -> Result<String, MiniCpm5ParseError> {
    let text = std::str::from_utf8(raw).map_err(|_| MiniCpm5ParseError::InvalidUtf8)?;
    if text.contains("<!DOCTYPE") || text.contains("<!ENTITY") || text.contains("<?xml") {
        return Err(MiniCpm5ParseError::ExternalEntity);
    }
    let text = text.trim();
    if let Some(cdata) = text.strip_prefix("<![CDATA[") {
        let cdata = cdata
            .strip_suffix("]]>")
            .ok_or_else(|| MiniCpm5ParseError::Malformed("unterminated CDATA".into()))?;
        if cdata.contains("]]>") {
            return Err(MiniCpm5ParseError::Malformed(
                "nested CDATA terminator".into(),
            ));
        }
        return Ok(cdata.to_string());
    }
    if text.contains("<![CDATA[") || text.contains('<') {
        return Err(MiniCpm5ParseError::Malformed(
            "parameter contains unsupported markup".into(),
        ));
    }
    decode_entities(text)
}

fn decode_entities(text: &str) -> Result<String, MiniCpm5ParseError> {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find('&') {
        out.push_str(&rest[..start]);
        let end = rest[start..]
            .find(';')
            .ok_or_else(|| MiniCpm5ParseError::Malformed("unterminated entity".into()))?
            + start;
        let entity = &rest[start + 1..end];
        let value = match entity {
            "amp" => "&",
            "lt" => "<",
            "gt" => ">",
            "quot" => "\"",
            "apos" => "'",
            _ => return Err(MiniCpm5ParseError::ExternalEntity),
        };
        out.push_str(value);
        rest = &rest[end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}
