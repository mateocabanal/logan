use serde::Deserialize;
use serde_json::Value;

use crate::engine::GenerationSettings;

pub const ASSISTANT_NON_THINKING_PREFIX: &str = "<|im_start|>assistant\n<think>\n\n</think>\n\n";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApiMessage {
    pub role: String,
    pub text: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Value,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Value>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct ChatStreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<ChatStreamOptions>,
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub n: Option<usize>,
    #[serde(default)]
    pub stop: Option<Value>,
    #[serde(default)]
    pub tools: Option<Value>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub response_format: Option<Value>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ResponsesRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub input: Option<Value>,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub max_output_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub previous_response_id: Option<String>,
    #[serde(default)]
    pub background: Option<bool>,
    #[serde(default)]
    pub tools: Option<Value>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub text: Option<Value>,
    #[serde(default)]
    pub metadata: Option<Value>,
    #[serde(default)]
    pub store: Option<bool>,
}

pub fn normalize_chat_messages(messages: &[ChatMessage]) -> Result<Vec<ApiMessage>, String> {
    if messages.is_empty() {
        return Err("messages must contain at least one message".into());
    }
    messages
        .iter()
        .map(|message| {
            let role = normalize_role(&message.role)?;
            if message.tool_calls.as_ref().is_some_and(nonempty_json) {
                return Err("tool calls are not supported by this Logan model server".into());
            }
            let text = extract_text_content(&message.content, false)?;
            if text.is_empty() && role != "assistant" {
                return Err(format!("{role} message content is empty"));
            }
            Ok(ApiMessage {
                role: role.to_string(),
                text,
            })
        })
        .collect()
}

pub fn normalize_responses_input(input: Option<&Value>) -> Result<Vec<ApiMessage>, String> {
    let Some(input) = input else {
        return Ok(Vec::new());
    };
    match input {
        Value::String(text) => Ok(vec![ApiMessage {
            role: "user".into(),
            text: text.clone(),
        }]),
        Value::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                let object = item.as_object().ok_or_else(|| {
                    "Responses input array items must be message objects".to_string()
                })?;
                let item_type = object.get("type").and_then(Value::as_str);
                if item_type.is_some_and(|kind| kind != "message") {
                    return Err(format!(
                        "Responses input item type {item_type:?} is not supported; Logan currently accepts text messages"
                    ));
                }
                let role = object
                    .get("role")
                    .and_then(Value::as_str)
                    .ok_or_else(|| "Responses input message is missing role".to_string())?;
                let role = normalize_role(role)?;
                let content = object
                    .get("content")
                    .ok_or_else(|| "Responses input message is missing content".to_string())?;
                let text = extract_text_content(content, true)?;
                out.push(ApiMessage {
                    role: role.to_string(),
                    text,
                });
            }
            Ok(out)
        }
        _ => Err("Responses input must be a string or an array of text messages".into()),
    }
}

pub fn render_qwen_prompt(
    messages: &[ApiMessage],
    instructions: Option<&str>,
) -> Result<String, String> {
    if messages.is_empty() {
        return Err("request has no input messages".into());
    }
    let mut out = String::new();
    if let Some(instructions) = instructions.filter(|value| !value.trim().is_empty()) {
        push_message(&mut out, "system", instructions);
    }
    for message in messages {
        match message.role.as_str() {
            "system" | "developer" => push_message(&mut out, "system", &message.text),
            "user" => push_message(&mut out, "user", &message.text),
            "assistant" => {
                out.push_str(ASSISTANT_NON_THINKING_PREFIX);
                out.push_str(&message.text);
                out.push_str("<|im_end|>\n");
            }
            role => return Err(format!("unsupported message role: {role}")),
        }
    }
    out.push_str(ASSISTANT_NON_THINKING_PREFIX);
    Ok(out)
}

pub fn settings_from_chat(request: &ChatCompletionRequest) -> Result<GenerationSettings, String> {
    if request.n.unwrap_or(1) != 1 {
        return Err("Logan currently supports n=1 only".into());
    }
    reject_stop(request.stop.as_ref())?;
    reject_tools(request.tools.as_ref(), request.tool_choice.as_ref())?;
    reject_chat_response_format(request.response_format.as_ref())?;
    let mut settings = GenerationSettings::default();
    settings.max_new = request
        .max_completion_tokens
        .or(request.max_tokens)
        .unwrap_or(settings.max_new)
        .clamp(1, 65_536);
    if let Some(value) = request.temperature {
        if !(0.0..=2.0).contains(&value) {
            return Err("temperature must be in 0..=2".into());
        }
        settings.temperature = value;
    }
    if let Some(value) = request.top_p {
        if !(0.01..=1.0).contains(&value) {
            return Err("top_p must be in 0.01..=1".into());
        }
        settings.top_p = value;
    }
    if let Some(value) = request.top_k {
        settings.top_k = value;
    }
    Ok(settings)
}

pub fn settings_from_responses(request: &ResponsesRequest) -> Result<GenerationSettings, String> {
    if request.background == Some(true) {
        return Err("background Responses are not supported by Logan".into());
    }
    reject_tools(request.tools.as_ref(), request.tool_choice.as_ref())?;
    reject_responses_text_format(request.text.as_ref())?;
    let mut settings = GenerationSettings::default();
    settings.max_new = request
        .max_output_tokens
        .unwrap_or(settings.max_new)
        .clamp(1, 65_536);
    if let Some(value) = request.temperature {
        if !(0.0..=2.0).contains(&value) {
            return Err("temperature must be in 0..=2".into());
        }
        settings.temperature = value;
    }
    if let Some(value) = request.top_p {
        if !(0.01..=1.0).contains(&value) {
            return Err("top_p must be in 0.01..=1".into());
        }
        settings.top_p = value;
    }
    if let Some(value) = request.top_k {
        settings.top_k = value;
    }
    Ok(settings)
}

fn normalize_role(role: &str) -> Result<&'static str, String> {
    match role {
        "developer" => Ok("developer"),
        "system" => Ok("system"),
        "user" => Ok("user"),
        "assistant" => Ok("assistant"),
        "tool" | "function" => Err("tool/function messages are not supported by Logan yet".into()),
        other => Err(format!("unsupported message role: {other}")),
    }
}

fn push_message(out: &mut String, role: &str, text: &str) {
    out.push_str("<|im_start|>");
    out.push_str(role);
    out.push('\n');
    out.push_str(text);
    out.push_str("<|im_end|>\n");
}

fn extract_text_content(content: &Value, responses: bool) -> Result<String, String> {
    match content {
        Value::String(text) => Ok(text.clone()),
        Value::Null => Ok(String::new()),
        Value::Array(parts) => {
            let mut out = String::new();
            for part in parts {
                let object = part
                    .as_object()
                    .ok_or_else(|| "message content parts must be objects".to_string())?;
                let kind = object.get("type").and_then(Value::as_str).unwrap_or("text");
                let text_kind = kind == "text"
                    || (responses && (kind == "input_text" || kind == "output_text"));
                if !text_kind {
                    return Err(format!(
                        "content part type {kind:?} is not supported; this Logan server is text-only"
                    ));
                }
                let text = object
                    .get("text")
                    .and_then(Value::as_str)
                    .ok_or_else(|| format!("content part {kind:?} is missing text"))?;
                out.push_str(text);
            }
            Ok(out)
        }
        _ => Err("message content must be a string or text content-part array".into()),
    }
}

fn reject_stop(stop: Option<&Value>) -> Result<(), String> {
    if stop.is_some_and(|value| !value.is_null()) {
        return Err("custom stop sequences are not supported by Logan yet".into());
    }
    Ok(())
}

fn reject_tools(tools: Option<&Value>, tool_choice: Option<&Value>) -> Result<(), String> {
    if tools.is_some_and(nonempty_json) {
        return Err(
            "tools/function calling are not supported by this Logan model server yet".into(),
        );
    }
    if let Some(choice) = tool_choice {
        let allowed = choice.is_null() || choice.as_str().is_some_and(|value| value == "none");
        if !allowed {
            return Err("tool_choice other than 'none' is not supported by Logan yet".into());
        }
    }
    Ok(())
}

fn reject_chat_response_format(format: Option<&Value>) -> Result<(), String> {
    let Some(format) = format else {
        return Ok(());
    };
    let kind = format.get("type").and_then(Value::as_str).unwrap_or("text");
    if kind != "text" {
        return Err(format!(
            "response_format type {kind:?} is not supported; Logan currently returns text"
        ));
    }
    Ok(())
}

fn reject_responses_text_format(text: Option<&Value>) -> Result<(), String> {
    let Some(text) = text else {
        return Ok(());
    };
    let kind = text
        .get("format")
        .and_then(|format| format.get("type"))
        .and_then(Value::as_str)
        .unwrap_or("text");
    if kind != "text" {
        return Err(format!(
            "Responses text.format type {kind:?} is not supported; Logan currently returns text"
        ));
    }
    Ok(())
}

fn nonempty_json(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Array(values) => !values.is_empty(),
        Value::Object(values) => !values.is_empty(),
        Value::String(value) => !value.is_empty(),
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn qwen_prompt_preserves_generated_assistant_prefix_for_cache_reuse() {
        let messages = vec![
            ApiMessage {
                role: "system".into(),
                text: "Be concise.".into(),
            },
            ApiMessage {
                role: "user".into(),
                text: "Hello".into(),
            },
            ApiMessage {
                role: "assistant".into(),
                text: "Hi".into(),
            },
            ApiMessage {
                role: "user".into(),
                text: "Again".into(),
            },
        ];
        let prompt = render_qwen_prompt(&messages, None).unwrap();
        assert!(prompt.contains("<|im_start|>assistant\n<think>\n\n</think>\n\nHi<|im_end|>\n"));
        assert!(prompt.ends_with(ASSISTANT_NON_THINKING_PREFIX));
    }

    #[test]
    fn chat_text_parts_are_accepted_and_images_are_rejected() {
        let text: ChatMessage = serde_json::from_value(json!({
            "role":"user",
            "content":[{"type":"text","text":"hello"}]
        }))
        .unwrap();
        assert_eq!(normalize_chat_messages(&[text]).unwrap()[0].text, "hello");
        let image: ChatMessage = serde_json::from_value(json!({
            "role":"user",
            "content":[{"type":"image_url","image_url":{"url":"x"}}]
        }))
        .unwrap();
        assert!(normalize_chat_messages(&[image]).is_err());
    }

    #[test]
    fn responses_string_input_becomes_user_message() {
        let input = json!("hello");
        assert_eq!(
            normalize_responses_input(Some(&input)).unwrap(),
            vec![ApiMessage {
                role: "user".into(),
                text: "hello".into()
            }]
        );
    }
}
