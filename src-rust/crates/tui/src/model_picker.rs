//! Model picker overlay (/model command).
//! Mirrors src/components/ModelPicker.tsx — including effort levels and
//! fast-mode notice.

use ratatui::buffer::Buffer;
use ratatui::layout::{Alignment, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use unicode_width::UnicodeWidthStr;

use crate::overlays::centered_rect;

// ---------------------------------------------------------------------------
// Effort level
// ---------------------------------------------------------------------------

/// Mirrors the TS `EffortLevel` enum and `effortLevelToSymbol()` helper.
///
/// Effort controls the extended-thinking `budget_tokens` parameter sent to the
/// API. Only models that support extended thinking honour this; for others it
/// is silently ignored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EffortLevel {
    Low,
    Normal,
    High,
    Max,
}

impl EffortLevel {
    /// Unicode quarter-circle symbol used in the TS UI.
    pub fn symbol(self) -> &'static str {
        match self {
            Self::Low    => "\u{25cb}", // ○  empty circle
            Self::Normal => "\u{25d0}", // ◐  half
            Self::High   => "\u{25d5}", // ◕  three-quarter
            Self::Max    => "\u{25cf}", // ●  full
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Low    => "low",
            Self::Normal => "normal",
            Self::High   => "high",
            Self::Max    => "max",
        }
    }

    /// Returns the budget_tokens value to pass to the API, or `None` for the
    /// default (no extended thinking).
    pub fn budget_tokens(self) -> Option<u32> {
        match self {
            Self::Low    => Some(1_024),
            Self::Normal => None,
            Self::High   => Some(16_000),
            Self::Max    => Some(32_000),
        }
    }

    /// Cycle to next level; skips `Max` when the selected model does not
    /// support it.
    pub fn next(self, supports_max: bool) -> Self {
        match self {
            Self::Low    => Self::Normal,
            Self::Normal => Self::High,
            Self::High   => if supports_max { Self::Max } else { Self::Low },
            Self::Max    => Self::Low,
        }
    }

    /// Cycle to previous level.
    pub fn prev(self, supports_max: bool) -> Self {
        match self {
            Self::Low    => if supports_max { Self::Max } else { Self::High },
            Self::Normal => Self::Low,
            Self::High   => Self::Normal,
            Self::Max    => Self::High,
        }
    }
}

impl Default for EffortLevel {
    fn default() -> Self { Self::Normal }
}

// ---------------------------------------------------------------------------
// Model capability helpers
// ---------------------------------------------------------------------------

/// Returns `true` for models that support extended thinking / effort levels.
pub fn model_supports_effort(id: &str) -> bool {
    id.starts_with("claude-3-7")
        || id.starts_with("claude-opus-4")
        || id.starts_with("claude-sonnet-4")
}

/// Returns `true` for models that support the maximum effort tier.
pub fn model_supports_max_effort(id: &str) -> bool {
    id.starts_with("claude-opus-4")
}

/// The model ID that fast-mode locks to.
pub const FAST_MODE_MODEL: &str = "claude-haiku-4-5";

/// Returns a short description string based on the model family inferred from
/// the model ID.  Used when converting API model entries to `ModelEntry`.
pub fn model_family_description(id: &str) -> String {
    let lower = id.to_lowercase();
    if lower.contains("opus") {
        "Most capable — best for complex reasoning and analysis".to_string()
    } else if lower.contains("sonnet") {
        "Balanced performance and speed — great for coding tasks".to_string()
    } else if lower.contains("haiku") {
        "Fast and efficient — ideal for quick completions".to_string()
    } else {
        "Claude AI model".to_string()
    }
}

// ---------------------------------------------------------------------------
// Provider grouping helpers
// ---------------------------------------------------------------------------

/// Format a model display line with optional context window and cost info.
///
/// Example: `"gpt-4o  128K ctx  $5.00/M"`
pub fn format_model_line(model_str: &str, context_window: Option<u32>, cost_per_1m: Option<f64>) -> String {
    let mut parts = vec![model_str.to_string()];
    if let Some(ctx) = context_window {
        parts.push(format!("{}K ctx", ctx / 1000));
    }
    if let Some(cost) = cost_per_1m {
        if cost == 0.0 {
            parts.push("free".to_string());
        } else {
            parts.push(format!("${:.2}/M", cost));
        }
    }
    parts.join("  ")
}

/// A group of models belonging to the same provider, for structured display.
pub struct ProviderSection {
    pub provider_name: String,
    pub models: Vec<String>, // model ID strings in "provider/model" format
}

impl ModelPickerState {
    /// Build grouped model sections from a flat list of model strings.
    ///
    /// Models with a `"provider/model"` slash format are grouped by their
    /// provider prefix.  Bare model names are heuristically assigned to a
    /// provider based on the model name pattern.
    pub fn build_provider_sections(models: &[String]) -> Vec<ProviderSection> {
        use std::collections::HashMap;
        let mut by_provider: HashMap<String, Vec<String>> = HashMap::new();

        for m in models {
            let provider = if let Some((p, _)) = m.split_once('/') {
                p.to_string()
            } else {
                // Bare model name — detect provider from model name
                if m.contains("claude") {
                    "anthropic".to_string()
                } else if m.starts_with("gpt") || m.starts_with("o3") || m.starts_with("o4") {
                    "openai".to_string()
                } else if m.contains("gemini") {
                    "google".to_string()
                } else {
                    "other".to_string()
                }
            };
            by_provider.entry(provider).or_default().push(m.clone());
        }

        // Define display order
        let order = ["anthropic", "openai", "google", "ollama", "other"];
        let mut sections = Vec::new();
        for provider in order {
            if let Some(models) = by_provider.remove(provider) {
                sections.push(ProviderSection {
                    provider_name: match provider {
                        "anthropic" => "ANTHROPIC".to_string(),
                        "openai" => "OPENAI".to_string(),
                        "google" => "GOOGLE".to_string(),
                        "ollama" => "OLLAMA (local)".to_string(),
                        _ => provider.to_uppercase(),
                    },
                    models,
                });
            }
        }
        // Add any remaining providers not in the order list
        for (provider, models) in by_provider {
            sections.push(ProviderSection {
                provider_name: provider.to_uppercase(),
                models,
            });
        }
        sections
    }
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A single model entry shown in the picker.
#[derive(Debug, Clone)]
pub struct ModelEntry {
    pub id: String,
    pub display_name: String,
    pub description: String,
    /// Whether this is the currently active model.
    pub is_current: bool,
}

/// State for the /model picker overlay.
pub struct ModelPickerState {
    pub visible: bool,
    pub selected_idx: usize,
    pub models: Vec<ModelEntry>,
    /// Live filter typed by the user.
    pub filter: String,
    /// Current effort level for models that support extended thinking.
    pub effort_level: EffortLevel,
    /// Whether fast mode is currently active (locks model to FAST_MODE_MODEL).
    pub fast_mode: bool,
    /// `true` once the dynamic model list has been loaded from the API.
    pub models_loaded: bool,
    /// `true` while the background fetch is in flight.
    pub loading_models: bool,
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

impl ModelPickerState {
    /// Create a new picker with the default model list (not yet visible).
    pub fn new() -> Self {
        Self {
            visible: false,
            selected_idx: 0,
            models: Self::default_models(),
            filter: String::new(),
            effort_level: EffortLevel::Normal,
            fast_mode: false,
            models_loaded: false,
            loading_models: false,
        }
    }

    /// Open the overlay.
    ///
    /// `current_model` is highlighted as active; `current_effort` and
    /// `fast_mode` are carried over from app state so the user sees the live
    /// values.
    pub fn open(&mut self, current_model: &str) {
        self.open_with_state(current_model, EffortLevel::Normal, false);
    }

    /// Open the overlay with full state context.
    pub fn open_with_state(&mut self, current_model: &str, effort: EffortLevel, fast_mode: bool) {
        for m in &mut self.models {
            m.is_current = m.id == current_model;
        }
        self.selected_idx = self
            .models
            .iter()
            .position(|m| m.is_current)
            .unwrap_or(0);
        self.filter.clear();
        self.effort_level = effort;
        self.fast_mode = fast_mode;
        self.visible = true;
    }

    /// Close the overlay without selecting.
    pub fn close(&mut self) {
        self.visible = false;
        self.filter.clear();
    }

    /// Move selection up one row (wraps to last if at top).
    pub fn select_prev(&mut self) {
        let count = self.filtered_models().len();
        if count == 0 { return; }
        if self.selected_idx == 0 {
            self.selected_idx = count - 1;
        } else {
            self.selected_idx -= 1;
        }
    }

    /// Move selection down one row (wraps to first if at bottom).
    pub fn select_next(&mut self) {
        let count = self.filtered_models().len();
        if count == 0 { return; }
        self.selected_idx = (self.selected_idx + 1) % count;
    }

    /// Cycle effort level forward (→ key).
    pub fn effort_next(&mut self) {
        let filtered = self.filtered_models();
        let id = filtered.get(self.selected_idx).map(|m| m.id.as_str()).unwrap_or("");
        let supports_max = model_supports_max_effort(id);
        self.effort_level = self.effort_level.next(supports_max);
    }

    /// Cycle effort level backward (← key).
    pub fn effort_prev(&mut self) {
        let filtered = self.filtered_models();
        let id = filtered.get(self.selected_idx).map(|m| m.id.as_str()).unwrap_or("");
        let supports_max = model_supports_max_effort(id);
        self.effort_level = self.effort_level.prev(supports_max);
    }

    /// Returns the effective effort for the currently highlighted model:
    /// `None` if the model does not support extended thinking.
    pub fn effective_effort(&self) -> Option<EffortLevel> {
        let filtered = self.filtered_models();
        let id = filtered.get(self.selected_idx).map(|m| m.id.as_str()).unwrap_or("");
        if model_supports_effort(id) {
            Some(self.effort_level)
        } else {
            None
        }
    }

    /// Confirm the current selection.
    ///
    /// Returns `(model_id, effort)` where `effort` is `None` for models that
    /// do not support extended thinking.  Closes the picker.
    ///
    /// Persists the selected model (and effort level, if applicable) to
    /// `~/.claude/settings.json` under the keys `"model"` and `"effort"`.
    pub fn confirm(&mut self) -> Option<(String, Option<EffortLevel>)> {
        let filtered = self.filtered_models();
        let entry = filtered.get(self.selected_idx)?;
        let id = entry.id.clone();
        let effort = if model_supports_effort(&id) { Some(self.effort_level) } else { None };
        // If user chose a model other than the fast-mode model while fast mode is
        // active, the caller should turn off fast mode (mirrors TS behaviour).
        self.close();

        // Persist selection to ~/.claude/settings.json (best-effort; ignore I/O errors).
        let settings_path = claurst_core::config::Settings::global_settings_path();
        let existing = std::fs::read_to_string(&settings_path).unwrap_or_default();
        let mut json: serde_json::Value = serde_json::from_str(&existing)
            .unwrap_or_else(|_| serde_json::json!({}));
        if let Some(obj) = json.as_object_mut() {
            obj.insert("model".to_string(), serde_json::Value::String(id.clone()));
            if let Some(e) = effort {
                obj.insert("effort".to_string(), serde_json::Value::String(e.label().to_string()));
            } else {
                obj.remove("effort");
            }
        }
        if let Ok(serialized) = serde_json::to_string_pretty(&json) {
            if let Some(parent) = settings_path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let _ = std::fs::write(&settings_path, serialized);
        }

        Some((id, effort))
    }

    /// Append a character to the filter string and reset the selection.
    pub fn push_filter_char(&mut self, c: char) {
        self.filter.push(c);
        self.selected_idx = 0;
    }

    /// Remove the last character from the filter string.
    pub fn pop_filter_char(&mut self) {
        self.filter.pop();
        self.selected_idx = 0;
    }

    /// Return models that match the current filter (case-insensitive).
    pub fn filtered_models(&self) -> Vec<&ModelEntry> {
        if self.filter.is_empty() {
            return self.models.iter().collect();
        }
        let needle = self.filter.to_lowercase();
        self.models
            .iter()
            .filter(|m| {
                m.id.to_lowercase().contains(needle.as_str())
                    || m.display_name.to_lowercase().contains(needle.as_str())
                    || m.description.to_lowercase().contains(needle.as_str())
            })
            .collect()
    }

    /// Replace the model list with dynamically loaded entries.
    ///
    /// Called by the app event loop when the background fetch completes.
    /// Resets `loading_models` and sets `models_loaded`.
    pub fn set_models(&mut self, entries: Vec<ModelEntry>) {
        self.models = entries;
        self.loading_models = false;
        self.models_loaded = true;
        // Keep selected_idx in bounds.
        let count = self.filtered_models().len();
        if count > 0 && self.selected_idx >= count {
            self.selected_idx = count - 1;
        }
    }

    /// Fetch the list of available models from the Anthropic API and convert
    /// them to `ModelEntry` values.
    ///
    /// On success, models are sorted newest-first (by `created_at` descending).
    /// On any error, returns `default_models()` as a fallback so the picker is
    /// never left empty.
    pub async fn fetch_models(client: &claurst_api::AnthropicClient) -> Vec<ModelEntry> {
        match client.fetch_available_models().await {
            Ok(available) => {
                if available.is_empty() {
                    return Self::default_models();
                }

                let mut entries: Vec<(i64, ModelEntry)> = available
                    .into_iter()
                    .map(|m| {
                        let display = m
                            .display_name
                            .clone()
                            .unwrap_or_else(|| m.id.clone());
                        let description = model_family_description(&m.id);
                        let ts = m.created_at.unwrap_or(0);
                        (ts, ModelEntry {
                            id: m.id,
                            display_name: display,
                            description,
                            is_current: false,
                        })
                    })
                    .collect();

                // Sort newest-first.
                entries.sort_by(|a, b| b.0.cmp(&a.0));
                entries.into_iter().map(|(_, e)| e).collect()
            }
            Err(_) => Self::default_models(),
        }
    }

    /// Hardcoded list of Claude models available as of 2025.
    pub fn default_models() -> Vec<ModelEntry> {
        vec![
            ModelEntry {
                id: "claude-opus-4-6".to_string(),
                display_name: "Claude Opus 4.6".to_string(),
                description: "Most capable model — best for complex reasoning and analysis".to_string(),
                is_current: false,
            },
            ModelEntry {
                id: "claude-sonnet-4-6".to_string(),
                display_name: "Claude Sonnet 4.6".to_string(),
                description: "Balanced performance and speed — great for coding tasks".to_string(),
                is_current: false,
            },
            ModelEntry {
                id: "claude-haiku-4-5-20251001".to_string(),
                display_name: "Claude Haiku 4.5 (2025-10-01)".to_string(),
                description: "Fast and efficient — ideal for quick completions".to_string(),
                is_current: false,
            },
            ModelEntry {
                id: "claude-opus-4-5".to_string(),
                display_name: "Claude Opus 4.5".to_string(),
                description: "Previous Opus generation — powerful multimodal reasoning".to_string(),
                is_current: false,
            },
            ModelEntry {
                id: "claude-sonnet-4-5".to_string(),
                display_name: "Claude Sonnet 4.5".to_string(),
                description: "Previous Sonnet generation — solid coding and writing".to_string(),
                is_current: false,
            },
            ModelEntry {
                id: "claude-haiku-4-5".to_string(),
                display_name: "Claude Haiku 4.5".to_string(),
                description: "Previous Haiku generation — lightweight and responsive".to_string(),
                is_current: false,
            },
            ModelEntry {
                id: "claude-3-7-sonnet-20250219".to_string(),
                display_name: "Claude 3.7 Sonnet (2025-02-19)".to_string(),
                description: "Sonnet 3.7 with enhanced instruction following".to_string(),
                is_current: false,
            },
            ModelEntry {
                id: "claude-3-5-sonnet-20241022".to_string(),
                display_name: "Claude 3.5 Sonnet (2024-10-22)".to_string(),
                description: "Highly capable 3.5 Sonnet — reliable and well-tested".to_string(),
                is_current: false,
            },
            ModelEntry {
                id: "claude-3-5-haiku-20241022".to_string(),
                display_name: "Claude 3.5 Haiku (2024-10-22)".to_string(),
                description: "Fast 3.5 Haiku — great for high-throughput pipelines".to_string(),
                is_current: false,
            },
        ]
    }
}

impl Default for ModelPickerState {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// Render the model picker overlay directly into `buf`.
///
/// Draws a centred modal (≈70 wide × ≈22 tall) with:
/// - Fast-mode notice when fast mode is active
/// - A filter line when the user is typing
/// - A scrollable list of models with effort indicator for supporting models
/// - Selection highlight on the focused row
/// - Bottom hint bar with ←/→ keys for effort adjustment
pub fn render_model_picker(state: &ModelPickerState, area: Rect, buf: &mut Buffer) {
    if !state.visible {
        return;
    }

    const MODAL_W: u16 = 70;
    const MODAL_H: u16 = 22;

    let dialog_area = centered_rect(
        MODAL_W.min(area.width.saturating_sub(2)),
        MODAL_H.min(area.height.saturating_sub(2)),
        area,
    );

    // --- Clear background -------------------------------------------------
    for y in dialog_area.y..dialog_area.y + dialog_area.height {
        for x in dialog_area.x..dialog_area.x + dialog_area.width {
            if let Some(cell) = buf.cell_mut((x, y)) {
                cell.reset();
            }
        }
    }

    // --- Build line list --------------------------------------------------
    let mut lines: Vec<Line> = Vec::new();

    // Fast-mode notice
    if state.fast_mode {
        lines.push(Line::from(vec![
            Span::styled("  \u{26a1} ", Style::default().fg(Color::Yellow)),
            Span::styled(
                format!(
                    "Fast mode is ON ({} only). Switching turns it off.",
                    FAST_MODE_MODEL
                ),
                Style::default().fg(Color::Yellow).add_modifier(Modifier::ITALIC),
            ),
        ]));
        lines.push(Line::from(""));
    }

    // Loading-models notice
    if state.loading_models {
        lines.push(Line::from(vec![
            Span::styled("  ", Style::default()),
            Span::styled(
                "\u{29d7} Loading models\u{2026}",
                Style::default().fg(Color::Rgb(100, 180, 255)).add_modifier(Modifier::ITALIC),
            ),
        ]));
        lines.push(Line::from(""));
    }

    // Optional filter line
    if !state.filter.is_empty() {
        lines.push(Line::from(vec![
            Span::styled("  Filter: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                state.filter.clone(),
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            ),
        ]));
        lines.push(Line::from(""));
    }

    let filtered = state.filtered_models();

    if filtered.is_empty() {
        lines.push(Line::from(vec![Span::styled(
            "  No models match filter",
            Style::default().fg(Color::DarkGray),
        )]));
    } else {
        let inner_w = dialog_area.width.saturating_sub(2) as usize;

        for (i, model) in filtered.iter().enumerate() {
            let is_selected = i == state.selected_idx;
            let supports_effort = model_supports_effort(&model.id);

            // Bullet: filled circle for the currently active model.
            let bullet = if model.is_current { "\u{25cf}" } else { "\u{25cb}" };
            let bullet_style = if model.is_current {
                Style::default().fg(Color::Green)
            } else {
                Style::default().fg(Color::DarkGray)
            };

            let row_style = if is_selected {
                Style::default().bg(Color::Rgb(40, 60, 80)).add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            let name_style = if is_selected {
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD).bg(Color::Rgb(40, 60, 80))
            } else {
                Style::default().fg(Color::White)
            };
            let desc_style = if is_selected {
                Style::default().fg(Color::Rgb(180, 200, 220)).bg(Color::Rgb(40, 60, 80))
            } else {
                Style::default().fg(Color::DarkGray)
            };

            // Effort indicator for supported models (shown right of name).
            let effort_span: Option<Span<'static>> = if supports_effort && is_selected {
                let sym = state.effort_level.symbol();
                let lbl = state.effort_level.label();
                Some(Span::styled(
                    format!("  {} {}", sym, lbl),
                    Style::default().fg(Color::Rgb(100, 200, 120)).bg(Color::Rgb(40, 60, 80)),
                ))
            } else if supports_effort {
                // Subtle indicator when not selected
                Some(Span::styled(
                    format!("  {}", state.effort_level.symbol()),
                    Style::default().fg(Color::Rgb(60, 100, 60)),
                ))
            } else {
                None
            };

            // Description budget accounts for effort span width.
            let effort_w = effort_span
                .as_ref()
                .map(|s| s.content.width())
                .unwrap_or(0);
            let name_w = model.display_name.width();
            let desc_budget = inner_w.saturating_sub(4 + name_w + effort_w + 2);
            let desc: String = if model.description.width() > desc_budget && desc_budget > 3 {
                let mut s = model.description.clone();
                while s.width() > desc_budget.saturating_sub(1) { s.pop(); }
                format!("{s}\u{2026}")
            } else {
                model.description.clone()
            };

            let mut spans = vec![
                Span::styled("  ", row_style),
                Span::styled(bullet, bullet_style.patch(row_style)),
                Span::styled(" ", row_style),
                Span::styled(model.display_name.clone(), name_style),
            ];
            if let Some(es) = effort_span {
                spans.push(es);
            }
            spans.push(Span::styled("  ", row_style));
            spans.push(Span::styled(desc, desc_style));

            lines.push(Line::from(spans));
        }
    }

    // Spacer + hint
    lines.push(Line::from(""));
    let hint_line = Line::from(vec![
        Span::styled("  ", Style::default()),
        Span::styled("Enter", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::styled("=select  ", Style::default().fg(Color::DarkGray)),
        Span::styled("\u{2190}\u{2192}", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::styled("=effort  ", Style::default().fg(Color::DarkGray)),
        Span::styled("Esc", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::styled("=cancel  ", Style::default().fg(Color::DarkGray)),
        Span::styled("Type to filter", Style::default().fg(Color::DarkGray)),
    ]);
    lines.push(hint_line);

    let block = Block::default()
        .borders(Borders::ALL)
        .title(" Model Picker ")
        .title_alignment(Alignment::Center)
        .border_style(Style::default().fg(Color::Cyan));

    let para = Paragraph::new(lines)
        .block(block)
        .alignment(Alignment::Left)
        .wrap(Wrap { trim: false });

    use ratatui::widgets::Widget;
    para.render(dialog_area, buf);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_picker_with_current(current: &str) -> ModelPickerState {
        let mut p = ModelPickerState::new();
        p.open(current);
        p
    }

    // 1. Default model list is non-empty and contains expected IDs.
    #[test]
    fn default_models_are_populated() {
        let models = ModelPickerState::default_models();
        assert!(!models.is_empty(), "default model list must not be empty");
        let ids: Vec<&str> = models.iter().map(|m| m.id.as_str()).collect();
        assert!(ids.contains(&"claude-sonnet-4-6"));
        assert!(ids.contains(&"claude-opus-4-6"));
        assert!(ids.contains(&"claude-3-5-haiku-20241022"));
    }

    // 2. open() marks exactly one model as current.
    #[test]
    fn open_marks_current_model() {
        let mut p = ModelPickerState::new();
        p.open("claude-sonnet-4-6");
        let current_count = p.models.iter().filter(|m| m.is_current).count();
        assert_eq!(current_count, 1);
        assert!(p.models.iter().find(|m| m.id == "claude-sonnet-4-6").unwrap().is_current);
    }

    // 3. open() with an unknown model ID marks none as current and sets idx=0.
    #[test]
    fn open_unknown_model_selects_first() {
        let mut p = ModelPickerState::new();
        p.open("unknown-model");
        assert_eq!(p.selected_idx, 0);
        assert!(p.models.iter().all(|m| !m.is_current));
    }

    // 4. select_next() wraps around to 0 after the last entry.
    #[test]
    fn select_next_wraps() {
        let mut p = make_picker_with_current("claude-opus-4-6");
        let total = p.filtered_models().len();
        p.selected_idx = total - 1;
        p.select_next();
        assert_eq!(p.selected_idx, 0);
    }

    // 5. select_prev() wraps around to last after idx 0.
    #[test]
    fn select_prev_wraps() {
        let mut p = make_picker_with_current("claude-opus-4-6");
        p.selected_idx = 0;
        p.select_prev();
        let total = p.filtered_models().len();
        assert_eq!(p.selected_idx, total - 1);
    }

    // 6. filter reduces visible entries.
    #[test]
    fn filter_reduces_results() {
        let mut p = make_picker_with_current("claude-opus-4-6");
        for c in "sonnet".chars() { p.push_filter_char(c); }
        let all = p.models.len();
        let filtered = p.filtered_models();
        assert!(filtered.len() < all, "filter should reduce the result count");
        assert!(!filtered.is_empty(), "at least one sonnet model must match");
        for m in &filtered {
            let haystack = format!("{} {} {}", m.id, m.display_name, m.description).to_lowercase();
            assert!(haystack.contains("sonnet"), "model '{}' does not match filter", m.id);
        }
    }

    // 7. pop_filter_char removes last char.
    #[test]
    fn pop_filter_char_removes_last() {
        let mut p = make_picker_with_current("claude-opus-4-6");
        p.push_filter_char('h'); p.push_filter_char('a'); p.push_filter_char('i');
        assert_eq!(p.filter, "hai");
        p.pop_filter_char();
        assert_eq!(p.filter, "ha");
    }

    // 8. confirm() returns selected model ID and closes the picker.
    #[test]
    fn confirm_returns_id_and_closes() {
        let mut p = make_picker_with_current("claude-opus-4-6");
        p.selected_idx = 0;
        let first_id = p.filtered_models()[0].id.clone();
        let result = p.confirm();
        assert_eq!(result.map(|(id, _)| id), Some(first_id));
        assert!(!p.visible, "picker should be closed after confirm");
    }

    // 9. confirm() on empty filter list returns None.
    #[test]
    fn confirm_empty_filter_returns_none() {
        let mut p = make_picker_with_current("claude-opus-4-6");
        p.filter = "zzznomatch999".to_string();
        p.selected_idx = 0;
        let result = p.confirm();
        assert!(result.is_none());
    }

    // 10. close() clears filter and hides overlay.
    #[test]
    fn close_clears_state() {
        let mut p = make_picker_with_current("claude-opus-4-6");
        p.push_filter_char('x');
        p.close();
        assert!(!p.visible);
        assert!(p.filter.is_empty());
    }

    // 11. effort cycling works for effort-supporting models.
    #[test]
    fn effort_cycles_correctly() {
        let mut p = make_picker_with_current("claude-sonnet-4-6");
        // sonnet-4-6 supports effort but not max
        assert_eq!(p.effort_level, EffortLevel::Normal);
        p.effort_next();
        assert_eq!(p.effort_level, EffortLevel::High);
        p.effort_next();
        // no max for sonnet → wraps to Low
        assert_eq!(p.effort_level, EffortLevel::Low);
    }

    // 12. Opus supports max effort.
    #[test]
    fn opus_supports_max_effort() {
        assert!(model_supports_max_effort("claude-opus-4-6"));
        assert!(!model_supports_max_effort("claude-sonnet-4-6"));
        assert!(!model_supports_max_effort("claude-haiku-4-5"));
    }

    // 13. Non-effort models return None from effective_effort.
    #[test]
    fn haiku_has_no_effort() {
        let mut p = make_picker_with_current("claude-haiku-4-5");
        p.selected_idx = p.models.iter().position(|m| m.id == "claude-haiku-4-5").unwrap();
        assert!(!model_supports_effort("claude-haiku-4-5"));
        let effort = p.confirm();
        assert!(effort.is_some_and(|(_, e)| e.is_none()));
    }

    // 14. render_model_picker does not panic for a default-area call.
    #[test]
    fn render_does_not_panic() {
        let mut p = ModelPickerState::new();
        p.open("claude-sonnet-4-6");
        let area = Rect::new(0, 0, 120, 40);
        let mut buf = Buffer::empty(area);
        render_model_picker(&p, area, &mut buf);
    }

    // 15. render does nothing when not visible.
    #[test]
    fn render_noop_when_hidden() {
        let p = ModelPickerState::new();
        let area = Rect::new(0, 0, 80, 24);
        let mut buf = Buffer::empty(area);
        render_model_picker(&p, area, &mut buf);
        for cell in buf.content() {
            assert_eq!(cell.symbol(), " ", "buffer should be empty when picker is hidden");
        }
    }
}
