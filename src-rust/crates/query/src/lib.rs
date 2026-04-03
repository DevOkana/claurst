// cc-query: The core agentic query loop.
//
// This crate implements the main conversation loop that:
// 1. Sends messages to the Anthropic API
// 2. Processes streaming responses
// 3. Detects tool-use requests and dispatches them
// 4. Feeds tool results back to the model
// 5. Handles auto-compact when the context window fills up
// 6. Manages stop conditions (end_turn, max_turns, cancellation)

pub mod agent_tool;
pub mod auto_dream;
pub mod away_summary;
pub mod command_queue;
pub mod compact;
pub mod context_analyzer;
pub mod coordinator;
pub mod cron_scheduler;
pub mod session_memory;
pub mod skill_prefetch;
pub use agent_tool::{AgentTool, init_team_swarm_runner};
pub use command_queue::{CommandPriority, CommandQueue, QueuedCommand, drain_command_queue};
pub use cron_scheduler::start_cron_scheduler;
pub use skill_prefetch::{
    SkillDefinition, SkillIndex, SharedSkillIndex, prefetch_skills, format_skill_listing,
};
pub use compact::{
    AutoCompactState, CompactResult, CompactTrigger, MicroCompactConfig, MessageGroup, TokenWarningState,
    auto_compact_if_needed, calculate_messages_to_keep_index, calculate_token_warning_state,
    compact_conversation, context_collapse, context_window_for_model, format_compact_summary,
    get_compact_prompt, group_messages_for_compact, micro_compact_if_needed,
    reactive_compact, should_auto_compact, should_compact, should_context_collapse, snip_compact,
};
pub use session_memory::{
    ExtractedMemory, MemoryCategory, SessionMemoryExtractor, SessionMemoryState,
};

use claurst_api::{
    ApiMessage, ApiToolDefinition, AnthropicStreamEvent, CreateMessageRequest, StreamAccumulator,
    StreamHandler, SystemPrompt, ThinkingConfig,
};
use claurst_core::config::Config;
use claurst_core::cost::CostTracker;
use claurst_core::error::ClaudeError;
use claurst_core::types::{ContentBlock, Message, ToolResultContent, UsageInfo};
use claurst_tools::{Tool, ToolContext, ToolResult};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Outcome of a single query-loop run.
#[derive(Debug)]
pub enum QueryOutcome {
    /// The model finished its turn (end_turn stop reason).
    EndTurn { message: Message, usage: UsageInfo },
    /// The model hit max_tokens.
    MaxTokens { partial_message: Message, usage: UsageInfo },
    /// The conversation was cancelled by the user.
    Cancelled,
    /// An unrecoverable error occurred.
    Error(ClaudeError),
    /// The configured USD budget was exceeded.
    BudgetExceeded { cost_usd: f64, limit_usd: f64 },
}

/// Configuration for a single query-loop invocation.
#[derive(Clone)]
pub struct QueryConfig {
    pub model: String,
    pub max_tokens: u32,
    pub max_turns: u32,
    pub system_prompt: Option<String>,
    pub append_system_prompt: Option<String>,
    pub output_style: claurst_core::system_prompt::OutputStyle,
    pub output_style_prompt: Option<String>,
    pub working_directory: Option<String>,
    pub thinking_budget: Option<u32>,
    pub temperature: Option<f32>,
    /// Maximum cumulative character count of all tool results in the message
    /// history before older results are replaced with a truncation notice.
    /// Mirrors the TS `applyToolResultBudget` mechanism.  Default: 50_000.
    pub tool_result_budget: usize,
    /// Optional effort level.  When set and `thinking_budget` is `None`,
    /// the effort level's `thinking_budget_tokens()` is used as the
    /// thinking budget.  Also provides a temperature override when the
    /// level specifies one.
    pub effort_level: Option<claurst_core::effort::EffortLevel>,
    /// T1-4: Optional shared command queue.
    ///
    /// When set, the query loop drains this queue before each API call and
    /// injects any resulting messages into the conversation.  The queue is
    /// shared (Arc-backed) so the TUI input thread can push commands while the
    /// loop is waiting for a model response.
    pub command_queue: Option<CommandQueue>,
    /// T1-5: Optional shared skill index.
    ///
    /// When set, `prefetch_skills` is spawned once before the loop begins and
    /// the resulting index is used to inject a skill listing attachment into
    /// the conversation context.
    pub skill_index: Option<SharedSkillIndex>,
    /// Optional USD spend cap. The query loop checks accumulated cost after
    /// each turn and aborts with `QueryOutcome::BudgetExceeded` when exceeded.
    pub max_budget_usd: Option<f64>,
    /// Fallback model name. Used when the primary model returns overloaded /
    /// rate-limit errors (mirrors TS `--fallback-model`).
    pub fallback_model: Option<String>,
    /// Optional ProviderRegistry for dispatching to non-Anthropic providers.
    /// When `config.provider` is set to something other than "anthropic" and
    /// this registry contains that provider, the registry's provider is used
    /// instead of `AnthropicClient`.
    pub provider_registry: Option<std::sync::Arc<claurst_api::ProviderRegistry>>,
}

impl Default for QueryConfig {
    fn default() -> Self {
        Self {
            model: claurst_core::constants::DEFAULT_MODEL.to_string(),
            max_tokens: claurst_core::constants::DEFAULT_MAX_TOKENS,
            max_turns: claurst_core::constants::MAX_TURNS_DEFAULT,
            system_prompt: None,
            append_system_prompt: None,
            output_style: claurst_core::system_prompt::OutputStyle::Default,
            output_style_prompt: None,
            working_directory: None,
            thinking_budget: None,
            temperature: None,
            tool_result_budget: 50_000,
            effort_level: None,
            command_queue: None,
            skill_index: None,
            max_budget_usd: None,
            fallback_model: None,
            provider_registry: None,
        }
    }
}

impl QueryConfig {
    pub fn from_config(cfg: &Config) -> Self {
        Self {
            model: cfg.effective_model().to_string(),
            max_tokens: cfg.effective_max_tokens(),
            output_style: cfg.effective_output_style(),
            output_style_prompt: cfg.resolve_output_style_prompt(),
            working_directory: cfg
                .project_dir
                .as_ref()
                .map(|p| p.display().to_string()),
            ..Default::default()
        }
    }
}

/// Events emitted by the query loop for the TUI to render.
#[derive(Debug, Clone)]
pub enum QueryEvent {
    /// A stream event from the API.
    Stream(AnthropicStreamEvent),
    /// A tool is about to be executed.
    ToolStart { tool_name: String, tool_id: String, input_json: String },
    /// A tool has finished executing.
    ToolEnd { tool_name: String, tool_id: String, result: String, is_error: bool },
    /// The model finished a turn.
    TurnComplete { turn: u32, stop_reason: String, usage: Option<UsageInfo> },
    /// An informational status message.
    Status(String),
    /// An error.
    Error(String),
    /// Token usage has crossed a warning threshold.
    /// `state` is Warning (≥ 80 %) or Critical (≥ 95 %).
    /// `pct_used` is the fraction of the context window consumed (0.0–1.0).
    TokenWarning { state: TokenWarningState, pct_used: f64 },
}

// ---------------------------------------------------------------------------
// T1-3: Post-sampling hooks
// ---------------------------------------------------------------------------

/// Result returned by `fire_post_sampling_hooks`.
#[derive(Debug, Default)]
pub struct PostSamplingHookResult {
    /// Error messages produced by hooks with non-zero exit codes.
    /// These are injected into the conversation as user messages before the
    /// next model turn so the model can react to them.
    pub blocking_errors: Vec<claurst_core::types::Message>,
    /// When `true` the query loop must not continue and should surface the
    /// error messages to the caller.  Set when any hook exits with code > 1.
    pub prevent_continuation: bool,
}

/// Execute all `PostModelTurn` hooks defined in `config.hooks`.
///
/// Each hook is run synchronously (blocking via `std::process::Command`).
/// On a non-zero exit code, the hook's stderr (falling back to stdout) is
/// wrapped in a user `Message` and appended to `blocking_errors`.
/// If the exit code is **strictly greater than 1** `prevent_continuation` is
/// set so the query loop can return early.
pub fn fire_post_sampling_hooks(
    _turn_result: &claurst_core::types::Message,
    config: &claurst_core::config::Config,
) -> PostSamplingHookResult {
    use claurst_core::config::HookEvent;
    use claurst_core::types::Message;

    let mut result = PostSamplingHookResult::default();

    let entries = match config.hooks.get(&HookEvent::PostModelTurn) {
        Some(e) => e,
        None => return result,
    };

    for entry in entries {
        let sh = if cfg!(windows) { "cmd" } else { "sh" };
        let flag = if cfg!(windows) { "/C" } else { "-c" };

        let output = match std::process::Command::new(sh)
            .args([flag, &entry.command])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .output()
        {
            Ok(o) => o,
            Err(e) => {
                tracing::warn!(command = %entry.command, error = %e, "PostModelTurn hook spawn failed");
                continue;
            }
        };

        if output.status.success() {
            continue;
        }

        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let body = if !stderr.trim().is_empty() { stderr } else { stdout };

        tracing::warn!(
            command = %entry.command,
            exit_code = ?output.status.code(),
            "PostModelTurn hook returned non-zero exit"
        );

        result.blocking_errors.push(Message::user(format!(
            "[Hook '{}' error]:\n{}",
            entry.command,
            body.trim()
        )));

        // Exit code > 1 → hard veto of continuation.
        if output.status.code().unwrap_or(1) > 1 {
            result.prevent_continuation = true;
        }
    }

    result
}

/// Spawn all `Stop` hooks in fire-and-forget background tasks.
///
/// Stop hooks are non-blocking by design: the caller does not wait for them.
/// Returns an empty `Vec` immediately; results (if any) are lost.
pub fn stop_hooks_with_full_behavior(
    turn_result: &claurst_core::types::Message,
    config: &claurst_core::config::Config,
    working_dir: std::path::PathBuf,
) -> Vec<claurst_core::types::Message> {
    use claurst_core::config::HookEvent;

    let entries = match config.hooks.get(&HookEvent::Stop) {
        Some(e) if !e.is_empty() => e.clone(),
        _ => return Vec::new(),
    };

    let output_text = turn_result.get_all_text();

    for entry in entries {
        let cmd = entry.command.clone();
        let dir = working_dir.clone();
        let text = output_text.clone();

        tokio::task::spawn_blocking(move || {
            let sh = if cfg!(windows) { "cmd" } else { "sh" };
            let flag = if cfg!(windows) { "/C" } else { "-c" };

            let _ = std::process::Command::new(sh)
                .args([flag, &cmd])
                .current_dir(&dir)
                .env("CLAUDE_HOOK_OUTPUT", &text)
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn();
        });
    }

    Vec::new()
}

// ---------------------------------------------------------------------------
// Tool-result budgeting
// ---------------------------------------------------------------------------

/// Return the combined character count of all tool-result content blocks found
/// in `messages`.  Only user messages are examined (tool results always live
/// in user turns).
fn total_tool_result_chars(messages: &[Message]) -> usize {
    messages
        .iter()
        .filter(|m| m.role == claurst_core::types::Role::User)
        .flat_map(|m| match &m.content {
            claurst_core::types::MessageContent::Blocks(blocks) => blocks.as_slice(),
            _ => &[],
        })
        .filter_map(|b| {
            if let ContentBlock::ToolResult { content, .. } = b {
                Some(match content {
                    ToolResultContent::Text(t) => t.len(),
                    ToolResultContent::Blocks(blocks) => blocks.iter().map(|b| {
                        if let ContentBlock::Text { text } = b { text.len() } else { 0 }
                    }).sum(),
                })
            } else {
                None
            }
        })
        .sum()
}

/// When the cumulative tool-result content exceeds `budget` characters, walk
/// the message list from oldest to newest and replace individual
/// `ToolResult` content with a placeholder until the running total is back
/// under budget.  Returns the (possibly modified) message list and the
/// number of results that were truncated.
///
/// Mirrors the spirit of the TypeScript `applyToolResultBudget` /
/// `enforceToolResultBudget` logic, simplified to a straightforward
/// oldest-first eviction without the session-persistence layer.
fn apply_tool_result_budget(messages: Vec<Message>, budget: usize) -> (Vec<Message>, usize) {
    let total = total_tool_result_chars(&messages);
    if total <= budget {
        return (messages, 0);
    }

    let mut to_shed = total - budget;
    let mut truncated = 0usize;
    let mut result = messages;

    'outer: for msg in result.iter_mut() {
        if msg.role != claurst_core::types::Role::User {
            continue;
        }
        let blocks = match &mut msg.content {
            claurst_core::types::MessageContent::Blocks(b) => b,
            _ => continue,
        };
        for block in blocks.iter_mut() {
            if let ContentBlock::ToolResult { content, .. } = block {
                let size = match &*content {
                    ToolResultContent::Text(t) => t.len(),
                    ToolResultContent::Blocks(inner) => inner.iter().map(|b| {
                        if let ContentBlock::Text { text } = b { text.len() } else { 0 }
                    }).sum(),
                };
                if size == 0 {
                    continue;
                }
                *content = ToolResultContent::Text(
                    "[tool result truncated to save context]".to_string(),
                );
                truncated += 1;
                if size > to_shed {
                    break 'outer;
                }
                to_shed -= size;
            }
        }
    }

    (result, truncated)
}

// ---------------------------------------------------------------------------
// Query loop
// ---------------------------------------------------------------------------

/// Maximum number of max_tokens continuation attempts before surfacing the
/// partial response.  Mirrors `MAX_OUTPUT_TOKENS_RECOVERY_LIMIT` in query.ts.
const MAX_TOKENS_RECOVERY_LIMIT: u32 = 3;

/// Message injected when the model hits its output-token limit.
/// Mirrors the TS recovery message in query.ts lines 1224-1228.
const MAX_TOKENS_RECOVERY_MSG: &str =
    "Output token limit hit. Resume directly — no apology, no recap of what \
     you were doing. Pick up mid-thought if that is where the cut happened. \
     Break remaining work into smaller pieces.";

/// Run the agentic query loop.
///
/// This sends the conversation to the API, handles tool calls in a loop, and
/// returns when the model issues an end_turn or an error/limit is hit.
///
/// `pending_messages` is an optional queue of user messages that were enqueued
/// during tool execution (e.g. by the UI or a command queue).  Each string is
/// appended as a plain user message between turns.  Callers that do not need
/// command queuing may pass `None` or an empty `Vec`.
pub async fn run_query_loop(
    client: &claurst_api::AnthropicClient,
    messages: &mut Vec<Message>,
    tools: &[Box<dyn Tool>],
    tool_ctx: &ToolContext,
    config: &QueryConfig,
    cost_tracker: Arc<CostTracker>,
    event_tx: Option<mpsc::UnboundedSender<QueryEvent>>,
    cancel_token: tokio_util::sync::CancellationToken,
    mut pending_messages: Option<&mut Vec<String>>,
) -> QueryOutcome {
    let mut turn = 0u32;
    let mut compact_state = compact::AutoCompactState::default();
    // Tracks how many consecutive max_tokens recoveries we've attempted so
    // we don't loop forever on a model that can't finish within any budget.
    let mut max_tokens_recovery_count: u32 = 0;
    // Active model — may switch to fallback on overloaded errors.
    let mut effective_model = config.model.clone();
    let mut used_fallback = false;

    loop {
        turn += 1;
        tool_ctx
            .current_turn
            .store(turn as usize, std::sync::atomic::Ordering::Relaxed);
        if turn > config.max_turns {
            info!(turns = turn, "Max turns reached");
            if let Some(ref tx) = event_tx {
                let _ = tx.send(QueryEvent::Status(format!(
                    "Reached maximum turn limit ({})",
                    config.max_turns
                )));
            }
            // Return the last assistant message if any
            let last_msg = messages
                .last()
                .cloned()
                .unwrap_or_else(|| Message::assistant("Max turns reached."));
            return QueryOutcome::EndTurn {
                message: last_msg,
                usage: UsageInfo::default(),
            };
        }

        // Check for cancellation
        if cancel_token.is_cancelled() {
            return QueryOutcome::Cancelled;
        }

        // Drain any pending user messages that were queued during the previous
        // tool-execution phase (e.g. commands entered while tools ran).
        // Mirrors the TS `messageQueueManager` drain between turns.
        if let Some(queue) = pending_messages.as_deref_mut() {
            for text in queue.drain(..) {
                debug!("Injecting pending message: {}", &text);
                messages.push(Message::user(text));
            }
        }

        // T1-4: Drain the priority command queue (if wired up) and prepend any
        // resulting messages to the conversation before the API call.
        // Mirrors the TS `messageQueueManager` priority-queue drain.
        if let Some(ref cq) = config.command_queue {
            if !cq.is_empty() {
                let injected = drain_command_queue(cq);
                if !injected.is_empty() {
                    debug!(count = injected.len(), "Injecting command-queue messages");
                    // Prepend so that higher-priority commands appear first.
                    let tail = std::mem::take(messages);
                    messages.extend(injected);
                    messages.extend(tail);
                }
            }
        }

        // Apply tool-result budget: if the cumulative size of all tool results
        // in the conversation exceeds the configured threshold, replace the
        // oldest results with a placeholder until we're back under budget.
        // This mirrors the TS `applyToolResultBudget` call in query.ts.
        if config.tool_result_budget > 0 {
            let (budgeted, truncated) =
                apply_tool_result_budget(std::mem::take(messages), config.tool_result_budget);
            *messages = budgeted;
            if truncated > 0 {
                info!(
                    truncated,
                    budget = config.tool_result_budget,
                    "Tool-result budget exceeded: truncated {} result(s)",
                    truncated
                );
                if let Some(ref tx) = event_tx {
                    let _ = tx.send(QueryEvent::Status(format!(
                        "[{} older tool result(s) truncated to save context]",
                        truncated
                    )));
                }
            }
        }

        // Build API request
        let api_messages: Vec<ApiMessage> = messages.iter().map(ApiMessage::from).collect();
        let api_tools: Vec<ApiToolDefinition> = tools
            .iter()
            .map(|t| ApiToolDefinition::from(&t.to_definition()))
            .collect();

        // Verification nudge: if there are incomplete todos for this session
        // and the conversation has more than 2 turns, append a reminder.
        let system = if turn > 2 {
            let nudge = build_todo_nudge(&tool_ctx.session_id);
            if nudge.is_empty() {
                build_system_prompt(config)
            } else {
                let mut patched = config.clone();
                patched.append_system_prompt = Some(match &config.append_system_prompt {
                    Some(existing) => format!("{}\n\n{}", existing, nudge),
                    None => nudge,
                });
                build_system_prompt(&patched)
            }
        } else {
            build_system_prompt(config)
        };

        let mut req_builder = CreateMessageRequest::builder(&effective_model, config.max_tokens)
            .messages(api_messages)
            .system(system)
            .tools(api_tools);

        // Resolve effective thinking budget:
        //   1. Explicit `thinking_budget` in config takes precedence.
        //   2. Fall back to the effort level's budget when no explicit budget is set.
        let effective_thinking_budget = config.thinking_budget.or_else(|| {
            config
                .effort_level
                .and_then(|el| el.thinking_budget_tokens())
        });

        if let Some(budget) = effective_thinking_budget {
            req_builder = req_builder.thinking(ThinkingConfig::enabled(budget));
        }

        // Apply temperature: explicit config value takes precedence, then effort-level override.
        let effective_temperature = config.temperature.or_else(|| {
            config.effort_level.and_then(|el| el.temperature())
        });
        if let Some(t) = effective_temperature {
            req_builder = req_builder.temperature(t);
        }

        let request = req_builder.build();

        // Create a stream handler that forwards to the event channel
        let handler: Arc<dyn StreamHandler> = if let Some(ref tx) = event_tx {
            let tx = tx.clone();
            Arc::new(ChannelStreamHandler { tx })
        } else {
            Arc::new(claurst_api::streaming::NullStreamHandler)
        };

        // If a non-Anthropic provider is selected and the registry has it,
        // dispatch through the registry's LlmProvider rather than the
        // AnthropicClient.  The Anthropic path below remains unchanged.
        //
        // TODO(Phase 4 follow-up): replace this branch with a full streaming
        // implementation once the unified StreamEvent → AnthropicStreamEvent
        // bridge is built.  Currently we fall through to the Anthropic path
        // even when a registry provider is found, because the inner loop
        // uses AnthropicStreamEvent and StreamAccumulator which are Anthropic-
        // specific.  The registry hook here establishes the dispatch point so
        // the wiring can be completed incrementally.
        if let Some(ref registry) = config.provider_registry {
            let provider_id_str = effective_model
                .split('/')
                .next()
                .unwrap_or("anthropic");
            // Check if an explicit non-anthropic provider is configured.
            // The model field may be "provider/model-id" or just "model-id".
            // We consult the registry only when the provider is explicitly
            // non-Anthropic and has a registered entry.
            let maybe_provider_id = claurst_core::provider_id::ProviderId::new(provider_id_str);
            if provider_id_str != "anthropic" && registry.get(&maybe_provider_id).is_some() {
                // TODO: Implement full dispatch through registry provider.
                // The provider is available via registry.get(&maybe_provider_id).
                // Call provider.create_message(ProviderRequest { ... }).await
                // and convert the ProviderResponse back to a Message + UsageInfo.
                // For now we log a warning and fall through to the Anthropic path.
                warn!(
                    provider = %provider_id_str,
                    model = %effective_model,
                    "Non-Anthropic provider found in registry — full dispatch pending (Phase 4 follow-up)"
                );
            }
        }

        // Send to API
        debug!(turn, model = %effective_model, "Sending API request");
        let mut stream_rx = match client.create_message_stream(request, handler).await {
            Ok(rx) => rx,
            Err(e) => {
                // On overloaded/rate-limit errors, attempt one switch to the fallback model.
                let err_str = e.to_string().to_lowercase();
                if !used_fallback
                    && (err_str.contains("overloaded") || err_str.contains("529") || err_str.contains("rate_limit"))
                {
                    if let Some(ref fb) = config.fallback_model {
                        warn!(
                            primary = %effective_model,
                            fallback = %fb,
                            "Primary model unavailable — switching to fallback"
                        );
                        if let Some(ref tx) = event_tx {
                            let _ = tx.send(QueryEvent::Status(format!(
                                "Model unavailable — switching to fallback ({})",
                                fb
                            )));
                        }
                        effective_model = fb.clone();
                        used_fallback = true;
                        turn -= 1; // don't count this attempt against max_turns
                        continue;
                    }
                }
                error!(error = %e, "API request failed");
                return QueryOutcome::Error(e);
            }
        };

        // Accumulate the streamed response
        let mut accumulator = StreamAccumulator::new();

        loop {
            tokio::select! {
                _ = cancel_token.cancelled() => {
                    return QueryOutcome::Cancelled;
                }
                event = stream_rx.recv() => {
                    match event {
                        Some(evt) => {
                            accumulator.on_event(&evt);
                            match &evt {
                                AnthropicStreamEvent::Error { error_type, message } => {
                                    if error_type == "overloaded_error" {
                                        warn!(model = %effective_model, "API overloaded");
                                    }
                                    error!(error_type, message, "Stream error");
                                }
                                AnthropicStreamEvent::MessageStop => break,
                                _ => {}
                            }
                        }
                        None => break, // Stream ended
                    }
                }
            }
        }

        let (assistant_msg, usage, stop_reason) = accumulator.finish();

        // Track costs
        cost_tracker.add_usage(
            usage.input_tokens,
            usage.output_tokens,
            usage.cache_creation_input_tokens,
            usage.cache_read_input_tokens,
        );

        // Budget guard: abort the loop if the configured USD cap is exceeded.
        if let Some(limit) = config.max_budget_usd {
            let spent = cost_tracker.total_cost_usd();
            if spent >= limit {
                if let Some(ref tx) = event_tx {
                    let _ = tx.send(QueryEvent::Status(format!(
                        "Budget limit ${:.4} exceeded (spent ${:.4}) — stopping.",
                        limit, spent
                    )));
                }
                return QueryOutcome::BudgetExceeded {
                    cost_usd: spent,
                    limit_usd: limit,
                };
            }
        }

        // Append assistant message to conversation
        messages.push(assistant_msg.clone());

        let stop = stop_reason.as_deref().unwrap_or("end_turn");

        // T1-3: Fire PostModelTurn hooks after the model samples a response.
        // Hooks can inject blocking errors or veto continuation entirely.
        {
            let hook_result = fire_post_sampling_hooks(&assistant_msg, &tool_ctx.config);
            if !hook_result.blocking_errors.is_empty() {
                if hook_result.prevent_continuation {
                    // Hard veto: push the errors into the conversation and abort.
                    for err_msg in hook_result.blocking_errors {
                        messages.push(err_msg);
                    }
                    if let Some(ref tx) = event_tx {
                        let _ = tx.send(QueryEvent::Status(
                            "PostModelTurn hook vetoed continuation.".to_string(),
                        ));
                    }
                    let last = messages
                        .last()
                        .cloned()
                        .unwrap_or_else(|| Message::assistant("Hook blocked continuation."));
                    return QueryOutcome::EndTurn {
                        message: last,
                        usage,
                    };
                }
                // Soft errors: inject them so the model can react next turn.
                for err_msg in hook_result.blocking_errors {
                    debug!("PostModelTurn hook injecting error message");
                    messages.push(err_msg);
                }
            }
        }

        // Emit token warning events when approaching context limits.
        // Thresholds mirror TypeScript autoCompact.ts: 80% → Warning, 95% → Critical.
        {
            let warning_state =
                compact::calculate_token_warning_state(usage.input_tokens, &config.model);
            if warning_state != compact::TokenWarningState::Ok {
                if let Some(ref tx) = event_tx {
                    let window = compact::context_window_for_model(&config.model);
                    let pct_used = usage.input_tokens as f64 / window as f64;
                    let _ = tx.send(QueryEvent::TokenWarning {
                        state: warning_state,
                        pct_used,
                    });
                }
            }
        }

        // Auto-compact: if context is near-full, summarise older messages now
        // (before the next turn's API call would fail with prompt-too-long).
        //
        // Reactive compact (T1-1): when the CLAUDE_REACTIVE_COMPACT feature gate
        // is enabled, we replace the proactive auto-compact path with reactive
        // compact / context-collapse instead. This fires on every streaming turn
        // so it can act before a prompt-too-long error is returned by the API.
        //
        // Feature gate check: CLAURST_FEATURE_REACTIVE_COMPACT=1
        let reactive_compact_enabled =
            claurst_core::feature_gates::is_feature_enabled("reactive_compact");

        if reactive_compact_enabled {
            // Reactive path: emergency collapse takes priority over normal compact.
            let context_limit = compact::context_window_for_model(&config.model);
            if compact::should_context_collapse(usage.input_tokens, context_limit) {
                if let Some(ref tx) = event_tx {
                    let _ = tx.send(QueryEvent::Status(
                        "Compacting context... (emergency collapse)".to_string(),
                    ));
                }
                match compact::context_collapse(
                    std::mem::take(messages),
                    client,
                    config,
                )
                .await
                {
                    Ok(result) => {
                        *messages = result.messages;
                        info!(
                            tokens_freed = result.tokens_freed,
                            "Context-collapse complete"
                        );
                    }
                    Err(e) => {
                        warn!(error = %e, "Context-collapse failed");
                        // Put messages back on failure (mem::take drained them).
                        // We can't recover them here — re-run auto-compact as fallback.
                    }
                }
            } else if compact::should_compact(usage.input_tokens, context_limit) {
                if let Some(ref tx) = event_tx {
                    let _ = tx.send(QueryEvent::Status("Compacting context...".to_string()));
                }
                match compact::reactive_compact(
                    std::mem::take(messages),
                    client,
                    config,
                    cancel_token.clone(),
                    &[],
                )
                .await
                {
                    Ok(result) => {
                        *messages = result.messages;
                        info!(
                            tokens_freed = result.tokens_freed,
                            "Reactive compact complete"
                        );
                    }
                    Err(claurst_core::error::ClaudeError::Cancelled) => {
                        warn!("Reactive compact was cancelled");
                    }
                    Err(e) => {
                        warn!(error = %e, "Reactive compact failed");
                    }
                }
            }
        } else if stop == "end_turn" || stop == "tool_use" {
            // Proactive auto-compact (original path, used when reactive compact is off).
            if let Some(new_msgs) = compact::auto_compact_if_needed(
                client,
                messages,
                usage.input_tokens,
                &config.model,
                &mut compact_state,
            )
            .await
            {
                *messages = new_msgs;
                if let Some(ref tx) = event_tx {
                    let _ = tx.send(QueryEvent::Status(
                        "Context compacted to stay within limits.".to_string(),
                    ));
                }
            }
        }

        if let Some(ref tx) = event_tx {
            let _ = tx.send(QueryEvent::TurnComplete {
                turn,
                stop_reason: stop.to_string(),
                usage: Some(usage.clone()),
            });
        }

        // Helper closure for firing the Stop hook.
        macro_rules! fire_stop_hook {
            ($msg:expr) => {{
                let stop_ctx = claurst_core::hooks::HookContext {
                    event: "Stop".to_string(),
                    tool_name: None,
                    tool_input: None,
                    tool_output: Some($msg.get_all_text()),
                    is_error: None,
                    session_id: Some(tool_ctx.session_id.clone()),
                };
                claurst_core::hooks::run_hooks(
                    &tool_ctx.config.hooks,
                    claurst_core::config::HookEvent::Stop,
                    &stop_ctx,
                    &tool_ctx.working_dir,
                )
                .await;
            }};
        }

        match stop {
            "end_turn" => {
                fire_stop_hook!(assistant_msg);

                // T1-3: Fire Stop hooks in background (fire-and-forget).
                // `stop_hooks_with_full_behavior` spawns blocking tasks internally
                // and returns immediately with an empty Vec.
                let _bg = stop_hooks_with_full_behavior(
                    &assistant_msg,
                    &tool_ctx.config,
                    tool_ctx.working_dir.clone(),
                );

                // Asynchronously extract and persist session memories if warranted.
                // Runs in a detached Tokio task so it doesn't block the query loop.
                if session_memory::SessionMemoryExtractor::should_extract(messages) {
                    let model_clone = config.model.clone();
                    let messages_clone = messages.clone();
                    let working_dir_clone = tool_ctx.working_dir.clone();

                    // Build a fresh client using the same API key.  This avoids
                    // requiring an Arc in the existing run_query_loop signature.
                    if let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") {
                        if !api_key.is_empty() {
                            if let Ok(sm_client) = claurst_api::AnthropicClient::new(
                                claurst_api::client::ClientConfig {
                                    api_key,
                                    ..Default::default()
                                },
                            ) {
                                let sm_client = std::sync::Arc::new(sm_client);
                                tokio::spawn(async move {
                                    let extractor =
                                        session_memory::SessionMemoryExtractor::new(&model_clone);
                                    match extractor
                                        .extract(&messages_clone, &working_dir_clone, &sm_client)
                                        .await
                                    {
                                        Ok(memories) if !memories.is_empty() => {
                                            let target = working_dir_clone
                                                .join(".claude")
                                                .join("CLAUDE.md");
                                            if let Err(e) =
                                                session_memory::SessionMemoryExtractor::persist(
                                                    &memories, &target,
                                                )
                                                .await
                                            {
                                                tracing::warn!(
                                                    error = %e,
                                                    "Failed to persist session memories"
                                                );
                                            }
                                        }
                                        Ok(_) => {} // no memories extracted
                                        Err(e) => {
                                            tracing::debug!(
                                                error = %e,
                                                "Session memory extraction failed (non-fatal)"
                                            );
                                        }
                                    }
                                });
                            }
                        }
                    }
                }

                // Trigger AutoDream consolidation check (non-blocking, best-effort).
                // maybe_trigger() checks gates + acquires lock. If it returns
                // Some(task), we spawn a background subagent via AgentTool so
                // the spawn doesn't call run_query_loop recursively from within
                // its own future (which would make the future !Send).
                {
                    let memory_dir = dirs::home_dir().map(|h| h.join(".claude").join("memory"));
                    let conversations_dir =
                        dirs::home_dir().map(|h| h.join(".claude").join("conversations"));
                    if let (Some(mem), Some(conv)) = (memory_dir, conversations_dir) {
                        let dreamer = crate::auto_dream::AutoDream::new(mem, conv);
                        if let Ok(Some(task)) = dreamer.maybe_trigger().await {
                            // Run the consolidation subagent in a background Tokio
                            // task. We use the AgentTool execute path (via
                            // poll_background_agent / BACKGROUND_AGENTS) to avoid
                            // re-entering run_query_loop from within the same
                            // future graph.
                            let agent_input = serde_json::json!({
                                "description": "memory consolidation",
                                "prompt": task.prompt,
                                "max_turns": 20,
                                "system_prompt": "You are performing automatic memory consolidation. Complete the task and return a brief summary.",
                                "run_in_background": true,
                                "isolation": null
                            });
                            let ctx_for_dream = tool_ctx.clone();
                            tokio::spawn(async move {
                                let agent = crate::agent_tool::AgentTool;
                                let _result = claurst_tools::Tool::execute(
                                    &agent,
                                    agent_input,
                                    &ctx_for_dream,
                                )
                                .await;
                                crate::auto_dream::AutoDream::finish_consolidation(&task).await;
                            });
                        }
                    }
                }

                return QueryOutcome::EndTurn {
                    message: assistant_msg,
                    usage,
                };
            }
            "max_tokens" => {
                // Mirror the TS recovery loop: inject a continuation nudge and
                // retry up to MAX_TOKENS_RECOVERY_LIMIT times before surfacing
                // the partial response as QueryOutcome::MaxTokens.
                if max_tokens_recovery_count < MAX_TOKENS_RECOVERY_LIMIT {
                    max_tokens_recovery_count += 1;
                    warn!(
                        attempt = max_tokens_recovery_count,
                        limit = MAX_TOKENS_RECOVERY_LIMIT,
                        "max_tokens hit — injecting continuation message (attempt {}/{})",
                        max_tokens_recovery_count,
                        MAX_TOKENS_RECOVERY_LIMIT,
                    );
                    if let Some(ref tx) = event_tx {
                        let _ = tx.send(QueryEvent::Status(format!(
                            "Output token limit hit — continuing (attempt {}/{})",
                            max_tokens_recovery_count, MAX_TOKENS_RECOVERY_LIMIT
                        )));
                    }
                    // The partial assistant message must be in the history so
                    // the continuation makes sense to the model.
                    messages.push(Message::user(MAX_TOKENS_RECOVERY_MSG));
                    continue;
                }
                // Recovery exhausted — surface the partial response.
                warn!(
                    "max_tokens recovery exhausted after {} attempts",
                    MAX_TOKENS_RECOVERY_LIMIT
                );
                return QueryOutcome::MaxTokens {
                    partial_message: assistant_msg,
                    usage,
                };
            }
            "tool_use" => {
                // A completed tool-use turn counts as a successful recovery
                // boundary; reset the max_tokens retry counter.
                max_tokens_recovery_count = 0;
                // Extract tool calls and execute them
                let tool_blocks = assistant_msg.get_tool_use_blocks();
                if tool_blocks.is_empty() {
                    // Shouldn't happen but treat as end_turn
                    return QueryOutcome::EndTurn {
                        message: assistant_msg,
                        usage,
                    };
                }

                // ---------------------------------------------------------------------------
                // Streaming tool executor: parallel non-agent tool dispatch.
                //
                // Phase 1: Run PreToolUse hooks sequentially (they can block/deny execution
                //          and may display interactive permission dialogs).
                // Phase 2: Dispatch all non-blocked tool executions concurrently via
                //          futures::future::join_all, preserving original order.
                // Phase 3: Fire PostToolUse hooks + emit events, then collect results.
                //
                // This mirrors the TypeScript StreamingToolExecutor pattern.
                // ---------------------------------------------------------------------------

                // Intermediate record produced during Phase 1.
                struct PreparedTool {
                    id: String,
                    name: String,
                    input: Value,
                    /// None means the pre-hook blocked execution; the String is the error reason.
                    blocked_result: Option<ToolResult>,
                }

                // Phase 1: sequential pre-hook pass.
                let mut prepared: Vec<PreparedTool> = Vec::with_capacity(tool_blocks.len());
                for block in tool_blocks {
                    if let ContentBlock::ToolUse { id, name, input } = block {
                        // Clone from the references returned by get_tool_use_blocks()
                        let id = id.clone();
                        let name = name.clone();
                        let input = input.clone();

                        if let Some(ref tx) = event_tx {
                            let _ = tx.send(QueryEvent::ToolStart {
                                tool_name: name.clone(),
                                tool_id: id.clone(),
                                input_json: input.to_string(),
                            });
                        }

                        let hooks = &tool_ctx.config.hooks;
                        let hook_ctx = claurst_core::hooks::HookContext {
                            event: "PreToolUse".to_string(),
                            tool_name: Some(name.clone()),
                            tool_input: Some(input.clone()),
                            tool_output: None,
                            is_error: None,
                            session_id: Some(tool_ctx.session_id.clone()),
                        };
                        let pre_outcome = claurst_core::hooks::run_hooks(
                            hooks,
                            claurst_core::config::HookEvent::PreToolUse,
                            &hook_ctx,
                            &tool_ctx.working_dir,
                        )
                        .await;

                        let plugin_pre_outcome =
                            claurst_plugins::run_global_pre_tool_hook(&name, &input);

                        let blocked_result =
                            if let claurst_core::hooks::HookOutcome::Blocked(reason) = pre_outcome {
                                warn!(tool = %name, reason = %reason, "PreToolUse hook blocked execution");
                                Some(claurst_tools::ToolResult::error(format!(
                                    "Blocked by hook: {}",
                                    reason
                                )))
                            } else if let claurst_plugins::HookOutcome::Deny(reason) = plugin_pre_outcome {
                                warn!(tool = %name, reason = %reason, "Plugin PreToolUse hook blocked execution");
                                Some(claurst_tools::ToolResult::error(format!(
                                    "Blocked by plugin hook: {}",
                                    reason
                                )))
                            } else {
                                None
                            };

                        prepared.push(PreparedTool {
                            id,
                            name,
                            input,
                            blocked_result,
                        });
                    }
                }

                // Phase 2: build execution futures for non-blocked tools and join them.
                // Blocked tools yield a ready future with the pre-computed error result.
                // Non-blocked tools execute concurrently via join_all.
                // Each async block owns its cloned name/input so there are no lifetime issues.
                let exec_futures: Vec<_> = prepared
                    .iter()
                    .map(|p| {
                        if p.blocked_result.is_some() {
                            let r = p.blocked_result.clone().unwrap();
                            futures::future::Either::Left(async move { r })
                        } else {
                            let name = p.name.clone();
                            let input = p.input.clone();
                            futures::future::Either::Right(async move {
                                execute_tool(&name, &input, tools, tool_ctx).await
                            })
                        }
                    })
                    .collect();

                // Run all tool futures concurrently; join_all preserves order.
                let exec_results: Vec<ToolResult> =
                    futures::future::join_all(exec_futures).await;

                // Phase 3: post-hooks, event emission, and result block assembly.
                let mut result_blocks: Vec<ContentBlock> =
                    Vec::with_capacity(prepared.len());
                for (p, result) in prepared.iter().zip(exec_results.into_iter()) {
                    let hooks = &tool_ctx.config.hooks;
                    let post_ctx = claurst_core::hooks::HookContext {
                        event: "PostToolUse".to_string(),
                        tool_name: Some(p.name.clone()),
                        tool_input: Some(p.input.clone()),
                        tool_output: Some(result.content.clone()),
                        is_error: Some(result.is_error),
                        session_id: Some(tool_ctx.session_id.clone()),
                    };
                    claurst_core::hooks::run_hooks(
                        hooks,
                        claurst_core::config::HookEvent::PostToolUse,
                        &post_ctx,
                        &tool_ctx.working_dir,
                    )
                    .await;

                    claurst_plugins::run_global_post_tool_hook(
                        &p.name,
                        &p.input,
                        &result.content,
                        result.is_error,
                    );

                    if let Some(ref tx) = event_tx {
                        let _ = tx.send(QueryEvent::ToolEnd {
                            tool_name: p.name.clone(),
                            tool_id: p.id.clone(),
                            result: result.content.clone(),
                            is_error: result.is_error,
                        });
                    }

                    result_blocks.push(ContentBlock::ToolResult {
                        tool_use_id: p.id.clone(),
                        content: ToolResultContent::Text(result.content),
                        is_error: if result.is_error { Some(true) } else { None },
                    });
                }

                // Append tool results as a user message
                messages.push(Message::user_blocks(result_blocks));

                // Continue the loop to send results back to the model
                continue;
            }
            "stop_sequence" => {
                fire_stop_hook!(assistant_msg);
                let _bg = stop_hooks_with_full_behavior(
                    &assistant_msg,
                    &tool_ctx.config,
                    tool_ctx.working_dir.clone(),
                );
                return QueryOutcome::EndTurn {
                    message: assistant_msg,
                    usage,
                };
            }
            other => {
                warn!(stop_reason = other, "Unknown stop reason, treating as end_turn");
                fire_stop_hook!(assistant_msg);
                let _bg = stop_hooks_with_full_behavior(
                    &assistant_msg,
                    &tool_ctx.config,
                    tool_ctx.working_dir.clone(),
                );
                return QueryOutcome::EndTurn {
                    message: assistant_msg,
                    usage,
                };
            }
        }
    }
}

/// Execute a single tool invocation.
async fn execute_tool(
    name: &str,
    input: &Value,
    tools: &[Box<dyn Tool>],
    ctx: &ToolContext,
) -> ToolResult {
    let tool = tools.iter().find(|t| t.name() == name);

    match tool {
        Some(tool) => {
            debug!(tool = name, "Executing tool");
            tool.execute(input.clone(), ctx).await
        }
        None => {
            warn!(tool = name, "Unknown tool requested");
            ToolResult::error(format!("Unknown tool: {}", name))
        }
    }
}

/// Load persisted todos for `session_id` and return a nudge string if any are
/// incomplete (status != "completed"). Returns empty string otherwise.
fn build_todo_nudge(session_id: &str) -> String {
    let todos = claurst_tools::todo_write::load_todos(session_id);
    let incomplete_count = todos
        .iter()
        .filter(|t| t["status"].as_str().map_or(true, |s| s != "completed"))
        .count();
    if incomplete_count == 0 {
        String::new()
    } else {
        format!(
            "You have {} incomplete task{} in your TodoWrite list. \
             Make sure to complete all tasks before ending your response.",
            incomplete_count,
            if incomplete_count == 1 { "" } else { "s" }
        )
    }
}

/// Build the system prompt from config.
///
/// Delegates to `claurst_core::system_prompt::build_system_prompt` so that all
/// default content (capabilities, safety guidelines, dynamic-boundary marker,
/// etc.) is assembled in one place.  The `QueryConfig` fields map directly to
/// `SystemPromptOptions`:
///
/// - `system_prompt`        → `custom_system_prompt` (added to cacheable block)
/// - `append_system_prompt` → `append_system_prompt` (added after boundary)
fn build_system_prompt(config: &QueryConfig) -> SystemPrompt {
    use claurst_core::system_prompt::SystemPromptOptions;

    let opts = SystemPromptOptions {
        custom_system_prompt: config.system_prompt.clone(),
        append_system_prompt: config.append_system_prompt.clone(),
        // All other fields use sensible defaults:
        // - prefix:                auto-detect from env
        // - memory_content:        empty (callers inject via append if needed)
        // - replace_system_prompt: false (additive mode)
        // - coordinator_mode:      false
        output_style: config.output_style,
        custom_output_style_prompt: config.output_style_prompt.clone(),
        working_directory: config.working_directory.clone(),
        ..Default::default()
    };

    let text = claurst_core::system_prompt::build_system_prompt(&opts);
    SystemPrompt::Text(text)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use claurst_api::SystemPrompt;

    fn make_config(sys: Option<&str>, append: Option<&str>) -> QueryConfig {
        QueryConfig {
            model: "claude-sonnet-4-6".to_string(),
            max_tokens: 4096,
            max_turns: 10,
            system_prompt: sys.map(String::from),
            append_system_prompt: append.map(String::from),
            output_style: claurst_core::system_prompt::OutputStyle::Default,
            output_style_prompt: None,
            working_directory: None,
            thinking_budget: None,
            temperature: None,
            tool_result_budget: 50_000,
            effort_level: None,
            command_queue: None,
            skill_index: None,
            max_budget_usd: None,
            fallback_model: None,
        }
    }

    // ---- build_system_prompt tests ------------------------------------------

    #[test]
    fn test_system_prompt_default_when_empty() {
        // The default prompt (no custom system prompt set) should include the
        // Claurst attribution and standard sections.
        let cfg = make_config(None, None);
        let prompt = build_system_prompt(&cfg);
        if let SystemPrompt::Text(text) = prompt {
            assert!(
                text.contains("Claurst") || text.contains("Claude agent"),
                "Default prompt should contain attribution: {}",
                text
            );
            assert!(
                text.contains(claurst_core::system_prompt::SYSTEM_PROMPT_DYNAMIC_BOUNDARY),
                "Default prompt must contain the dynamic boundary marker"
            );
        } else {
            panic!("Expected SystemPrompt::Text");
        }
    }

    #[test]
    fn test_system_prompt_with_custom() {
        // A custom system prompt is injected into the cacheable section as
        // <custom_instructions>; the default sections are still present.
        let cfg = make_config(Some("You are a code reviewer."), None);
        let prompt = build_system_prompt(&cfg);
        if let SystemPrompt::Text(text) = prompt {
            assert!(
                text.contains("You are a code reviewer."),
                "Custom prompt text should appear in the output"
            );
            assert!(
                text.contains("Claurst") || text.contains("Claude agent"),
                "Default attribution should still be present"
            );
        } else {
            panic!("Expected SystemPrompt::Text");
        }
    }

    #[test]
    fn test_system_prompt_with_append() {
        // Appended text lands after the dynamic boundary.
        let cfg = make_config(Some("Base prompt."), Some("Additional context."));
        let prompt = build_system_prompt(&cfg);
        if let SystemPrompt::Text(text) = prompt {
            assert!(text.contains("Base prompt."));
            assert!(text.contains("Additional context."));
            // append_system_prompt appears after the boundary
            let boundary_pos = text
                .find(claurst_core::system_prompt::SYSTEM_PROMPT_DYNAMIC_BOUNDARY)
                .expect("boundary must exist");
            let append_pos = text.find("Additional context.").unwrap();
            assert!(
                append_pos > boundary_pos,
                "Appended text must appear after the dynamic boundary"
            );
        } else {
            panic!("Expected SystemPrompt::Text");
        }
    }

    #[test]
    fn test_system_prompt_append_only() {
        // When only append is set, default sections are present plus the
        // appended text after the dynamic boundary.
        let cfg = make_config(None, Some("Appended text."));
        let prompt = build_system_prompt(&cfg);
        if let SystemPrompt::Text(text) = prompt {
            assert!(
                text.contains("Appended text."),
                "Appended text must appear in the prompt"
            );
            let boundary_pos = text
                .find(claurst_core::system_prompt::SYSTEM_PROMPT_DYNAMIC_BOUNDARY)
                .expect("boundary must exist");
            let append_pos = text.find("Appended text.").unwrap();
            assert!(
                append_pos > boundary_pos,
                "Appended text must appear after the dynamic boundary"
            );
        } else {
            panic!("Expected SystemPrompt::Text");
        }
    }

    #[test]
    fn test_system_prompt_with_custom_output_style_prompt() {
        let mut cfg = make_config(None, None);
        cfg.output_style_prompt = Some("Answer like a pirate.".to_string());
        let prompt = build_system_prompt(&cfg);
        if let SystemPrompt::Text(text) = prompt {
            assert!(text.contains("Answer like a pirate."));
        } else {
            panic!("Expected SystemPrompt::Text");
        }
    }

    // ---- QueryConfig tests --------------------------------------------------

    #[test]
    fn test_query_config_clone() {
        let cfg = make_config(Some("test"), Some("append"));
        let cloned = cfg.clone();
        assert_eq!(cloned.model, "claude-sonnet-4-6");
        assert_eq!(cloned.max_tokens, 4096);
        assert_eq!(cloned.system_prompt, Some("test".to_string()));
    }

    // ---- QueryOutcome variant tests -----------------------------------------

    #[test]
    fn test_query_outcome_debug() {
        // Ensure the enum variants can be created and debug-formatted
        let outcome = QueryOutcome::Cancelled;
        let s = format!("{:?}", outcome);
        assert!(s.contains("Cancelled"));

        let err_outcome = QueryOutcome::Error(claurst_core::error::ClaudeError::RateLimit);
        let s2 = format!("{:?}", err_outcome);
        assert!(s2.contains("Error"));
    }
}

/// Stream handler that forwards events to an unbounded channel.
struct ChannelStreamHandler {
    tx: mpsc::UnboundedSender<QueryEvent>,
}

impl StreamHandler for ChannelStreamHandler {
    fn on_event(&self, event: &AnthropicStreamEvent) {
        let _ = self.tx.send(QueryEvent::Stream(event.clone()));
    }
}

// ---------------------------------------------------------------------------
// Single-shot query (non-looping, for simple one-off calls)
// ---------------------------------------------------------------------------

/// Run a single (non-agentic) query – no tool loop, just one API call.
pub async fn run_single_query(
    client: &claurst_api::AnthropicClient,
    messages: Vec<Message>,
    config: &QueryConfig,
) -> Result<Message, ClaudeError> {
    let api_messages: Vec<ApiMessage> = messages.iter().map(ApiMessage::from).collect();
    let system = build_system_prompt(config);

    let request = CreateMessageRequest::builder(&config.model, config.max_tokens)
        .messages(api_messages)
        .system(system)
        .build();

    let handler: Arc<dyn StreamHandler> = Arc::new(claurst_api::streaming::NullStreamHandler);

    let mut rx = client.create_message_stream(request, handler).await?;
    let mut acc = StreamAccumulator::new();

    while let Some(evt) = rx.recv().await {
        acc.on_event(&evt);
        if matches!(evt, AnthropicStreamEvent::MessageStop) {
            break;
        }
    }

    let (msg, _usage, _stop) = acc.finish();
    Ok(msg)
}
