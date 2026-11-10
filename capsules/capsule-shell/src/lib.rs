#![deny(unsafe_code)]
#![deny(clippy::all)]
#![deny(unreachable_pub)]
#![allow(missing_docs)]

//! Shell execution tools capsule for Unicity AOS.
//!
//! Provides the `run_shell_command` tool to agents, wrapping executions
//! securely in the host-level Escape Hatch (Seatbelt/bwrap).
//!
//! Each command is parsed to extract an approval action: the program name
//! plus subcommand tokens for known multi-command tools. Approved commands
//! create allowances at that granularity (e.g. "git push" approves all
//! `git push` variants, "docker compose up" approves all
//! `docker compose up` variants). Unknown programs use program-name-only
//! to avoid leaking positional arguments into allowance patterns.

mod safety;

use astrid_sdk::prelude::*;
use astrid_sdk::schemars;
use safety::{check_catastrophic, split_on_shell_operators};
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// The main entry point for the Shell Tools capsule.
///
/// Holds a registry mapping integer IDs (returned to the LLM agent) to
/// the live RAII `process::Process` handles. Tool calls are stateless
/// invocations on a `&'static ShellTools`, but the post-PR#752 SDK
/// expresses background processes as Drop-managed resources owned by
/// the capsule. Storing handles here keeps them alive across tool
/// calls while still giving the host the resource it needs to track.
#[derive(Default)]
pub struct ShellTools {
    /// Live background process handles, keyed by the integer ID the
    /// LLM agent receives. Dropping a handle releases the kernel-side
    /// process resource. Entries are removed only by explicit
    /// `kill_process` — `read_process_logs` reports exit state without
    /// freeing the slot so the agent can drain final logs across
    /// multiple reads before calling `kill_process` to release.
    background: Mutex<HashMap<u64, process::Process>>,
    /// Monotonic counter for assigning IDs to background processes.
    /// Per-capsule; capsules are reloaded with a fresh counter, which
    /// is fine — IDs only need to be unique within a capsule lifetime.
    next_id: AtomicU64,
}

/// Input arguments for the `run_shell_command` tool.
#[derive(Debug, Default, Deserialize, schemars::JsonSchema)]
pub struct RunShellArgs {
    /// The exact shell command to execute (POSIX sh on hosts without Bash).
    pub command: String,
}

/// Input arguments for the `spawn_background_process` tool.
#[derive(Debug, Default, Deserialize, schemars::JsonSchema)]
pub struct SpawnBackgroundArgs {
    /// The exact shell command to run in the background (POSIX sh without Bash).
    pub command: String,
}

/// Input arguments for the `read_process_logs` tool.
#[derive(Debug, Default, Deserialize, schemars::JsonSchema)]
pub struct ReadProcessLogsArgs {
    /// The process handle ID returned by `spawn_background_process`.
    pub id: u64,
}

/// Input arguments for the `kill_process` tool.
#[derive(Debug, Default, Deserialize, schemars::JsonSchema)]
pub struct KillProcessArgs {
    /// The process handle ID returned by `spawn_background_process`.
    pub id: u64,
}

/// Determine the safe subcommand depth for a given program.
fn get_safe_depth(tokens: &[&str]) -> usize {
    if tokens.is_empty() {
        return 0;
    }

    // Depth 3 (Sub-sub-commands)
    if tokens.len() >= 2 {
        let prefix2 = format!("{} {}", tokens[0], tokens[1]);
        if matches!(
            prefix2.as_str(),
            "docker compose"
                | "aws s3"
                | "gcloud s3"
                | "kubectl config"
                | "git remote"
                | "npm run"
                | "yarn run"
                | "pnpm run"
                | "bun run"
                | "deno run"
        ) {
            return 3;
        }
    }

    // Depth 2 (Standard CLI subcommands)
    if matches!(
        tokens[0],
        "git"
            | "docker"
            | "kubectl"
            | "cargo"
            | "npm"
            | "npx"
            | "yarn"
            | "pnpm"
            | "pip"
            | "pip3"
            | "poetry"
            | "apt"
            | "apt-get"
            | "brew"
            | "dnf"
            | "yum"
            | "pacman"
            | "snap"
            | "flatpak"
            | "helm"
            | "terraform"
            | "ansible"
            | "vagrant"
            | "make"
            | "cmake"
            | "go"
            | "rustup"
            | "bun"
            | "deno"
            | "uv"
            | "nix"
            | "podman"
            | "gh"
            | "fly"
            | "flyctl"
            | "stripe"
            | "supabase"
            | "vercel"
            | "wrangler"
            | "firebase"
    ) {
        return 2;
    }

    // Default fallback: 0 means exact match only (no subcommands extracted).
    0
}

/// Extract the approval action from a shell command string.
///
/// Uses an exhaustive whitelist to determine how many subcommand tokens
/// are safe to include in the allowance pattern. For unknown programs,
/// returns the exact full command string.
///
/// Returning the full string for unknown programs is a critical security
/// boundary. If `rm -rf /tmp/foo` returned just `rm`, the session allowance
/// would be `rm *`, which would allow a malicious capsule to execute `rm -rf /`
/// without prompting. By returning the full string, the allowance becomes
/// safely scoped to `rm -rf /tmp/foo *`.
///
/// ```text
/// git push --force origin main      -> "git push"
/// docker compose up -d              -> "docker compose up"
/// kubectl config set-context --cur  -> "kubectl config set-context"
/// ls -la /tmp                       -> "ls -la /tmp"
/// cargo build --release             -> "cargo build"
/// python -c "code"                  -> "python -c \"code\""
/// cat /etc/passwd                   -> "cat /etc/passwd"
/// rm -rf /tmp/foo                   -> "rm -rf /tmp/foo"
/// rm /tmp/foo                       -> "rm /tmp/foo"
/// ```
fn extract_action(command: &str) -> String {
    if split_on_shell_operators(command).len() > 1 {
        // SECURITY: Chained or multi-line commands must use exact-match
        // fallback so a second command cannot piggyback on the first
        // program's subcommand allowance (e.g. "git status && ...").
        return command.to_string();
    }

    let tokens: Vec<&str> = command.split_whitespace().collect();
    if tokens.is_empty() || tokens[0].starts_with('-') {
        return String::new();
    }

    let depth = get_safe_depth(&tokens);
    if depth == 0 {
        // SECURITY: Unknown programs must use exact-match fallback to prevent
        // generating dangerously broad `program *` glob allowances.
        return command.to_string();
    }

    let mut parts = Vec::new();
    for token in tokens {
        if token.starts_with('-') || parts.len() == depth {
            break;
        }
        parts.push(token);
    }
    parts.join(" ")
}

// The approved command is passed as a separate argv element.  POSIX sh
// chooses the interpreter inside the same sandbox, then execs it exactly
// once.  A failed host gate or Bash launch cannot replay user code.
const SHELL_LAUNCHER: &str =
    "if command -v bash >/dev/null 2>&1; then exec bash -c \"$1\"; else exec sh -c \"$1\"; fi";

fn shell_args(command: &str) -> [&str; 4] {
    ["-c", SHELL_LAUNCHER, "aos-shell", command]
}

/// Sandboxed shell execution tools for running commands and managing processes.
///
/// All commands run through the host's sandbox escape hatch (Seatbelt on macOS,
/// bubblewrap on Linux). Catastrophic commands (rm -rf /, mkfs, fork bombs) are
/// hard-blocked before reaching the approval prompt. All other commands require
/// human approval.
#[capsule]
impl ShellTools {
    /// Execute a shell command and return its output. The command runs in Bash
    /// when available, or POSIX sh on hosts without Bash.
    /// via the host sandbox. Requires human approval — the approval prompt
    /// shows the command and its risk level. Use this for running tests,
    /// builds, git operations, or any CLI tool.
    ///
    /// Returns stdout on success. Returns an error with stderr on non-zero
    /// exit code so you can see what went wrong.
    #[astrid::tool("run_shell_command")]
    pub fn run_shell_command(&self, args: RunShellArgs) -> Result<String, SysError> {
        let trimmed = args.command.trim();
        if trimmed.is_empty() {
            return Err(SysError::ApiError("Command cannot be empty".into()));
        }

        // Hard-deny catastrophic commands before they reach the approval prompt.
        if let Some(reason) = check_catastrophic(trimmed) {
            return Err(SysError::ApiError(format!("Command blocked: {reason}")));
        }

        let action = extract_action(trimmed);

        // Request approval - blocks until the user responds or timeout.
        let approved = approval::request(&action, trimmed)?;
        if !approved {
            return Err(SysError::ApiError(format!(
                "Command '{trimmed}' was not approved by user",
            )));
        }

        // Spawn the command via the SDK Process Airlock.
        // The core OS enforces the Capability and wraps it in bwrap/Seatbelt.
        let result = process::spawn("sh", &shell_args(trimmed))?;

        // If the command fails, we return the stderr as an API error so the LLM knows it failed.
        // `Output.exit` carries the typed `ExitInfo` so we can tell signal kills from
        // normal exits; both are reported to the LLM with the relevant detail.
        if !result.exit.success() {
            let detail = match (result.exit.exit_code, result.exit.signal) {
                (Some(code), _) => format!("exit code {code}"),
                (None, Some(sig)) => format!("killed by signal {sig}"),
                (None, None) => "unknown termination".to_string(),
            };
            return Err(SysError::ApiError(format!(
                "Command failed ({detail}): {}",
                result.stderr
            )));
        }

        // Return stdout back to the LLM agent
        Ok(result.stdout)
    }

    /// Start a long-running process in the background. Returns a message
    /// containing the process handle ID. Use this for dev servers, watch processes, or anything that shouldn't
    /// block the conversation. Requires human approval. Use `read_process_logs`
    /// to check output and `kill_process` to stop it.
    #[astrid::tool("spawn_background_process")]
    pub fn spawn_background_process(&self, args: SpawnBackgroundArgs) -> Result<String, SysError> {
        let trimmed = args.command.trim();
        if trimmed.is_empty() {
            return Err(SysError::ApiError("Command cannot be empty".into()));
        }

        if let Some(reason) = check_catastrophic(trimmed) {
            return Err(SysError::ApiError(format!("Command blocked: {reason}")));
        }

        let action = extract_action(trimmed);

        let approved = approval::request(&action, trimmed)?;
        if !approved {
            return Err(SysError::ApiError(format!(
                "Command '{trimmed}' was not approved by user",
            )));
        }

        // `Process` is a Drop-managed kernel resource — we stash it in the
        // per-capsule registry so it stays alive across tool calls and the
        // LLM can continue to address it by integer ID. Dropping the
        // `Process` (on kill or final read) releases the kernel slot.
        let handle = process::spawn_background("sh", &shell_args(trimmed))?;
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut map = self
            .background
            .lock()
            .map_err(|_| SysError::ApiError("background process registry lock poisoned".into()))?;
        map.insert(id, handle);
        Ok(format!(
            "Background process started with id: {id}. Use read_process_logs to check output and kill_process to stop it.",
        ))
    }

    /// Read buffered stdout/stderr from a background process. Each call
    /// returns only the new output since the last read. Also reports whether
    /// the process is still running and its exit code if terminated. No
    /// approval required — the process was already approved at spawn time.
    #[astrid::tool("read_process_logs")]
    pub fn read_process_logs(&self, args: ReadProcessLogsArgs) -> Result<String, SysError> {
        // Look up the live `Process` handle in the registry; read_logs is a
        // method on the resource in the post-PR#752 SDK.
        let map = self
            .background
            .lock()
            .map_err(|_| SysError::ApiError("background process registry lock poisoned".into()))?;
        let handle = map.get(&args.id).ok_or_else(|| {
            SysError::ApiError(format!(
                "no background process with id {} (already killed or never started)",
                args.id,
            ))
        })?;

        let logs = handle.read_logs()?;

        let status = if logs.running {
            "running".to_string()
        } else {
            match logs.exit.as_ref().map(|e| (e.exit_code, e.signal)) {
                Some((Some(code), _)) => format!("exited with code {code}"),
                Some((None, Some(sig))) => format!("killed by signal {sig}"),
                Some((None, None)) | None => "exited (unknown termination)".to_string(),
            }
        };

        let mut output = format!("Process {} status: {status}\n", args.id);
        if !logs.running {
            output.push_str(
                "(Process has exited. Call kill_process to release the handle and free the slot.)\n",
            );
        }
        use std::fmt::Write;
        if !logs.stdout.is_empty() {
            let _ = write!(&mut output, "--- stdout ---\n{}\n", logs.stdout);
        }
        if !logs.stderr.is_empty() {
            let _ = write!(&mut output, "--- stderr ---\n{}\n", logs.stderr);
        }
        if logs.stdout.is_empty() && logs.stderr.is_empty() {
            let _ = writeln!(&mut output, "(no new output)");
        }

        Ok(output)
    }

    /// Terminate a background process and return any remaining output.
    /// Releases the process handle slot. No additional approval required.
    #[astrid::tool("kill_process")]
    pub fn kill_process(&self, args: KillProcessArgs) -> Result<String, SysError> {
        // Remove the handle from the registry up front so the kernel-side
        // `process-handle` resource is dropped after we extract its final
        // logs — that's what "releases the handle slot" means in the new
        // resource-oriented model.
        let handle = {
            let mut map = self.background.lock().map_err(|_| {
                SysError::ApiError("background process registry lock poisoned".into())
            })?;
            map.remove(&args.id).ok_or_else(|| {
                SysError::ApiError(format!(
                    "no background process with id {} (already killed or never started)",
                    args.id,
                ))
            })?
        };

        let result = handle.kill()?;

        let exit_info = match result.exit.as_ref().map(|e| (e.exit_code, e.signal)) {
            Some((Some(code), _)) => format!("exit code {code}"),
            Some((None, Some(sig))) => format!("killed by signal {sig}"),
            Some((None, None)) | None => "unknown termination".to_string(),
        };

        let mut output = format!("Process {} killed ({exit_info}).\n", args.id);
        use std::fmt::Write;
        if !result.stdout.is_empty() {
            let _ = write!(&mut output, "--- final stdout ---\n{}\n", result.stdout);
        }
        if !result.stderr.is_empty() {
            let _ = write!(&mut output, "--- final stderr ---\n{}\n", result.stderr);
        }

        // `handle` falls out of scope here and Drop releases the kernel slot.
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn shell_launcher_executes_exactly_once_without_bash() {
        use std::os::unix::fs::symlink;
        use std::process::Command;

        let path = std::env::temp_dir().join(format!(
            "aos-shell-no-bash-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&path).unwrap();
        symlink("/bin/sh", path.join("sh")).unwrap();
        let output = Command::new("/bin/sh")
            .args(shell_args("printf 'one $HOME ; \"quoted\" $1';\nexit 127"))
            .env("PATH", &path)
            .output()
            .unwrap();
        std::fs::remove_dir_all(path).unwrap();
        assert_eq!(output.status.code(), Some(127));
        assert_eq!(output.stdout, b"one $HOME ; \"quoted\" $1");
    }

    #[cfg(unix)]
    #[test]
    fn shell_launcher_uses_bash_when_present() {
        use std::os::unix::fs::symlink;
        use std::process::Command;

        let bash = ["/bin/bash", "/usr/bin/bash"]
            .into_iter()
            .find(|path| std::path::Path::new(path).is_file());
        let Some(bash) = bash else { return };
        let path = std::env::temp_dir().join(format!(
            "aos-shell-with-bash-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&path).unwrap();
        symlink("/bin/sh", path.join("sh")).unwrap();
        symlink(bash, path.join("bash")).unwrap();
        let output = Command::new("/bin/sh")
            .args(shell_args("[[ 1 == 1 ]] && printf bash"))
            .env("PATH", &path)
            .output()
            .unwrap();
        std::fs::remove_dir_all(path).unwrap();
        assert!(output.status.success(), "{:?}", output.stderr);
        assert_eq!(output.stdout, b"bash");
    }

    // -- Subcommand extraction (depth 1) --

    #[test]
    fn action_git_push() {
        assert_eq!(extract_action("git push --force origin main"), "git push");
    }

    #[test]
    fn action_git_status() {
        assert_eq!(extract_action("git status"), "git status");
    }

    #[test]
    fn action_cargo_build() {
        assert_eq!(extract_action("cargo build --release"), "cargo build");
    }

    // -- Sub-sub-command extraction (depth 2) --

    #[test]
    fn action_docker_compose_up() {
        assert_eq!(extract_action("docker compose up -d"), "docker compose up");
    }

    #[test]
    fn action_kubectl_config_set_context() {
        assert_eq!(
            extract_action("kubectl config set-context --current"),
            "kubectl config set-context"
        );
    }

    #[test]
    fn action_git_remote_add() {
        assert_eq!(
            extract_action("git remote add origin https://example.com"),
            "git remote add"
        );
    }

    // -- Depth cap (stops at MAX_SUBCOMMAND_DEPTH + 1 tokens for known programs) --

    #[test]
    fn action_depth_cap_known_program() {
        // npm is in the whitelist; 3rd non-flag token is NOT included
        assert_eq!(extract_action("npm run build dist"), "npm run build");
    }

    // -- Unknown programs return full command (exact-match safety) --

    #[test]
    fn action_unknown_program_returns_full_command() {
        // Unknown programs return the full command string to prevent
        // dangerously broad "program *" session allowances.
        assert_eq!(extract_action("cat /etc/passwd"), "cat /etc/passwd");
        assert_eq!(extract_action("rm /tmp/foo"), "rm /tmp/foo");
        assert_eq!(extract_action("rm -rf /tmp/foo"), "rm -rf /tmp/foo");
        assert_eq!(extract_action("a b c d e"), "a b c d e");
    }

    // -- Flag stops extraction (known programs only) --

    #[test]
    fn action_known_program_flag_stops() {
        // Flags stop subcommand extraction for whitelisted programs
        assert_eq!(extract_action("cargo build --release"), "cargo build");
        assert_eq!(extract_action("git push --force origin main"), "git push");
    }

    #[test]
    fn action_unknown_program_with_flags_returns_full() {
        // Unknown programs always return full command, flags included
        assert_eq!(extract_action("ls -la /tmp"), "ls -la /tmp");
        assert_eq!(
            extract_action("python -c 'print(1)'"),
            "python -c 'print(1)'"
        );
        assert_eq!(
            extract_action("my-tool --flag value"),
            "my-tool --flag value"
        );
    }

    // -- Edge cases --

    #[test]
    fn action_empty() {
        assert_eq!(extract_action(""), "");
    }

    #[test]
    fn action_single_word() {
        assert_eq!(extract_action("git"), "git");
    }

    #[test]
    fn action_only_flags() {
        assert_eq!(extract_action("--help"), "");
    }

    #[test]
    fn action_chained_known_program_returns_full_command() {
        // Chained commands starting with a whitelisted program must not
        // collapse to the first program's subcommand allowance.
        assert_eq!(
            extract_action("git status && rm -rf /tmp/foo"),
            "git status && rm -rf /tmp/foo"
        );
        assert_eq!(
            extract_action("cargo build; sh deploy.sh"),
            "cargo build; sh deploy.sh"
        );
        assert_eq!(extract_action("git status | sh"), "git status | sh");
        assert_eq!(
            extract_action("git status & rm -rf /tmp/foo"),
            "git status & rm -rf /tmp/foo"
        );
        assert_eq!(
            extract_action("git status\nrm -rf /tmp/foo"),
            "git status\nrm -rf /tmp/foo"
        );
    }
}
