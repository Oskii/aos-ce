//! Shell command safety checks for blocking catastrophic commands before approval.

use std::collections::VecDeque;

/// System-critical paths that must never be targets of destructive commands
/// from an AI agent. Relative paths and workspace-local paths are fine.
const PROTECTED_PATHS: &[&str] = &[
    "/",
    "/*",
    "/etc",
    "/usr",
    "/bin",
    "/sbin",
    "/var",
    "/boot",
    "/lib",
    "/lib64",
    "/opt",
    "/root",
    "/home",
    "/proc",
    "/sys",
    "/dev",
    // macOS
    "/System",
    "/Library",
    "/Applications",
    "/Users",
];

/// Commands that are unconditionally blocked regardless of arguments.
const BLOCKED_COMMANDS: &[&str] = &[
    "mkfs",
    "mkfs.ext4",
    "mkfs.xfs",
    "mkfs.btrfs",
    "mkfs.vfat",
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
    "init",
];

/// Wrapper utilities that delegate execution to a subsequent command token.
const COMMAND_WRAPPERS: &[&str] = &["sudo", "env", "command", "nohup", "nice", "exec", "time"];

/// Maximum number of `env -S` / `--split-string` expansions per command.
const MAX_SPLIT_STRING_EXPANSIONS: usize = 8;

#[derive(Clone, Copy, PartialEq, Eq)]
enum QuoteState {
    None,
    Single,
    Double,
}

/// Shell operators that chain multiple commands. We split on these to check
/// each sub-command independently for catastrophic patterns.
const SHELL_CHAIN_OPERATORS: &[&str] = &["&&", "||", ";", "|", "\r\n", "\n", "\r"];

/// Check if a command is catastrophic and should be hard-denied before
/// reaching the approval prompt.
///
/// Splits on shell chaining operators (`&&`, `||`, `;`, `|`, `&`, and newlines)
/// and checks each sub-command independently. `ls && rm -rf /` is caught even
/// though the first command is benign.
///
/// Returns `Some(reason)` if the command is blocked, `None` if safe to proceed.
pub(crate) fn check_catastrophic(command: &str) -> Option<&'static str> {
    // Fork bomb patterns checked on the raw string before splitting
    if command.contains(":(){ :|:&") || command.contains(":(){:|:&") {
        return Some("Fork bombs are blocked.");
    }

    // Split on shell chaining operators and check each sub-command
    for subcmd in split_on_shell_operators(command) {
        if let Some(reason) = check_single_command_catastrophic(subcmd.trim()) {
            return Some(reason);
        }
    }

    None
}

/// Split a command string on shell chaining operators.
pub(crate) fn split_on_shell_operators(command: &str) -> Vec<&str> {
    // Replace multi-char operators with a single sentinel, then split.
    // We need to handle &&, ||, ;, |, newlines, and standalone & -
    // longer operators are processed first so || and && are split before | and &.
    let mut result = vec![command];
    for op in SHELL_CHAIN_OPERATORS {
        let mut next = Vec::new();
        for segment in result {
            for part in segment.split(op) {
                next.push(part);
            }
        }
        result = next;
    }

    let mut final_parts = Vec::new();
    for segment in result {
        final_parts.extend(split_on_background_operator(segment));
    }
    final_parts
}

/// Split a segment on standalone `&` background operators without breaking
/// file-descriptor redirections such as `2>&1`, `<&0`, or `&>`.
fn split_on_background_operator(segment: &str) -> Vec<&str> {
    let bytes = segment.as_bytes();
    let mut parts = Vec::new();
    let mut start = 0;
    for (i, &byte) in bytes.iter().enumerate() {
        if byte == b'&' {
            let prev_redir = i > 0 && matches!(bytes[i - 1], b'>' | b'<');
            let next_redir = i + 1 < bytes.len() && bytes[i + 1] == b'>';
            if !prev_redir && !next_redir {
                parts.push(&segment[start..i]);
                start = i + 1;
            }
        }
    }
    parts.push(&segment[start..]);
    parts
}

/// Tokenize a single shell command into unquoted words, preserving spaces
/// inside single/double quotes and resolving backslash escapes.
fn tokenize_shell_words(command: &str) -> Vec<String> {
    tokenize_words_internal(command, false)
}

/// Tokenize an `env -S` / `--split-string` argument into words, treating
/// `\_` outside quotes as a whitespace delimiter in addition to standard
/// shell quoting and escaping.
fn tokenize_env_split_string(input: &str) -> Vec<String> {
    tokenize_words_internal(input, true)
}

fn tokenize_words_internal(input: &str, env_split_mode: bool) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut in_word = false;
    let mut state = QuoteState::None;
    let mut chars = input.chars().peekable();

    while let Some(c) = chars.next() {
        match state {
            QuoteState::None => match c {
                c if c.is_whitespace() => {
                    if in_word {
                        tokens.push(std::mem::take(&mut current));
                        in_word = false;
                    }
                }
                '\'' => {
                    in_word = true;
                    state = QuoteState::Single;
                }
                '"' => {
                    in_word = true;
                    state = QuoteState::Double;
                }
                '\\' => {
                    if let Some(next) = chars.next() {
                        if next == '\n' {
                            continue;
                        }
                        if env_split_mode && matches!(next, '_' | 't' | 'n' | 'r' | 'f' | 'v') {
                            if in_word {
                                tokens.push(std::mem::take(&mut current));
                                in_word = false;
                            }
                            continue;
                        }
                        in_word = true;
                        current.push(next);
                    }
                }
                _ => {
                    in_word = true;
                    current.push(c);
                }
            },
            QuoteState::Single => {
                if c == '\'' {
                    state = QuoteState::None;
                } else {
                    current.push(c);
                }
            }
            QuoteState::Double => match c {
                '"' => state = QuoteState::None,
                '\\' => {
                    if let Some(&next) = chars.peek() {
                        if next == '\n' {
                            chars.next();
                        } else if env_split_mode && next == '_' {
                            chars.next();
                            current.push(' ');
                        } else if matches!(next, '$' | '`' | '"' | '\\') {
                            current.push(chars.next().unwrap());
                        } else {
                            current.push('\\');
                        }
                    } else {
                        current.push('\\');
                    }
                }
                _ => current.push(c),
            },
        }
    }

    if in_word {
        tokens.push(current);
    }

    tokens
}

/// Extract the bare executable name from an unquoted command token.
fn program_basename(token: &str) -> &str {
    token.rsplit('/').next().unwrap_or(token)
}

/// Check whether an unquoted token is a `KEY=VALUE` environment assignment.
fn is_env_assignment(token: &str) -> bool {
    let Some((key, _)) = token.split_once('=') else {
        return false;
    };
    !key.is_empty() && !key.starts_with('-') && !key.contains('/')
}

/// Whether a short option on `wrapper` consumes a separate following argument token
/// when it appears at the end of a short-option cluster.
fn wrapper_short_opt_takes_arg(wrapper: &str, opt: char) -> bool {
    match wrapper {
        "sudo" => matches!(
            opt,
            'a' | 'C' | 'c' | 'D' | 'f' | 'g' | 'h' | 'p' | 'R' | 'r' | 'T' | 't' | 'U' | 'u'
        ),
        "env" => matches!(opt, 'u' | 'C' | 'P' | 'a'),
        "nice" => opt == 'n',
        "exec" => opt == 'a',
        "time" => matches!(opt, 'f' | 'o'),
        _ => false,
    }
}

/// Whether a long option on `wrapper` consumes a separate following argument token
/// when written without an inline `=value`.
fn wrapper_long_opt_takes_arg(wrapper: &str, opt: &str) -> bool {
    match wrapper {
        "sudo" => matches!(
            opt,
            "--auth-type"
                | "--chdir"
                | "--chroot"
                | "--close-from"
                | "--command-timeout"
                | "--group"
                | "--host"
                | "--login-class"
                | "--other-user"
                | "--prompt"
                | "--role"
                | "--type"
                | "--user"
        ),
        "env" => matches!(
            opt,
            "--unset"
                | "--chdir"
                | "--argv0"
                | "--default-signal"
                | "--ignore-signal"
                | "--block-signal"
        ),
        "nice" => opt == "--adjustment",
        "time" => matches!(opt, "--format" | "--output"),
        _ => false,
    }
}

/// Prepend split-string tokens onto the front of `queue`.
fn prepend_split_string_tokens(queue: &mut VecDeque<String>, raw: &str, expansions: &mut usize) {
    if *expansions >= MAX_SPLIT_STRING_EXPANSIONS {
        return;
    }
    *expansions += 1;
    for tok in tokenize_env_split_string(raw).into_iter().rev() {
        queue.push_front(tok);
    }
}

/// Advance past leading `KEY=VALUE` assignments and command wrappers (`sudo`, `env`, etc.).
fn strip_wrappers(tokens: Vec<String>) -> Vec<String> {
    let mut queue: VecDeque<String> = VecDeque::from(tokens);
    let mut split_expansions = 0;

    while let Some(front) = queue.front() {
        if is_env_assignment(front) {
            queue.pop_front();
            continue;
        }

        let base = program_basename(front).to_string();
        if !COMMAND_WRAPPERS.contains(&base.as_str()) {
            break;
        }
        queue.pop_front();

        while let Some(opt_ref) = queue.front() {
            if !opt_ref.starts_with('-') {
                break;
            }
            let opt = queue.pop_front().unwrap();
            if opt == "--" {
                break;
            }
            if opt == "-" {
                continue;
            }
            if opt.starts_with("--") {
                if base == "env" {
                    if opt == "--split-string" {
                        if let Some(val) = queue.pop_front() {
                            prepend_split_string_tokens(&mut queue, &val, &mut split_expansions);
                        }
                        continue;
                    }
                    if let Some(val) = opt.strip_prefix("--split-string=") {
                        prepend_split_string_tokens(&mut queue, val, &mut split_expansions);
                        continue;
                    }
                }
                if !opt.contains('=') && wrapper_long_opt_takes_arg(&base, &opt) {
                    queue.pop_front();
                }
                continue;
            }

            let mut chars = opt[1..].chars();
            while let Some(c) = chars.next() {
                if base == "env" && c == 'S' {
                    let rest = chars.as_str();
                    let val = if rest.is_empty() {
                        queue.pop_front()
                    } else {
                        Some(rest.strip_prefix('=').unwrap_or(rest).to_string())
                    };
                    if let Some(val) = val {
                        prepend_split_string_tokens(&mut queue, &val, &mut split_expansions);
                    }
                    break;
                }
                if wrapper_short_opt_takes_arg(&base, c) {
                    if chars.as_str().is_empty() {
                        queue.pop_front();
                    }
                    break;
                }
            }
        }
    }

    Vec::from(queue)
}

/// Check a single command (no shell operators) for catastrophic patterns.
fn check_single_command_catastrophic(command: &str) -> Option<&'static str> {
    let raw_tokens = tokenize_shell_words(command);
    let tokens = strip_wrappers(raw_tokens);
    if tokens.is_empty() {
        return None;
    }

    let program = program_basename(&tokens[0]);

    // Unconditionally blocked commands
    if BLOCKED_COMMANDS.contains(&program) {
        return Some(
            "This command is blocked for safety. It can cause irreversible system damage.",
        );
    }

    // dd targeting block devices
    if program == "dd"
        && tokens[1..].iter().any(|t| {
            t.strip_prefix("of=")
                .is_some_and(|path| normalize_shell_path(path).starts_with("/dev/"))
        })
    {
        return Some("Writing directly to block devices via dd is blocked.");
    }

    // chmod/chown -R on system paths
    if matches!(program, "chmod" | "chown")
        && tokens[1..].iter().any(|flag| {
            flag == "-R"
                || flag == "--recursive"
                || (flag.starts_with('-') && !flag.starts_with("--") && flag[1..].contains('R'))
        })
        && tokens[1..]
            .iter()
            .filter(|t| !t.starts_with('-'))
            .any(|t| is_protected_path(t))
    {
        return Some("Recursive permission changes on system paths are blocked.");
    }

    // rm targeting protected paths (rm -rf /tmp/foo is fine, rm -rf / is not)
    if program == "rm" {
        for token in &tokens[1..] {
            if token.starts_with('-') {
                continue;
            }
            if is_protected_path(token) {
                return Some(
                    "Removing system-critical paths is blocked. \
                     Workspace-relative and /tmp paths are allowed.",
                );
            }
        }
    }

    None
}

/// Expand home shortcuts and resolve `.` / `..` components on absolute paths
/// so `/tmp/..` and `/tmp/../etc` are recognized as `/` and `/etc`.
fn normalize_shell_path(unquoted: &str) -> String {
    let expanded = if unquoted == "~" || unquoted == "$HOME" || unquoted == "${HOME}" {
        "/home/user".to_string()
    } else if let Some(rest) = unquoted
        .strip_prefix("~/")
        .or_else(|| unquoted.strip_prefix("$HOME/"))
        .or_else(|| unquoted.strip_prefix("${HOME}/"))
    {
        format!("/home/user/{rest}")
    } else {
        unquoted.to_string()
    };

    if !expanded.starts_with('/') {
        return expanded;
    }

    let mut segments = Vec::new();
    for segment in expanded.split('/') {
        match segment {
            "" | "." => {}
            ".." => {
                segments.pop();
            }
            _ => segments.push(segment),
        }
    }

    if segments.is_empty() {
        "/".to_string()
    } else {
        format!("/{}", segments.join("/"))
    }
}

/// Check if an unquoted path is a protected system path.
///
/// Matches exact paths and paths directly under protected roots.
/// Does NOT block subdirectories of allowed workspace paths.
fn is_protected_path(path: &str) -> bool {
    let normalized = normalize_shell_path(path);
    let path = normalized.as_str();

    // Exact match against protected paths
    if PROTECTED_PATHS.contains(&path) {
        return true;
    }
    // One level under protected roots: /usr/local is protected, /tmp/build is not
    for root in PROTECTED_PATHS {
        if *root == "/" || *root == "/*" {
            continue;
        }
        if let Some(rest) = path.strip_prefix(root) {
            // /usr -> blocked, /usr/ -> blocked, /usr/local -> blocked
            // But /usrfoo -> not blocked (not a real subdirectory)
            if rest.is_empty() || rest.starts_with('/') {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catastrophic_rm_root() {
        assert!(check_catastrophic("rm -rf /").is_some());
        assert!(check_catastrophic("rm -rf /*").is_some());
        assert!(check_catastrophic("rm /").is_some());
    }

    #[test]
    fn catastrophic_rm_system_paths() {
        assert!(check_catastrophic("rm -rf /etc").is_some());
        assert!(check_catastrophic("rm -rf /usr").is_some());
        assert!(check_catastrophic("rm -rf /usr/local").is_some());
        assert!(check_catastrophic("rm -rf /home").is_some());
        assert!(check_catastrophic("rm -rf ~").is_some());
        assert!(check_catastrophic("rm -rf $HOME").is_some());
    }

    #[test]
    fn catastrophic_rm_safe_paths_allowed() {
        // Workspace-relative and /tmp paths are fine
        assert!(check_catastrophic("rm -rf ./build").is_none());
        assert!(check_catastrophic("rm -rf /tmp/build").is_none());
        assert!(check_catastrophic("rm -rf target/debug").is_none());
        assert!(check_catastrophic("rm foo.txt").is_none());
    }

    #[test]
    fn catastrophic_blocked_commands() {
        assert!(check_catastrophic("mkfs /dev/sda1").is_some());
        assert!(check_catastrophic("mkfs.ext4 /dev/sda1").is_some());
        assert!(check_catastrophic("shutdown -h now").is_some());
        assert!(check_catastrophic("reboot").is_some());
        assert!(check_catastrophic("halt").is_some());
        assert!(check_catastrophic("init 0").is_some());
    }

    #[test]
    fn catastrophic_dd_block_device() {
        assert!(check_catastrophic("dd if=/dev/zero of=/dev/sda").is_some());
        assert!(check_catastrophic("dd if=/dev/zero of=/dev/disk0").is_some());
        // dd to a file is fine
        assert!(check_catastrophic("dd if=/dev/zero of=./test.img bs=1M count=10").is_none());
    }

    #[test]
    fn catastrophic_chmod_system() {
        assert!(check_catastrophic("chmod -R 777 /").is_some());
        assert!(check_catastrophic("chown -R root:root /usr").is_some());
        // chmod on workspace files is fine
        assert!(check_catastrophic("chmod -R 755 ./dist").is_none());
        assert!(check_catastrophic("chmod 644 foo.txt").is_none());
    }

    #[test]
    fn catastrophic_fork_bomb() {
        assert!(check_catastrophic(":(){ :|:& };:").is_some());
    }

    #[test]
    fn catastrophic_chained_commands() {
        // Dangerous command hidden after benign one
        assert!(check_catastrophic("ls && rm -rf /").is_some());
        assert!(check_catastrophic("echo hello; shutdown -h now").is_some());
        assert!(check_catastrophic("cat /tmp/f || rm -rf /usr").is_some());
        assert!(check_catastrophic("ls | dd if=/dev/zero of=/dev/sda").is_some());
        assert!(check_catastrophic("ls & rm -rf /usr").is_some());
        assert!(check_catastrophic("ls\nrm -rf /").is_some());
        assert!(check_catastrophic("echo hello\r\nshutdown -h now").is_some());
        assert!(check_catastrophic("dd if=/dev/zero 2>&1 of=/dev/sda").is_some());
        // Both safe - should pass
        assert!(check_catastrophic("ls && echo done").is_none());
        assert!(check_catastrophic("cat foo; echo bar").is_none());
        assert!(check_catastrophic("ls\necho done").is_none());
        assert!(check_catastrophic("cargo build 2>&1").is_none());
    }

    #[test]
    fn catastrophic_normal_commands_pass() {
        assert!(check_catastrophic("git push origin main").is_none());
        assert!(check_catastrophic("cargo build --release").is_none());
        assert!(check_catastrophic("ls -la").is_none());
        assert!(check_catastrophic("cat /etc/hosts").is_none());
    }

    #[test]
    fn catastrophic_quoted_protected_paths_blocked() {
        assert!(check_catastrophic("rm -rf \"/\"").is_some());
        assert!(check_catastrophic("rm -rf '/'").is_some());
        assert!(check_catastrophic("rm -rf \"$HOME\"").is_some());
        assert!(check_catastrophic("rm -rf \"${HOME}\"").is_some());
        assert!(check_catastrophic("chmod -R 777 \"/etc\"").is_some());
        assert!(check_catastrophic("dd if=/dev/zero of=\"/dev/sda\"").is_some());
        assert!(check_catastrophic("dd if=/dev/zero \"of=/dev/sda\"").is_some());
        // Interior quotes and backslash escapes within a shell word
        assert!(check_catastrophic("rm -rf /tmp/\"../etc\"").is_some());
        assert!(check_catastrophic("rm -rf /tmp/../\"etc\"").is_some());
        assert!(check_catastrophic("rm -rf /tmp/'../etc'").is_some());
        assert!(check_catastrophic("rm -rf /tmp/\\.\\./etc").is_some());
        assert!(check_catastrophic("chmod -R 777 /tmp/\"../etc\"").is_some());
        assert!(check_catastrophic("dd if=/dev/zero of=/tmp/\"../dev/sda\"").is_some());
        assert!(check_catastrophic("dd if=/dev/zero of=/tmp/\\.\\./dev/sda").is_some());
        // Quoted paths containing spaces that traverse into protected roots
        assert!(check_catastrophic("rm -rf \"/tmp/a b/../../etc\"").is_some());
        assert!(check_catastrophic("rm -rf '/tmp/a b/../../etc'").is_some());
        assert!(check_catastrophic("rm -rf /tmp/a\\ b/../../etc").is_some());
        assert!(check_catastrophic("chmod -R 777 \"/tmp/a b/../../etc\"").is_some());
        assert!(check_catastrophic("dd if=/dev/zero of=\"/tmp/a b/../../dev/sda\"").is_some());
        // Quoted safe paths (including spaces) remain allowed
        assert!(check_catastrophic("rm -rf \"/tmp/build\"").is_none());
        assert!(check_catastrophic("rm -rf /tmp/\"build\"").is_none());
        assert!(check_catastrophic("rm -rf '/tmp/build'").is_none());
        assert!(check_catastrophic("rm -rf /tmp/\"a\"/../\"build\"").is_none());
        assert!(check_catastrophic("rm -rf \"/tmp/a b/build\"").is_none());
        assert!(check_catastrophic("rm -rf '/tmp/a b/build'").is_none());
        assert!(check_catastrophic("rm -rf \"/tmp/a b/../build\"").is_none());
        assert!(check_catastrophic("chmod -R 755 \"/tmp/a b/build\"").is_none());
    }

    #[test]
    fn catastrophic_parent_traversal_into_protected_paths_blocked() {
        assert!(check_catastrophic("rm -rf /tmp/..").is_some());
        assert!(check_catastrophic("rm -rf /tmp/../etc").is_some());
        assert!(check_catastrophic("rm -rf /tmp/../usr/local").is_some());
        assert!(check_catastrophic("chmod -R 777 /tmp/../etc").is_some());
        assert!(check_catastrophic("dd if=/dev/zero of=/tmp/../dev/sda").is_some());
        // Traversal that stays inside /tmp remains allowed
        assert!(check_catastrophic("rm -rf /tmp/a/../build").is_none());
    }

    #[test]
    fn catastrophic_path_qualified_and_wrapped_commands_blocked() {
        assert!(check_catastrophic("/bin/rm -rf /").is_some());
        assert!(check_catastrophic("/sbin/shutdown -h now").is_some());
        assert!(check_catastrophic("/usr/sbin/mkfs.ext4 /dev/sda1").is_some());
        assert!(check_catastrophic("sudo rm -rf /").is_some());
        assert!(check_catastrophic("sudo -n rm -rf /etc").is_some());
        assert!(check_catastrophic("sudo -u root rm -rf /").is_some());
        assert!(check_catastrophic("sudo -uroot rm -rf /").is_some());
        assert!(check_catastrophic("sudo -Eu root rm -rf /").is_some());
        assert!(check_catastrophic("sudo --user root rm -rf /").is_some());
        assert!(check_catastrophic("sudo --user=root rm -rf /").is_some());
        assert!(check_catastrophic("sudo -p \"hello world\" rm -rf /").is_some());
        assert!(check_catastrophic("sudo -p 'hello world' rm -rf /").is_some());
        assert!(check_catastrophic("sudo -p\"hello world\" rm -rf /").is_some());
        assert!(check_catastrophic("sudo --prompt \"hello world\" rm -rf /").is_some());
        assert!(check_catastrophic("sudo --prompt=\"hello world\" rm -rf /").is_some());
        assert!(check_catastrophic("sudo -p \"\" rm -rf /").is_some());
        assert!(check_catastrophic("env rm -rf /").is_some());
        assert!(check_catastrophic("env -u FOO rm -rf /").is_some());
        assert!(check_catastrophic("env --unset FOO rm -rf /").is_some());
        assert!(check_catastrophic("env -C /tmp rm -rf /").is_some());
        assert!(check_catastrophic("env -C \"/tmp/a b\" rm -rf /").is_some());
        // env -S / --split-string (short/long and attached-value spellings)
        assert!(check_catastrophic("env -S 'rm -rf /'").is_some());
        assert!(check_catastrophic("env -S \"rm -rf /\"").is_some());
        assert!(check_catastrophic("env -S'rm -rf /'").is_some());
        assert!(check_catastrophic("env -S\"rm -rf /\"").is_some());
        assert!(check_catastrophic("env -iS 'rm -rf /'").is_some());
        assert!(check_catastrophic("env -iS'rm -rf /'").is_some());
        assert!(check_catastrophic("env --split-string 'rm -rf /'").is_some());
        assert!(check_catastrophic("env --split-string \"rm -rf /\"").is_some());
        assert!(check_catastrophic("env --split-string='rm -rf /'").is_some());
        assert!(check_catastrophic("env --split-string=\"rm -rf /\"").is_some());
        assert!(check_catastrophic("env -S '-u FOO rm -rf /'").is_some());
        assert!(check_catastrophic("env -S 'rm -rf' /").is_some());
        assert!(check_catastrophic("nice -n 5 rm -rf /").is_some());
        assert!(check_catastrophic("nice -n -5 rm -rf /").is_some());
        assert!(check_catastrophic("nice --adjustment 5 rm -rf /").is_some());
        assert!(check_catastrophic("exec -a fake rm -rf /").is_some());
        assert!(check_catastrophic("exec -a \"fake name\" rm -rf /").is_some());
        assert!(check_catastrophic("time -o /tmp/t.log rm -rf /").is_some());
        assert!(check_catastrophic("time -f \"%E real\" rm -rf /").is_some());
        assert!(check_catastrophic("command -p rm -rf /").is_some());
        assert!(check_catastrophic("FOO=\"hello world\" rm -rf /").is_some());
        // Harmless wrapped commands remain allowed
        assert!(check_catastrophic("sudo -u root rm -rf /tmp/build").is_none());
        assert!(check_catastrophic("sudo -p \"hello world\" rm -rf \"/tmp/a b/build\"").is_none());
        assert!(check_catastrophic("env -u FOO cargo build --release").is_none());
        assert!(check_catastrophic("env -S 'cargo build --release'").is_none());
        assert!(check_catastrophic("env -S'cargo build --release'").is_none());
        assert!(check_catastrophic("env --split-string 'cargo build --release'").is_none());
        assert!(check_catastrophic("env --split-string='cargo build --release'").is_none());
        assert!(check_catastrophic("nice -n 5 ls -la /etc").is_none());
    }
}
