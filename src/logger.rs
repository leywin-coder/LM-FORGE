use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Info,
    Warn,
    Error,
    Debug,
}

impl LogLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Info => "INFO",
            LogLevel::Warn => "WARN",
            LogLevel::Error => "ERROR",
            LogLevel::Debug => "DEBUG",
        }
    }
}

#[derive(Debug, Clone)]
pub struct LogEntry {
    pub timestamp: String,
    pub level: LogLevel,
    pub target: String,
    pub message: String,
}

pub struct Logger {
    entries: Mutex<VecDeque<LogEntry>>,
    max_capacity: usize,
}

impl Logger {
    pub fn new(max_capacity: usize) -> Self {
        Self {
            entries: Mutex::new(VecDeque::with_capacity(max_capacity)),
            max_capacity,
        }
    }

    pub fn global() -> &'static Logger {
        static INSTANCE: OnceLock<Logger> = OnceLock::new();
        INSTANCE.get_or_init(|| Logger::new(500))
    }

    pub fn log(&self, level: LogLevel, target: &str, message: &str) {
        let ts = Self::current_time_str();
        println!("[{}] [{}] [{}] {}", ts, level.as_str(), target, message);

        let mut queue = self.entries.lock().unwrap();
        if queue.len() >= self.max_capacity {
            queue.pop_front();
        }
        queue.push_back(LogEntry {
            timestamp: ts,
            level,
            target: target.to_string(),
            message: message.to_string(),
        });
    }

    pub fn get_entries(&self) -> Vec<LogEntry> {
        let queue = self.entries.lock().unwrap();
        queue.iter().cloned().collect()
    }

    pub fn clear(&self) {
        let mut queue = self.entries.lock().unwrap();
        queue.clear();
    }

    fn current_time_str() -> String {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
        let total_secs = now.as_secs();
        let millis = now.subsec_millis();
        let secs = total_secs % 60;
        let mins = (total_secs / 60) % 60;
        let hours = (total_secs / 3600) % 24;
        format!("{:02}:{:02}:{:02}.{:03}", hours, mins, secs, millis)
    }
}

#[macro_export]
macro_rules! log_info {
    ($target:expr, $($arg:tt)+) => {
        $crate::logger::Logger::global().log($crate::logger::LogLevel::Info, $target, &format!($($arg)+))
    };
}

#[macro_export]
macro_rules! log_warn {
    ($target:expr, $($arg:tt)+) => {
        $crate::logger::Logger::global().log($crate::logger::LogLevel::Warn, $target, &format!($($arg)+))
    };
}

#[macro_export]
macro_rules! log_error {
    ($target:expr, $($arg:tt)+) => {
        $crate::logger::Logger::global().log($crate::logger::LogLevel::Error, $target, &format!($($arg)+))
    };
}

#[macro_export]
macro_rules! log_debug {
    ($target:expr, $($arg:tt)+) => {
        $crate::logger::Logger::global().log($crate::logger::LogLevel::Debug, $target, &format!($($arg)+))
    };
}
