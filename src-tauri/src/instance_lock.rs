//! One process owns the application's shared writable storage. The lock file
//! must never be unlinked: another inode at the same name would bypass it.
use std::fs::{self, File, OpenOptions, TryLockError};
use std::io;
use std::path::{Path, PathBuf};

pub(crate) enum InstanceClaim {
    Primary(InstanceLock),
    Secondary,
}

pub(crate) struct InstanceLock {
    _owner: File,
    activation: PathBuf,
}

impl InstanceLock {
    pub(crate) fn claim(data_dir: &Path) -> io::Result<InstanceClaim> {
        fs::create_dir_all(data_dir)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(data_dir, fs::Permissions::from_mode(0o700))?;
        }
        let mut options = OpenOptions::new();
        options.read(true).write(true).create(true).truncate(false);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let owner = options.open(data_dir.join(".instance.lock"))?;
        let activation = data_dir.join(".instance.activate");
        match owner.try_lock() {
            Ok(()) => {
                let instance = Self {
                    _owner: owner,
                    activation,
                };
                instance.take_activation()?; // Clear a previous process's signal.
                Ok(InstanceClaim::Primary(instance))
            }
            Err(TryLockError::WouldBlock) => {
                // Signals coalesce: all duplicate launches only request focus.
                let mut signal = OpenOptions::new();
                signal.write(true).create_new(true);
                #[cfg(unix)]
                {
                    use std::os::unix::fs::OpenOptionsExt;
                    signal.mode(0o600);
                }
                match signal.open(activation) {
                    Ok(_) => Ok(InstanceClaim::Secondary),
                    Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                        Ok(InstanceClaim::Secondary)
                    }
                    Err(error) => Err(error),
                }
            }
            Err(TryLockError::Error(error)) => Err(error),
        }
    }

    pub(crate) fn take_activation(&self) -> io::Result<bool> {
        match fs::remove_file(&self.activation) {
            Ok(()) => Ok(true),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(error) => Err(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    struct Storage(PathBuf);
    impl Storage {
        fn new() -> Self {
            static SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            Self(std::env::temp_dir().join(format!(
                "opentake-instance-test-{}-{}",
                std::process::id(),
                SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            )))
        }
    }
    impl Drop for Storage {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }

    fn primary(path: &Path) -> InstanceLock {
        match InstanceLock::claim(path).unwrap() {
            InstanceClaim::Primary(instance) => instance,
            InstanceClaim::Secondary => panic!("expected storage ownership"),
        }
    }

    /// Re-claims storage whose previous owner was just dropped. `flock` locks
    /// belong to the open file description, and a child that another test
    /// thread is forking holds a copy of every descriptor until its `exec`
    /// closes the close-on-exec ones, so the released lock can stay held for
    /// a moment after `drop`. Only that transient `Secondary` is retried; a
    /// lock that is never released still fails after the deadline.
    fn primary_after_release(path: &Path) -> InstanceLock {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            match InstanceLock::claim(path).unwrap() {
                InstanceClaim::Primary(instance) => return instance,
                InstanceClaim::Secondary if std::time::Instant::now() < deadline => {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                InstanceClaim::Secondary => panic!("released storage was never reclaimed"),
            }
        }
    }

    #[test]
    fn independent_handles_share_one_owner_and_activation_coalesces() {
        let storage = Storage::new();
        let owner = primary(&storage.0);
        assert!(!owner.take_activation().unwrap());
        for _ in 0..3 {
            assert!(matches!(
                InstanceLock::claim(&storage.0).unwrap(),
                InstanceClaim::Secondary
            ));
        }
        assert!(owner.take_activation().unwrap());
        assert!(!owner.take_activation().unwrap());
        drop(owner);
        let next = primary_after_release(&storage.0);
        // A retried claim leaves a focus request, which the new owner clears.
        assert!(!next.take_activation().unwrap());
    }

    #[test]
    fn duplicate_process_cannot_acquire_shared_storage_and_requests_focus() {
        let storage = Storage::new();
        let owner = primary(&storage.0);
        let test_module = module_path!().split_once("::").unwrap().1;
        let helper = format!("{test_module}::secondary_process");
        let status = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", &helper, "--ignored", "--nocapture"])
            .env("OPENTAKE_INSTANCE_TEST_ROOT", &storage.0)
            .status()
            .unwrap();
        assert!(status.success());
        assert!(owner.take_activation().unwrap());
    }

    #[test]
    #[ignore = "subprocess fixture"]
    fn secondary_process() {
        let path = std::env::var_os("OPENTAKE_INSTANCE_TEST_ROOT").unwrap();
        assert!(matches!(
            InstanceLock::claim(Path::new(&path)).unwrap(),
            InstanceClaim::Secondary
        ));
    }

    #[test]
    fn process_exit_releases_lock_without_unlinking_it() {
        let storage = Storage::new();
        let test_module = module_path!().split_once("::").unwrap().1;
        let helper = format!("{test_module}::primary_process_exit");
        let status = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", &helper, "--ignored", "--nocapture"])
            .env("OPENTAKE_INSTANCE_TEST_ROOT", &storage.0)
            .status()
            .unwrap();
        assert!(status.success());
        assert!(storage.0.join(".instance.lock").is_file());
        let owner = primary(&storage.0);
        assert!(!owner.take_activation().unwrap());
    }

    #[test]
    #[ignore = "subprocess fixture"]
    fn primary_process_exit() {
        let path = std::env::var_os("OPENTAKE_INSTANCE_TEST_ROOT").unwrap();
        let _owner = primary(Path::new(&path));
        std::process::exit(0);
    }

    #[cfg(unix)]
    #[test]
    fn storage_and_control_files_are_private() {
        use std::os::unix::fs::PermissionsExt;
        let storage = Storage::new();
        let owner = primary(&storage.0);
        assert!(matches!(
            InstanceLock::claim(&storage.0).unwrap(),
            InstanceClaim::Secondary
        ));
        assert_eq!(
            fs::metadata(&storage.0).unwrap().permissions().mode() & 0o777,
            0o700
        );
        for name in [".instance.lock", ".instance.activate"] {
            assert_eq!(
                fs::metadata(storage.0.join(name))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
        }
        assert!(owner.take_activation().unwrap());
    }
}
