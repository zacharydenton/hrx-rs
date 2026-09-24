//! Bounded preparation of an already validated, readable mapping.
use std::{
    io,
    sync::atomic::{AtomicUsize, Ordering},
};

const CHUNK_BYTES: usize = 16 << 20;
const MAX_WORKERS: usize = 16;

pub(super) fn populate(begin: usize, length: usize) -> io::Result<bool> {
    #[cfg(target_os = "linux")]
    {
        let workers = std::thread::available_parallelism().map_or(1, usize::from);
        match chunks(begin, length, workers, |address, bytes| {
            // The caller holds the mapping alive until all scoped workers join.
            if unsafe { libc::madvise(address as *mut _, bytes, libc::MADV_POPULATE_READ) } == 0 {
                Ok(())
            } else {
                Err(io::Error::last_os_error())
            }
        }) {
            Ok(()) => return Ok(true),
            Err(error) if unsupported(&error) => {}
            Err(error) => return Err(error),
        }
    }
    if unsafe { libc::madvise(begin as *mut _, length, libc::MADV_WILLNEED) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(false)
}

fn unsupported(error: &io::Error) -> bool {
    matches!(
        error.raw_os_error(),
        Some(libc::EINVAL | libc::ENOSYS | libc::EOPNOTSUPP)
    )
}

fn chunks(
    begin: usize,
    length: usize,
    workers: usize,
    operation: impl Fn(usize, usize) -> io::Result<()> + Sync,
) -> io::Result<()> {
    let count = length.div_ceil(CHUNK_BYTES);
    let next = AtomicUsize::new(0);
    let run = || {
        let mut first_error = None;
        loop {
            let index = next.fetch_add(1, Ordering::Relaxed);
            if index >= count {
                break;
            }
            let offset = index * CHUNK_BYTES;
            // A real I/O failure must not become an unsupported-platform fallback.
            if let Err(error) = operation(begin + offset, CHUNK_BYTES.min(length - offset))
                && first_error.as_ref().is_none_or(unsupported)
            {
                first_error = Some(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    };
    std::thread::scope(|scope| {
        let mut threads = Vec::new();
        for _ in 1..workers.clamp(1, MAX_WORKERS).min(count) {
            threads.push(std::thread::Builder::new().spawn_scoped(scope, run)?);
        }
        let mut result = run();
        for thread in threads {
            let worker = thread
                .join()
                .map_err(|_| io::Error::other("page preparation worker panicked"))?;
            if let Err(error) = worker
                && result.as_ref().err().is_none_or(unsupported)
            {
                result = Err(error);
            }
        }
        result
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn ranges_are_disjoint_bounded_and_cover_the_tail() {
        let seen = Mutex::new(Vec::new());
        chunks(4096, CHUNK_BYTES * 35 + 4096, 100, |address, length| {
            seen.lock().unwrap().push((address, length));
            Ok(())
        })
        .unwrap();
        let mut seen = seen.into_inner().unwrap();
        seen.sort_unstable();
        assert_eq!(seen.len(), 36);
        let mut end = 4096;
        for (address, length) in seen {
            assert_eq!(address, end);
            assert!(length <= CHUNK_BYTES);
            end += length;
        }
        assert_eq!(end, CHUNK_BYTES * 35 + 8192);
    }

    #[test]
    fn real_failure_wins_over_unsupported_and_workers_join() {
        let finished = AtomicUsize::new(0);
        let error = chunks(0, CHUNK_BYTES * 20, 16, |address, _| {
            finished.fetch_add(1, Ordering::SeqCst);
            Err(io::Error::from_raw_os_error(if address == CHUNK_BYTES {
                libc::EIO
            } else {
                libc::EINVAL
            }))
        })
        .unwrap_err();
        assert_eq!(error.raw_os_error(), Some(libc::EIO));
        assert_eq!(finished.load(Ordering::SeqCst), 20);
    }
}
