//! Cross-thread bridge to the CUDA backend.

use std::sync::mpsc::{self, Sender};
use std::sync::{Mutex, OnceLock};
use std::thread::JoinHandle;

use super::CudaError;
use super::backend::{CudaBackend, CudaJob, CudaResult};

/// One job submitted to the GPU worker: a batch of CudaJobs and a private reply channel for the.
type Request = (Vec<CudaJob>, Sender<Result<Vec<CudaResult>, CudaError>>);

struct DispatcherState {
    tx: Sender<Request>,
    handle: Option<JoinHandle<()>>,
}

static DISPATCHER: OnceLock<Mutex<Option<DispatcherState>>> = OnceLock::new();

/// Bring up the GPU worker thread.
pub fn start() -> Result<(), CudaError> {
    let cell = DISPATCHER.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().unwrap();
    if guard.is_some() {
        return Ok(()); // already running
    }

    let (tx, rx) = mpsc::channel::<Request>();
    let handle = std::thread::Builder::new()
        .name("kira-cuda-dispatch".to_string())
        .spawn(move || {
            let mut backend = match CudaBackend::new() {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("[KIRA_GPU_DISPATCHER] init failed: {e}");
                    while let Ok((_, reply)) = rx.recv() {
                        let _ = reply.send(Err(CudaError::Driver(format!(
                            "GPU worker thread is dead: {e}"
                        ))));
                    }
                    return;
                }
            };
            while let Ok((jobs, reply)) = rx.recv() {
                let result = backend.run_batch(&jobs);
                let _ = reply.send(result);
            }
        })
        .map_err(|e| CudaError::Driver(format!("failed to spawn GPU thread: {e}")))?;

    *guard = Some(DispatcherState {
        tx,
        handle: Some(handle),
    });
    Ok(())
}

/// Submit a batch of CudaJobs for the GPU.
pub fn dispatch(jobs: Vec<CudaJob>) -> Option<Vec<CudaResult>> {
    if jobs.is_empty() {
        return Some(Vec::new());
    }
    let cell = DISPATCHER.get()?;
    let guard = cell.lock().ok()?;
    let state = guard.as_ref()?;
    let (reply_tx, reply_rx) = mpsc::channel();
    state.tx.send((jobs, reply_tx)).ok()?;
    drop(guard);
    match reply_rx.recv() {
        Ok(Ok(results)) => Some(results),
        Ok(Err(e)) => {
            eprintln!("[KIRA_GPU_DISPATCHER] batch failed: {e}");
            None
        }
        Err(_) => None, // worker disappeared
    }
}

/// `true` if a worker thread is currently accepting submissions.
pub fn is_active() -> bool {
    DISPATCHER
        .get()
        .and_then(|m| m.lock().ok())
        .map(|g| g.is_some())
        .unwrap_or(false)
}

/// Tear down the worker thread.
pub fn stop() {
    let Some(cell) = DISPATCHER.get() else {
        return;
    };
    let state_opt = {
        let mut guard = cell.lock().unwrap();
        guard.take() // takes ownership of DispatcherState, drops `tx`
    };
    if let Some(mut state) = state_opt {
        if let Some(handle) = state.handle.take() {
            let _ = handle.join();
        }
        // `state.tx` is now dropped, the worker exits.
        drop(state);
    }
}
