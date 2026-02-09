"""
Symdex-100 Auto-Reindex Module

Provides automatic re-indexation capabilities:
- File watcher mode (monitors file changes in real-time)
- Scheduled mode (periodic re-indexing)
- Hybrid mode (watch + debounce)
"""

import time
import logging
from pathlib import Path
from typing import Optional, Callable
from threading import Thread, Event
import hashlib

logger = logging.getLogger(__name__)


class AutoReindexer:
    """Automatic re-indexation with file watching and scheduling."""
    
    def __init__(self, root_dir: Path, client, interval_seconds: int = 300, debounce_seconds: int = 5):
        """
        Initialize auto-reindexer.
        
        Args:
            root_dir: Directory to monitor
            client: Symdex client instance (for reindexing)
            interval_seconds: Minimum seconds between re-indexes (default: 300 = 5 min)
            debounce_seconds: Seconds to wait after last change before re-indexing (default: 5)
        """
        self.root_dir = root_dir
        self.client = client
        self.interval_seconds = interval_seconds
        self.debounce_seconds = debounce_seconds
        self._stop_event = Event()
        self._thread: Optional[Thread] = None
        self._last_reindex_time = 0.0
        self._pending_reindex = False
        
    def start_watch(self):
        """Start file watcher in background thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("AutoReindexer already running")
            return
        
        self._stop_event.clear()
        self._thread = Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        logger.info(f"AutoReindexer started (interval: {self.interval_seconds}s, debounce: {self.debounce_seconds}s)")
    
    def stop(self):
        """Stop file watcher."""
        if self._thread:
            self._stop_event.set()
            self._thread.join(timeout=5)
            logger.info("AutoReindexer stopped")
    
    def _watch_loop(self):
        """Main watch loop (runs in background thread)."""
        try:
            # Use watchdog if available, else fallback to polling
            try:
                from watchdog.observers import Observer
                from watchdog.events import FileSystemEventHandler
                
                class ChangeHandler(FileSystemEventHandler):
                    def __init__(self, reindexer):
                        self.reindexer = reindexer
                    
                    def on_modified(self, event):
                        if not event.is_directory and event.src_path.endswith('.py'):
                            self.reindexer._pending_reindex = True
                    
                    def on_created(self, event):
                        if not event.is_directory and event.src_path.endswith('.py'):
                            self.reindexer._pending_reindex = True
                    
                    def on_deleted(self, event):
                        if not event.is_directory and event.src_path.endswith('.py'):
                            self.reindexer._pending_reindex = True
                
                observer = Observer()
                observer.schedule(ChangeHandler(self), str(self.root_dir), recursive=True)
                observer.start()
                logger.info("Using watchdog for file monitoring")
                
                while not self._stop_event.is_set():
                    if self._pending_reindex:
                        time.sleep(self.debounce_seconds)  # Wait for changes to settle
                        if self._pending_reindex:  # Still pending after debounce?
                            self._try_reindex()
                            self._pending_reindex = False
                    time.sleep(1)
                
                observer.stop()
                observer.join()
                
            except ImportError:
                # Fallback: polling mode
                logger.info("watchdog not available, using polling mode")
                last_snapshot = self._get_directory_snapshot()
                
                while not self._stop_event.is_set():
                    time.sleep(self.interval_seconds)
                    if self._stop_event.is_set():
                        break
                    
                    current_snapshot = self._get_directory_snapshot()
                    if current_snapshot != last_snapshot:
                        logger.info("Changes detected in monitored directory")
                        self._try_reindex()
                        last_snapshot = current_snapshot
        
        except Exception as e:
            logger.error(f"AutoReindexer error: {e}", exc_info=True)
    
    def _try_reindex(self):
        """Attempt re-indexing if enough time has passed."""
        now = time.time()
        elapsed = now - self._last_reindex_time
        
        if elapsed < self.interval_seconds:
            logger.debug(f"Skipping reindex (last: {elapsed:.1f}s ago, min interval: {self.interval_seconds}s)")
            return
        
        try:
            logger.info(f"Re-indexing {self.root_dir}...")
            result = self.client.index(str(self.root_dir), force=False, show_progress=False)
            self._last_reindex_time = now
            logger.info(f"Re-index complete: {result.files_processed} files processed, {result.functions_indexed} functions indexed")
        except Exception as e:
            logger.error(f"Re-index failed: {e}")
    
    def _get_directory_snapshot(self) -> str:
        """Get hash of all .py files for change detection (polling fallback)."""
        try:
            hasher = hashlib.sha256()
            for py_file in sorted(self.root_dir.rglob('*.py')):
                try:
                    if any(part.startswith('.') for part in py_file.parts):
                        continue  # Skip hidden dirs
                    hasher.update(str(py_file).encode())
                    hasher.update(str(py_file.stat().st_mtime).encode())
                except Exception:
                    pass
            return hasher.hexdigest()
        except Exception as e:
            logger.warning(f"Could not create directory snapshot: {e}")
            return ""


def start_auto_reindex(root_dir: str | Path, client, interval_seconds: int = 300, debounce_seconds: int = 5) -> AutoReindexer:
    """
    Start automatic re-indexation.
    
    Args:
        root_dir: Directory to monitor
        client: Symdex client instance
        interval_seconds: Minimum seconds between re-indexes (default: 300 = 5 min)
        debounce_seconds: Seconds to wait after last change before re-indexing (default: 5)
    
    Returns:
        AutoReindexer instance (call .stop() to terminate)
    
    Example:
        ```python
        from symdex import Symdex
        from symdex.core.autoreindex import start_auto_reindex
        
        client = Symdex()
        reindexer = start_auto_reindex("./project", client, interval_seconds=600)
        
        # ... do work ...
        
        reindexer.stop()
        ```
    """
    reindexer = AutoReindexer(Path(root_dir), client, interval_seconds, debounce_seconds)
    reindexer.start_watch()
    return reindexer
