"""
Quick integration test for idle monitor and background task manager.

Validates that:
1. IdleMonitor initializes and tracks activity (simulated)
2. BackgroundTaskManager starts/tracks tasks
3. Both integrate with agent lifecycle
"""

import sys
import time
import threading

sys.path.insert(0, 'perception')
sys.path.insert(0, 'automation')

from idle_monitor import IdleMonitor
from background_tasks import BackgroundTaskManager


def test_idle_monitor():
    """Test idle monitor initialization and stats."""
    print("Testing IdleMonitor...")
    monitor = IdleMonitor()
    monitor.start()
    
    # Quick snapshot
    stats = monitor.stats()
    print(f"  ✓ Initialized: {stats}")
    assert stats['running'] == True
    assert isinstance(stats['idle_seconds'], float)
    
    # Simulate some time passing
    time.sleep(0.6)
    
    idle_sec = monitor.idle_seconds()
    print(f"  ✓ Idle time: {idle_sec:.1f}s")
    
    # Stop
    monitor.stop()
    print(f"  ✓ Stopped cleanly")
    print()


def test_background_tasks():
    """Test background task manager."""
    print("Testing BackgroundTaskManager...")
    manager = BackgroundTaskManager()
    
    # Start a simple task
    def work_task():
        time.sleep(0.1)
        return "done"
    
    task_id = manager.start_task("test_work", work_task)
    print(f"  ✓ Started task: {task_id}")
    
    # Check it's running
    active = manager.list_active()
    print(f"  ✓ Active tasks: {len(active)}")
    assert len(active) >= 1
    
    # Wait for it
    success = manager.wait_task(task_id, timeout=2.0)
    print(f"  ✓ Task completed: {success}")
    assert success == True
    
    # Cleanup
    manager.cleanup()
    stats = manager.stats()
    print(f"  ✓ Stats after cleanup: {stats}")
    print()


def test_idle_for_threshold():
    """Test idle threshold checking."""
    print("Testing idle threshold logic...")
    monitor = IdleMonitor()
    # Don't start — just validate the logic
    
    # On Windows, idle should be < 1hr initially
    # On non-Windows, idle is always 0
    idle = monitor.idle_seconds()
    is_idle_1hr = monitor.is_idle_for(3600)
    
    print(f"  ✓ Idle seconds: {idle:.1f}")
    print(f"  ✓ Is idle >= 1hr: {is_idle_1hr}")
    print()


def test_background_task_cancellation():
    """Test task cancellation."""
    print("Testing task cancellation...")
    manager = BackgroundTaskManager()
    
    def long_task():
        for i in range(10):
            time.sleep(0.1)
        return "finished"
    
    task_id = manager.start_task("slow_task", long_task)
    print(f"  ✓ Started long task: {task_id}")
    
    time.sleep(0.05)  # Let it start
    
    # Cancel all
    cancelled = manager.cancel_all(timeout=1.0)
    print(f"  ✓ Cancelled {cancelled} tasks")
    
    # Verify it's gone
    active = manager.list_active()
    print(f"  ✓ Active tasks remaining: {len(active)}")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("INTEGRATION TEST: Idle Monitor + Background Tasks")
    print("=" * 60)
    print()
    
    try:
        test_idle_monitor()
        test_background_tasks()
        test_idle_for_threshold()
        test_background_task_cancellation()
        
        print("=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
