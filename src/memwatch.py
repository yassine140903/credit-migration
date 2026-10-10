"""A background memory watch for long runs: peak RSS, minimum available system memory, and a hard floor.

    watch = MemWatch(on_abort=callback)     # callback(message) writes whatever was measured
    watch.start(); ...; watch.stop()
    watch.peak_rss(), watch.min_avail

If the system's available memory falls below abort_mb the callback is called and the process ends at once (a long
C call such as a LightGBM fit cannot be interrupted from Python any other way).
"""

import os
import threading

import psutil

MB = 2 ** 20


class MemWatch(threading.Thread):
    def __init__(self, on_abort, abort_mb=500, interval=0.5):
        super().__init__(daemon=True)
        self.proc = psutil.Process()
        self.on_abort, self.abort_mb, self.interval = on_abort, abort_mb, interval
        self.stop_flag = threading.Event()
        self.phase = "start"
        self.min_avail = psutil.virtual_memory().available
        self._peak = self.proc.memory_info().rss
        self.n = 0

    def run(self):
        while not self.stop_flag.is_set():
            a = psutil.virtual_memory().available
            self.min_avail = min(self.min_avail, a)
            self._peak = max(self._peak, self.proc.memory_info().rss)
            self.n += 1
            if a < self.abort_mb * MB:
                self.on_abort("available system memory fell to %.0f MB, below the %d MB limit, during '%s'"
                              % (a / MB, self.abort_mb, self.phase))
                os._exit(2)
            self.stop_flag.wait(self.interval)

    def stop(self):
        self.stop_flag.set()
        self.join()

    def peak_rss(self):
        """Largest resident set seen: the OS's own process-lifetime peak where it has one, else the sampled maximum."""
        return max(self._peak, getattr(self.proc.memory_info(), "peak_wset", 0))
