"""One background campaign, watchable from the browser.

A speed sweep is a 1.5-2 hour blocking call and the web server answers requests
synchronously, so the sweep cannot BE a request. It runs on a thread and the UI
polls this module for what has happened so far.

Deliberately one job, not a queue. A campaign owns the GPU outright - that is the
whole point of the preflight guard - so a second concurrent run would not be
slower, it would be wrong, and both sets of numbers would be wrong in a way that
still looks like numbers. A second start is refused rather than queued.

Cancellation is cooperative and lands between configs. See speed_sweep().
"""
import collections, threading, time

# The sweep is chatty - one config prints a load line, a row line and whatever
# llama-server said. An unbounded transcript on a two-hour run is a slow leak, so
# the tail is bounded and the UI is told how much it missed.
LOG_LINES = 500


class Job(object):
    """State of the current (or last) campaign. All access under `_lock`."""

    def __init__(self):
        self._lock = threading.RLock()
        self._thread = None
        self.reset()

    def reset(self):
        self.status = "idle"          # idle | running | done | failed | cancelled
        self.started = None
        self.finished = None
        self.kind = None              # what is running, for the UI's heading
        self.total = 0                # configs in the CURRENT stage (see set_total)
        self.stage = None             # which stage that is, under a chained search
        self.planned = 0              # the whole campaign's estimate
        self.rows = []
        self.log = collections.deque(maxlen=LOG_LINES)
        self.dropped = 0              # log lines that fell off the front
        self.error = None
        self.result = None
        self._cancel = threading.Event()
        self._abort = threading.Event()   # second Stop: abandon the config in flight
        self.live_proc = None             # the llama-server up right now, if any
        self._skips = 0                   # Skip presses, for the run's stamping

    # -- writing, from the worker thread --------------------------------------
    def _append(self, line):
        with self._lock:
            if len(self.log) == self.log.maxlen:
                self.dropped += 1
            self.log.append(str(line))

    def _add_row(self, row):
        with self._lock:
            self.rows.append(row)

    def set_total(self, n, stage=None, planned=None):
        """How many configs the bar is measuring against, and what that MEANS.

        Under a chained search the two are different numbers and conflating them
        misreads badly. `n` counts the stage currently running, because a later
        stage's configs are built from a baseline that does not exist yet;
        `planned` is the campaign's own estimate, printed in the header. A bar
        reading "1 of 9" against a log saying "36 to run" is not wrong twice, it
        is one number answering each question - but only if it says which."""
        with self._lock:
            self.total = int(n or 0)
            if stage is not None:
                self.stage = stage
            if planned is not None:
                self.planned = int(planned or 0)

    # -- reading, from request threads ----------------------------------------
    @property
    def running(self):
        return self.status == "running"

    def cancelled(self):
        return self._cancel.is_set()

    def snapshot(self, since=0):
        """Everything the UI needs for one poll.

        `since` is a log offset, so a poll ships only what is new. It counts lines
        ever emitted, including dropped ones, so the offset stays meaningful after
        the buffer has wrapped."""
        with self._lock:
            emitted = self.dropped + len(self.log)
            since = max(0, min(int(since or 0), emitted))
            start = max(0, since - self.dropped)
            return {
                "status": self.status, "kind": self.kind,
                "started": self.started, "finished": self.finished,
                "elapsed_s": round((self.finished or time.time()) - self.started, 1)
                             if self.started else None,
                "total": self.total, "done": len(self.rows),
                "stage": self.stage, "planned": self.planned,
                "cancelling": self._cancel.is_set() and self.status == "running",
                "aborting": self._abort.is_set() and self.status == "running",
                "rows": list(self.rows),
                "log": list(self.log)[start:], "log_next": emitted,
                "log_dropped": self.dropped,
                "error": self.error, "result": self.result,
            }

    # -- control --------------------------------------------------------------
    def start(self, kind, fn):
        """Run fn(job) on a thread. Returns (ok, message).

        `fn` receives this job so it can call `_append`, `_add_row`, `set_total`
        and `cancelled`."""
        with self._lock:
            if self.running:
                return False, "a %s is already running" % (self.kind or "job")
            self.reset()
            self.status = "running"
            self.kind = kind
            self.started = time.time()

        def run():
            try:
                res = fn(self)
                with self._lock:
                    self.result = res
                    # A run that was asked to stop reports `cancelled` even though
                    # it returned normally - it did, that is what cooperative
                    # cancellation looks like - so the UI does not claim a partial
                    # campaign finished.
                    self.status = "cancelled" if self._cancel.is_set() else "done"
            except Exception as e:
                with self._lock:
                    self.error = "%s: %s" % (type(e).__name__, e)
                    self.status = "failed"
                self._append("FAILED: %s" % self.error)
            finally:
                with self._lock:
                    self.finished = time.time()

        # daemon: Ctrl+C on the server should not be held hostage by a sweep with
        # an hour left to run. The rows already written are on disk and keyed, so
        # nothing is lost by dying here.
        self._thread = threading.Thread(target=run, name="vramplanner-%s" % kind,
                                        daemon=True)
        self._thread.start()
        return True, "started"

    def set_live_proc(self, proc):
        """The llama-server this campaign has up right now, or None between configs.

        Held so a HARD stop has something to act on. Cancellation is checked
        between configs by the worker thread, but the worker spends nearly all of
        its time inside one blocking HTTP call to that server - a 120k-token
        prefill is minutes - and a flag it will not look at until the call
        returns is not a stop button, however correct it is."""
        with self._lock:
            self.live_proc = proc

    def cancel(self):
        """First press finishes the config in flight. Second abandons it.

        The soft stop is the right default and stays the default: a row is only
        worth having if it was measured start to finish, so ending between
        configs is what keeps the store free of half-measurements. But it can be
        several minutes away, and someone pressing Stop twice wants the card
        back, not a lecture about data hygiene.

        A hard stop kills the server, which makes the in-flight request fail at
        once rather than at its timeout. The row that was being measured is then
        discarded rather than written - see run_group() - so the campaign resumes
        from the last COMPLETE row and re-measures the abandoned one."""
        with self._lock:
            if not self.running:
                return False, "nothing running"
            first = not self._cancel.is_set()
            self._cancel.set()
            proc = None
            if not first:
                self._abort.set()
                proc = self.live_proc
        if first:
            self._append("stop requested - finishing the config in flight first, so the "
                         "row it is measuring is complete rather than half-written. "
                         "Press Stop again to abandon it instead.")
            return True, "stopping"
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except Exception as e:
                self._append("could not kill the server: %s" % e)
        self._append("stop forced - the config in flight was abandoned and its row "
                     "discarded, so it will be re-measured rather than half-recorded")
        return True, "aborting"

    def aborting(self):
        return self._abort.is_set()

    def skipped(self):
        """How many skips have been requested, for the run's stamping.

        The sweep snapshots this before each config and compares after: a press
        is bound to the run it was made during, and a press between runs is
        consumed by the comparison without touching anything - nothing was in
        flight to skip."""
        with self._lock:
            return self._skips

    def skip(self):
        """Abandon the config in flight and move on, without ending the campaign.

        Stop is the escape from the campaign; Skip is the escape from ONE run -
        the hung server, the config that is clearly not going to work. The
        server is killed so the run fails promptly instead of sitting out its
        generation timeout, and bench_one() then stamps the row `skipped`: it is
        recorded and keyed, so this campaign - and a resumed one - never
        re-measures it. The rest of the queue is untouched."""
        with self._lock:
            if not self.running:
                return False, "nothing running"
            self._skips += 1
            proc = self.live_proc
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except Exception as e:
                self._append("could not kill the server: %s" % e)
        self._append("skip requested - abandoning the config in flight. Its row is "
                     "recorded as skipped and will not be re-measured; the campaign "
                     "continues with the next config.")
        return True, "skipping"


# The singleton. Module-level because the HTTP handler is instantiated per
# request and has nowhere else to keep it.
JOB = Job()
