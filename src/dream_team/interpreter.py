"""
Python interpreter for executing code snippets and capturing their output.
Supports:
- captures stdout and stderr
- captures exceptions and stack traces
- limits execution time
"""

import logging
import os
import queue
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from multiprocessing import Process, Queue
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any

# Try to import humanize, fallback if not present
try:
    import humanize
except ImportError:
    humanize = None

# Try to import dataclasses_json, fallback if not present
try:
    from dataclasses_json import DataClassJsonMixin
except ImportError:
    # Minimal fallback if dataclasses_json is missing
    class DataClassJsonMixin:
        def to_json(self):
            import json
            return json.dumps(self.__dict__)
        
        @classmethod
        def from_json(cls, json_str):
            import json
            data = json.loads(json_str)
            return cls(**data)

logger = logging.getLogger("dream_team")


@dataclass
class ExecutionResult(DataClassJsonMixin):
    """
    Result of executing a code snippet in the interpreter.
    Contains the output, execution time, and exception information.
    """

    term_out: List[str]
    exec_time: float
    exc_type: Optional[str]
    exc_info: Optional[Dict] = None
    exc_stack: Optional[List[Tuple]] = None


def exception_summary(e, working_dir, exec_file_name, format_tb_ipython):
    """Generates a string that summarizes an exception and its stack trace."""
    if format_tb_ipython:
        try:
            import IPython.core.ultratb
            tb = IPython.core.ultratb.VerboseTB(tb_offset=1, color_scheme="NoColor")
            tb_str = str(tb.text(*sys.exc_info()))
        except ImportError:
            tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
            tb_str = "".join(tb_lines)
    else:
        tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
        # skip parts of stack trace in workflow code
        tb_str = "".join(
            [l for l in tb_lines if "dream_team/" not in l and "importlib" not in l]
        )

    # replace whole path to file with just filename (to remove agent workspace dir)
    if working_dir and exec_file_name:
        tb_str = tb_str.replace(str(Path(working_dir) / exec_file_name), exec_file_name)

    exc_info = {}
    if hasattr(e, "args"):
        exc_info["args"] = [str(i) for i in e.args]
    for att in ["name", "msg", "obj"]:
        if hasattr(e, att):
            exc_info[att] = str(getattr(e, att))

    tb = traceback.extract_tb(e.__traceback__)
    exc_stack = [(t.filename, t.lineno, t.name, t.line) for t in tb]

    return tb_str, e.__class__.__name__, exc_info, exc_stack


class RedirectQueue:
    def __init__(self, queue):
        self.queue = queue

    def write(self, msg):
        self.queue.put(msg)

    def flush(self):
        pass


class Interpreter:
    def __init__(
        self,
        working_dir: Path | str,
        timeout: int = 3600,
        format_tb_ipython: bool = False,
        agent_file_name: str = "runfile.py",
        env_vars: Dict[str, str] = {},
        initial_globals: Dict[str, Any] = None,
    ):
        """
        Simulates a standalone Python REPL with an execution time limit.

        Args:
            working_dir (Path | str): working directory of the agent
            timeout (int, optional): Timeout for each code execution step. Defaults to 3600.
            format_tb_ipython (bool, optional): Whether to use IPython or default python REPL formatting for exceptions. Defaults to False.
            agent_file_name (str, optional): The name for the agent's code file. Defaults to "runfile.py".
            env_vars (dict[str, str], optional): Environment variables to set in the child process. Defaults to {}.
            initial_globals (dict[str, Any], optional): Initial global variables for the execution scope. Defaults to None.
        """
        self.working_dir = Path(working_dir).resolve()
        if not self.working_dir.exists():
            self.working_dir.mkdir(parents=True, exist_ok=True)
            
        self.timeout = timeout
        self.format_tb_ipython = format_tb_ipython
        self.agent_file_name = agent_file_name
        self.process: Optional[Process] = None
        self.env_vars = env_vars
        self.initial_globals = initial_globals or {}

    def child_proc_setup(self, result_outq: Queue) -> None:
        # disable all warnings (before importing anything)
        import warnings
        warnings.filterwarnings("ignore")
        
        try:
            import shutup
            shutup.mute_warnings()
        except ImportError:
            pass

        for key, value in self.env_vars.items():
            os.environ[key] = value

        os.chdir(str(self.working_dir))

        # Add working dir to path so imports work
        sys.path.append(str(self.working_dir))

        # capture stdout and stderr
        sys.stdout = sys.stderr = RedirectQueue(result_outq)

    def _run_session(
        self, code_inq: Queue, result_outq: Queue, event_outq: Queue
    ) -> None:
        self.child_proc_setup(result_outq)

        global_scope = self.initial_globals.copy()
        print(f"DEBUG: Interpreter initialized with globals: {list(global_scope.keys())}")
        while True:
            code = code_inq.get()
            os.chdir(str(self.working_dir))
            with open(self.agent_file_name, "w") as f:
                f.write(code)

            event_outq.put(("state:ready",))
            try:
                exec(compile(code, self.agent_file_name, "exec"), global_scope)
            except BaseException as e:
                tb_str, e_cls_name, exc_info, exc_stack = exception_summary(
                    e,
                    self.working_dir,
                    self.agent_file_name,
                    self.format_tb_ipython,
                )
                result_outq.put(tb_str)
                if e_cls_name == "KeyboardInterrupt":
                    e_cls_name = "TimeoutError"

                event_outq.put(("state:finished", e_cls_name, exc_info, exc_stack))
            else:
                event_outq.put(("state:finished", None, None, None))

            # put EOF marker to indicate that we're done
            result_outq.put("<|EOF|>")

    def create_process(self) -> None:
        # we use three queues to communicate with the child process:
        # - code_inq: send code to child to execute
        # - result_outq: receive stdout/stderr from child
        # - event_outq: receive events from child (e.g. state:ready, state:finished)
        self.code_inq, self.result_outq, self.event_outq = Queue(), Queue(), Queue()
        self.process = Process(
            target=self._run_session,
            args=(self.code_inq, self.result_outq, self.event_outq),
        )
        self.process.start()

    def _drain_queues(self):
        """Quickly drain all in-flight messages to prevent blocking."""
        while not self.result_outq.empty():
            try:
                self.result_outq.get_nowait()
            except Exception:
                break

        while not self.event_outq.empty():
            try:
                self.event_outq.get_nowait()
            except Exception:
                break

        while not self.code_inq.empty():
            try:
                self.code_inq.get_nowait()
            except Exception:
                break

    def cleanup_session(self):
        if self.process is None:
            return
        # give the child process a chance to terminate gracefully
        self.process.terminate()
        self._drain_queues()
        self.process.join(timeout=2)
        # kill the child process if it's still alive
        if self.process.exitcode is None:
            logger.warning("Child process failed to terminate gracefully, killing it..")
            self.process.kill()
            self._drain_queues()
            self.process.join(timeout=2)
        # don't wait for gc, clean up immediately
        self.process.close()
        self.process = None

    def run(self, code: str, reset_session=True) -> ExecutionResult:
        """
        Execute the provided Python command in a separate process and return its output.

        Parameters:
            code (str): Python code to execute.
            reset_session (bool, optional): Whether to reset the interpreter session before executing the code. Defaults to True.

        Returns:
            ExecutionResult: Object containing the output and metadata of the code execution.
        """

        logger.debug(f"REPL is executing code (reset_session={reset_session})")

        if reset_session:
            if self.process is not None:
                # terminate and clean up previous process
                self.cleanup_session()
            self.create_process()
        else:
            # reset_session needs to be True on first exec
            if self.process is None:
                self.create_process()
            assert self.process is not None

        assert self.process.is_alive()

        self.code_inq.put(code)

        # wait for child to actually start execution
        try:
            state = self.event_outq.get(timeout=10)
        except queue.Empty:
            msg = "REPL child process failed to start execution"
            logger.critical(msg)
            while not self.result_outq.empty():
                logger.error(f"REPL output queue dump: {self.result_outq.get()}")
            raise RuntimeError(msg) from None
        assert state[0] == "state:ready", state
        start_time = time.time()

        child_in_overtime = False

        while True:
            try:
                # check if the child is done
                state = self.event_outq.get(timeout=1)  # wait for state:finished
                assert state[0] == "state:finished", state
                exec_time = time.time() - start_time
                break
            except queue.Empty:
                # we haven't heard back from the child -> check if it's still alive
                if not child_in_overtime and not self.process.is_alive():
                    msg = "REPL child process died unexpectedly"
                    logger.critical(msg)
                    while not self.result_outq.empty():
                        logger.error(
                            f"REPL output queue dump: {self.result_outq.get()}"
                        )
                    raise RuntimeError(msg) from None

                # child is alive and still executing -> check if we should sigint..
                if self.timeout is None:
                    continue
                running_time = time.time() - start_time
                if running_time > self.timeout:
                    # send interrupt to child
                    os.kill(self.process.pid, signal.SIGINT)  # type: ignore
                    child_in_overtime = True
                    # terminate if we're overtime by more than a minute
                    if running_time > self.timeout + 60:
                        logger.warning("Child failed to terminate, killing it..")
                        self.cleanup_session()

                        state = (None, "TimeoutError", {}, [])
                        exec_time = self.timeout
                        break

        output: List[str] = []
        # read all stdout/stderr from child up to the EOF marker
        while not self.result_outq.empty() or not output or output[-1] != "<|EOF|>":
            output.append(self.result_outq.get())
        output.pop()  # remove the EOF marker

        e_cls_name, exc_info, exc_stack = state[1:]

        if e_cls_name == "TimeoutError":
            output.append(
                f"TimeoutError: Execution exceeded the time limit of {self.timeout}s"
            )
        else:
            if humanize:
                time_str = humanize.naturaldelta(exec_time)
            else:
                time_str = f"{exec_time:.2f}s"
            output.append(
                f"Execution time: {time_str} (time limit is {self.timeout}s)."
            )
        return ExecutionResult(output, exec_time, e_cls_name, exc_info, exc_stack)
