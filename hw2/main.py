# -*- coding: utf-8 -*-
# Download dataset
import os
import subprocess
import torch
import re

import sys
import psutil


# 檢查目錄是否存在
if not os.path.exists('./ML2025Spring-hw2-public'):
    print("Downloading dataset...")
    # 下載數據集
    subprocess.run(['gdown', '--id', '1Ah5uV6cu3Bnz6WfkUuxEZCLqj5k1lbpd'], check=True)
    # subprocess.run(['gdown', '--id', '1XtF9-hGw2tKe4WvUMW5YE6lj6p1QcWIc'], check=True)
    # subprocess.run(['gdown', '--id', '1diswE_9XoT-uII23ucRppau1ErEQkY2y'], check=True)
    # subprocess.run(['gdown', '--id', '1BAVMzLZqEgtG8rwog7ttC7xKPw5QTngn'], check=True)
    # subprocess.run(['gdown', '--id', '1PAI4_3kRWwIPQMscMdGt9HLqZZy1vWSD'], check=True)
    
    # 解壓縮數據集
    subprocess.run(['unzip', './ML2025Spring-hw2-public.zip'], check=True)
else:
    print("The dataset has been downloaded.")

import os
import subprocess

# 檢查檔案是否存在
# model_filename = 'Meta-Llama-3.1-8B-Instruct-Q8_0.gguf'
# model_link = 'https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf'

model_filename = 'qwen2.5-coder-14b-instruct-q5_0.gguf'
model_link = 'https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct-GGUF/resolve/main/qwen2.5-coder-14b-instruct-q5_0.gguf'
if not os.path.exists(model_filename):
    print(f"Downloading model file: {model_filename}...")
    # 下載模型檔案
    subprocess.run(['wget', model_link], check=True)
else:
    print(f"The model file {model_filename} has been downloaded.")


from llama_cpp import Llama

# Load the model onto GPU
myModel = Llama(
    model_filename,
    verbose=False,
    n_gpu_layers=-1,
    n_batch=1024,
    n_ctx=8192,    # This argument is how many tokens the model can take. The longer the better, but it will consume more memory.
)

OVERFITTING_THRESHOLD = 0.5

def generate_response(_model: Llama, _messages: str) -> str:
    '''
    This function will inference the model with given messages.
    '''
    _output = _model.create_chat_completion(
        _messages,
        stop=["<|eot_id|>", "<|end_of_text|>"],
        max_tokens=4096,    # This argument is how many tokens the model can generate.
        temperature=0,      # This argument is the randomness of the model. 0 means no randomness. We suggest setting the temperature value to 0 for reproducibility.
    )["choices"][0]["message"]["content"]
    return _output

"""## Functions

### Utils
"""

# Define a function to save the best solution and other good solutions to files.
def save_run(cfg, journal):
    # Retrieve and save the best found solution.
    best_node = journal.get_best_node(only_good=False)  # Get the best node.
    with open("best_solution.py", "w") as f:
        f.write(best_node.code)

    good_nodes = journal.get_good_nodes()  # Retrieve all good solution nodes.
    for i, node in enumerate(good_nodes):
        filename = f"good_solution_{i+1}.py"
        with open(filename, "w") as f:
            f.write(node.code)

"""### Interpreter (DO NOT MODIFY THIS CELL)"""

"""
DO NOT MODIFY THIS CELL

Python interpreter for executing code snippets and capturing their output.
"""


import logging
import os
import queue
import signal
import sys
import time
import traceback
import zipfile
from pathlib import Path
from shutil import rmtree
import shutil
from multiprocessing import Process, Queue
from typing import Hashable, cast

import humanize
import rich
import shutup
from rich.logging import RichHandler
from rich.syntax import Syntax
from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin


@dataclass
class ExecutionResult(DataClassJsonMixin):
    """
    Result of executing a code snippet in the interpreter.
    Contains the output, execution time, and exception information.
    """
    term_out: list[str]
    exec_time: float
    exc_type: str | None
    exc_info: dict | None = None
    exc_stack: list[tuple] | None = None

def exception_summary(e, exec_file_name):
    """Generates a string that summarizes an exception and its stack trace"""
    tb_lines = traceback.format_exception(e)
    # Combine the traceback lines into a single string, skipping lines that contain "importlib".
    tb_str = "".join(
        [
            line
            for line in tb_lines
            # if "importlib" not in line  # Filter out unwanted traceback lines.
        ]
    )

    exc_info = {}
    if hasattr(e, "args"):
        exc_info["args"] = [str(i) for i in e.args]  # Store the exception arguments as strings.
    for att in ["name", "msg", "obj"]:
        if hasattr(e, att):
            exc_info[att] = str(getattr(e, att))  # Store additional attributes if available.

    tb = traceback.extract_tb(e.__traceback__)  # Extract the traceback information.
    # Create a list of tuples for each frame in the traceback.
    exc_stack = [(t.filename, t.lineno, t.name, t.line) for t in tb]

    return tb_str, e.__class__.__name__, exc_info, exc_stack  # Return the formatted traceback and exception details.

# Define a class that redirects write operations to a multiprocessing queue.
class RedirectQueue:
    def __init__(self, queue, timeout=5):
        self.queue = queue  # Store the provided queue.
        self.timeout = timeout  # Set the timeout for queue operations.

    def write(self, msg):
        try:
            self.queue.put(msg, timeout=self.timeout)  # Attempt to put the message into the queue.
        except queue.Full:
            print.warning("Queue write timed out")  # Warn if the queue is full and the write times out.

    def flush(self):
        pass  # No operation is needed for flushing in this context.

# Define the Interpreter class that simulates a standalone Python REPL.
class Interpreter:
    def __init__(
        self,
        timeout: int = 3600,  # Default timeout of 3600 seconds.
        agent_file_name: str = "runfile.py",  # Default file name for writing the agent's code.
    ):
        """
        Simulates a standalone Python REPL with an execution time limit.

        Args:
            timeout (int, optional): Timeout for each code execution step. Defaults to 3600.
            agent_file_name (str, optional): The name for the agent's code file. Defaults to "runfile.py".
        """
        self.timeout = timeout  # Save the timeout value.
        self.agent_file_name = agent_file_name  # Save the agent file name.
        self.process: Process = None  # Initialize the process attribute (will hold the child process).

    def child_proc_setup(self, result_outq: Queue) -> None:
        # Import shutup to suppress warnings in the child process.
        import shutup

        shutup.mute_warnings()  # Mute all warnings before further execution.

        # Redirect both stdout and stderr to the provided result queue.
        # trunk-ignore(mypy/assignment)
        sys.stdout = sys.stderr = RedirectQueue(result_outq)

    def _run_session(
        self, code_inq: Queue, result_outq: Queue, event_outq: Queue
    ) -> None:
        self.child_proc_setup(result_outq)  # Set up the child process for capturing output.

        global_scope: dict = {}  # Create an empty dictionary to serve as the global scope.
        while True:  # Continuously wait for new code to execute.
            code = code_inq.get()  # Retrieve code from the code input queue.
            with open(self.agent_file_name, "w") as f:  # Open the agent file for writing.
                f.write(code)  # Write the received code into the file.

            event_outq.put(("state:ready",))  # Signal that the interpreter is ready to execute the code.
            try:
                # Compile and execute the code within the global scope.
                exec(compile(code, self.agent_file_name, "exec"), global_scope)
            except BaseException as e:
                # If an exception occurs, generate a summary of the exception.
                tb_str, e_cls_name, exc_info, exc_stack = exception_summary(
                    e,
                    self.agent_file_name,
                )
                result_outq.put(tb_str)  # Put the traceback string into the result queue.
                if e_cls_name == "KeyboardInterrupt":
                    e_cls_name = "TimeoutError"  # Convert a KeyboardInterrupt into a TimeoutError.

                event_outq.put(("state:finished", e_cls_name, exc_info, exc_stack))  # Signal that execution finished with an error.
            else:
                event_outq.put(("state:finished", None, None, None))  # Signal that execution finished successfully.

            os.remove(self.agent_file_name)  # Remove the agent file after execution.

            result_outq.put("<|EOF|>")  # Put an EOF marker to indicate the end of output.

    def create_process(self) -> None:
        # Create three queues for communication with the child process:
        # - code_inq: for sending code to execute.
        # - result_outq: for receiving output from the execution.
        # - event_outq: for receiving state events (like ready and finished).
        # trunk-ignore(mypy/var-annotated)
        self.code_inq, self.result_outq, self.event_outq = Queue(), Queue(), Queue()
        self.process = Process(
            target=self._run_session,  # Set the target function for the child process.
            args=(self.code_inq, self.result_outq, self.event_outq),  # Provide the necessary queues as arguments.
        )
        self.process.start()  # Start the child process.

    def cleanup_session(self):
        if self.process is None:  # If there is no process, nothing to clean up.
            return
        try:
            # Attempt to terminate the child process gracefully.
            self.process.terminate()  # Request the process to terminate.
            self.process.join(timeout=0.5)  # Wait for the process to finish with a 0.5-second timeout.

            if self.process.exitcode is None:  # If the process is still running,
                self.process.kill()  # Forcefully kill the process.
                self.process.join(timeout=0.5)  # Wait again for termination.

                if self.process.exitcode is None:  # If the process still hasn't terminated,
                    os.kill(self.process.pid, signal.SIGKILL)  # Send a SIGKILL signal.
        except Exception as e:
            print(f"Error during process cleanup: {e}")  # Print an error message if cleanup fails.
        finally:
            if self.process is not None:  # If the process exists,
                self.process.close()  # Close the process.
                self.process = None  # Reset the process attribute to None.

    def run(self, code: str, reset_session=True) -> ExecutionResult:
        """
        Execute the provided Python command in a separate process and return its output.

        Parameters:
            code (str): Python code to execute.
            reset_session (bool, optional): Whether to reset the interpreter session before executing the code. Defaults to True.

        Returns:
            ExecutionResult: Object containing the output and metadata of the code execution.
        """

        if reset_session:
            if self.process is not None:
                # If a previous process exists, clean it up before starting a new one.
                self.cleanup_session()
            self.create_process()  # Create a new child process.
        else:
            # For the first execution, reset_session must be True.
            assert self.process is not None

        assert self.process.is_alive()  # Ensure that the child process is running.

        self.code_inq.put(code)  # Send the code to the child process via the queue.

        # Wait for the child process to signal that it is ready.
        try:
            state = self.event_outq.get(timeout=10)  # Wait up to 10 seconds for the "state:ready" event.
        except queue.Empty:
            msg = "REPL child process failed to start execution"
            print.critical(msg)  # Log a critical error if the process does not start.
            while not self.result_outq.empty():
                continue  # Drain the result queue.
            raise RuntimeError(msg) from None
        assert state[0] == "state:ready", state  # Verify that the received state is "state:ready".
        start_time = time.time()  # Record the start time of execution.

        child_in_overtime = False  # Flag to indicate if the child process has exceeded the timeout.

        while True:
            try:
                # Try to get the finished state from the child process.
                state = self.event_outq.get(timeout=1)  # Wait for the "state:finished" event.
                assert state[0] == "state:finished", state  # Ensure the state is "state:finished".
                exec_time = time.time() - start_time  # Calculate the total execution time.
                break  # Exit the loop if execution is finished.
            except queue.Empty:
                # If no event is received, check whether the process is still alive.
                if not child_in_overtime and not self.process.is_alive():
                    msg = "REPL child process died unexpectedly"
                    raise RuntimeError(msg) from None

                # If the process is still running, check if it has exceeded the timeout.
                if self.timeout is None:
                    continue
                running_time = time.time() - start_time  # Determine the running time.
                if running_time > self.timeout:
                    print(f"Execution exceeded timeout of {self.timeout}s")  # Log a timeout message.
                    os.kill(self.process.pid, signal.SIGINT)  # Send SIGINT to the process.
                    child_in_overtime = True  # Mark that the process is now in overtime.

                    # If the process exceeds the timeout by more than 5 seconds, force cleanup.
                    if running_time > self.timeout + 5:
                        self.cleanup_session()  # Clean up the child process.

                        state = (None, "TimeoutError", {}, [])  # Set state to indicate a timeout error.
                        exec_time = self.timeout  # Set the execution time to the timeout limit.
                        break

        output: list[str] = []  # Initialize a list to collect output lines.
        # Collect all output from the result queue until the EOF marker is encountered.
        start_collect = time.time()  # Record the start time for output collection.
        while not self.result_outq.empty() or not output or output[-1] != "<|EOF|>":
            try:
                # If output collection exceeds 5 seconds, log a warning.
                if time.time() - start_collect > 5:
                    print.warning("Output collection timed out")
                    break
                output.append(self.result_outq.get(timeout=1))  # Append the next line of output.
            except queue.Empty:
                continue  # Continue if no output is available immediately.
        output.pop()  # Remove the EOF marker from the output list.

        # Extract exception information from the finished state.
        e_cls_name, exc_info, exc_stack = state[1:]

        if e_cls_name == "TimeoutError":
            # Append a timeout error message to the output if a timeout occurred.
            output.append(
                f"TimeoutError: Execution exceeded the time limit of {humanize.naturaldelta(self.timeout)}"
            )
        else:
            # Append the execution time information to the output.
            output.append(
                f"Execution time: {humanize.naturaldelta(exec_time)} seconds (time limit is {humanize.naturaldelta(self.timeout)})."
            )
        # Return an ExecutionResult object with all the execution details.
        return ExecutionResult(output, exec_time, e_cls_name, exc_info, exc_stack)

"""### Nodes

"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Literal, Optional

from dataclasses_json import DataClassJsonMixin


@dataclass(eq=False)
class Node(DataClassJsonMixin):
    """A single node in the solution tree. Contains code, execution results, and evaluation information."""

    # ---- code & plan ----
    code: str
    plan: str = field(default=None, kw_only=True)  # type: ignore

    # ---- general attrs ----
    step: int = field(default=None, kw_only=True)  # type: ignore
    id: str = field(default_factory=lambda: uuid.uuid4().hex, kw_only=True)
    ctime: float = field(default_factory=lambda: time.time(), kw_only=True)
    parent: Optional["Node"] = field(default=None, kw_only=True)
    children: set["Node"] = field(default_factory=set, kw_only=True)

    # ---- execution info ----
    _term_out: list[str] = field(default=None, kw_only=True)  # type: ignore
    exec_time: float = field(default=None, kw_only=True)  # type: ignore
    exc_type: str | None = field(default=None, kw_only=True)
    exc_info: dict | None = field(default=None, kw_only=True)
    exc_stack: list[tuple] | None = field(default=None, kw_only=True)

    # ---- evaluation ----
    # post-execution result analysis (findings/feedback)
    analysis: str = field(default=None, kw_only=True)  # type: ignore
    metric: float = field(default=None, kw_only=True)  # type: ignore
    # whether the agent decided that the code is buggy
    # -> always True if exc_type is not None or no valid metric
    is_buggy: bool = field(default=None, kw_only=True)  # type: ignore

    def __post_init__(self) -> None:
        if self.parent is not None:
            self.parent.children.add(self)

    @property
    def stage_name(self) -> Literal["draft", "debug", "improve"]:
        """
        Return the stage of the node:
        - "stage" if the node is an initial solution draft
        - "debug" if the node is the result of a debugging step
        - "improve" if the node is the result of an improvement step
        """
        if self.parent is None:
            return "draft"
        return "debug" if self.parent.is_buggy else "improve"

    def absorb_exec_result(self, exec_result: ExecutionResult):
        """Absorb the result of executing the code from this node."""
        self._term_out = exec_result.term_out
        self.exec_time = exec_result.exec_time
        self.exc_type = exec_result.exc_type
        self.exc_info = exec_result.exc_info
        self.exc_stack = exec_result.exc_stack

    @property
    def term_out(self) -> str:
        """Get the terminal output of the code execution (after truncating it)."""
        return trim_long_string("".join(self._term_out))

    @property
    def is_leaf(self) -> bool:
        """Check if the node is a leaf node in the solution tree."""
        return not self.children

    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

    @property
    def debug_depth(self) -> int:
        """
        Length of the current debug path
        - 0 if the node is not a debug node (parent is not buggy)
        - 1 if the parent is buggy but the skip parent isn't
        - n if there were n consecutive debugging steps
        """
        if self.stage_name != "debug":
            return 0
        return self.parent.debug_depth + 1  # type: ignore

"""### Tree"""

@dataclass
class Journal(DataClassJsonMixin):
    """A collection of nodes representing the solution tree."""

    nodes: list[Node] = field(default_factory=list)

    def __getitem__(self, idx: int) -> Node:
        return self.nodes[idx]

    def __len__(self) -> int:
        """Return the number of nodes in the journal."""
        return len(self.nodes)

    def append(self, node: Node) -> None:
        """Append a new node to the journal."""
        node.step = len(self.nodes)
        self.nodes.append(node)

    @property
    def draft_nodes(self) -> list[Node]:
        """Return a list of nodes representing intial coding drafts"""
        return [n for n in self.nodes if n.parent is None]

    @property
    def buggy_nodes(self) -> list[Node]:
        """Return a list of nodes that are considered buggy by the agent."""
        return [n for n in self.nodes if n.is_buggy]

    @property
    def good_nodes(self) -> list[Node]:
        """Return a list of nodes that are not considered buggy by the agent."""
        return [n for n in self.nodes if not n.is_buggy]

    def get_metric_history(self) -> list[float]:
        """Return a list of all metric values in the journal."""
        return [n.metric for n in self.nodes]

    def get_good_nodes(self) -> Node:
        return [n for n in self.nodes if not n.is_buggy]

    def get_best_node(self, only_good=True) -> None | Node:
        """Return the best solution found so far (node with the highest validation metric)."""
        if only_good:
            nodes = self.good_nodes
            if not nodes:
                return None
        else:
            nodes = self.nodes
        return min(nodes, key=lambda n: n.metric)

    def generate_summary(self, include_code: bool = False) -> str:
        """Generate a summary of the journal for the agent."""
        summary = []
        for n in self.good_nodes:
            summary_part = f"Design: {n.plan}\n"
            if include_code:
                summary_part += f"Code: {n.code}\n"
            summary_part += f"Results: {n.analysis}\n"
            summary_part += f"Validation Metric (Mean Squared Error): {n.metric}\n"
            summary.append(summary_part)
        return "\n-------------------------------\n".join(summary)

"""### Agent"""

import random
from typing import Any, Callable, cast

import re
import sys
import json
import humanize

from pydantic import BaseModel

ExecCallbackType = Callable[[str, bool], ExecutionResult]


class Agent:
    def __init__(
        self,
        cfg,
        journal: Journal,
    ):
        super().__init__()
        self.cfg = cfg
        self.journal = journal
        self.data_preview: str | None = None

    def search_policy(self) -> Node | None:
        """Select a node to work on (or None to draft a new node)."""
        search_cfg = self.cfg.agent.search

        # initial drafting
        if len(self.journal.draft_nodes) < search_cfg.num_drafts:
            return None

        # debugging
        if random.random() < search_cfg.debug_prob:
            # nodes that are buggy + leaf nodes + debug depth < max debug depth
            debuggable_nodes = [
                n
                for n in self.journal.buggy_nodes
                if n.is_leaf
            ]
            if debuggable_nodes:
                return random.choice(debuggable_nodes)


        # back to drafting if no nodes to improve
        good_nodes = self.journal.good_nodes
        if not good_nodes:
            return None

        # greedy
        greedy_node = self.journal.get_best_node()

        return greedy_node

    def plan_and_code_query(self, system_message, user_message, retries=3) -> tuple[str, str]:
        """Generate a natural language plan + code in the same LLM call and split them apart."""
        completion_text = None
        for _ in range(retries):

            response = generate_response(
                myModel,
                _messages=[
                    {'role': 'system', "content": system_message},
                    {'role': 'user', "content": user_message}
                ]
            )
            completion_text = response
            code = extract_code(completion_text)
            nl_text = extract_text_up_to_code(completion_text)

            if code:
                return nl_text, code

            print("Plan + code extraction failed, retrying...")
        print("Final plan + code extraction attempt failed, giving up...")
        return "", completion_text

    def _draft(self) -> Node:
        strategies = [
            # Linear Models
            # "Ridge regression with optimized alpha",
            # "Lasso regression with optimized alpha",
            # "ElasticNet regression (L1 + L2 regularization)",
            # "Bayesian Ridge regression",
            
            # # Tree-based Models (Shallow or Low Complexity)
            # "ExtraTrees with max_depth=5, n_estimators=50",
            # "Gradient Boosting with max_depth=3, learning_rate=0.1",
            # "Random Forest with max_depth=5, n_estimators=50",
            # "HistGradientBoosting with early stopping",
            
            # # KNN and SVM Variants
            # "KNeighborsRegressor with k=5 and weighted distances",
            # "Support Vector Regression (SVR) with linear kernel",

            # # Polynomial and Feature Engineering
            # "Polynomial regression (degree=2) with Ridge regularization",
            # "Sparse feature selection with Ridge regression",
            
            # # Ensemble and Hybrid Approaches
            # "StackingRegressor with Ridge, Lasso, and Gradient Boosting",
            "Averaging ensemble of Ridge, Lasso, and ElasticNet"
        ]
        
        selected_strategy = random.choice(strategies)
        
        system_prompt = "You are a ML engineer. Create a regression model avoiding overfitting."

        user_prompt = [
            f"Strategy: {selected_strategy}",
            "DATA:",
            "- Files: ML2025Spring-hw2-public/train.csv and test.csv",
            "- Target: tested_positive_day3",
            "- Sample Output: ML2025Spring-hw2-public/sample_submission.csv, which contains two columns: id and tested_positive_day3",
            "REQUIREMENTS:",
            "1. Use regularization",
            "2. Implement cross-validation",
            "3. Print MSE value with sys.stdout.flush()",
            "4. Save predictions to submission.csv with header tested_positive_day3",
            "5. AVOID using 'loss'='ls' or 'max_features'='auto' in GradientBoostingRegressor",
            "6. AVOID using LGBMRegressor",
            "7. Limit CPU usage with n_jobs=2",
            "8. Use weighted average rather than simple average for ensemble"
        ]

        system_message = system_prompt
        user_message = "\n".join(user_prompt)
        plan, code = self.plan_and_code_query(system_message=system_message, user_message=user_message)
        print(f"Draft Completed, using {selected_strategy}")
        return Node(plan=plan, code=code)


    def _improve(self, parent_node: Node) -> Node:
        current_mse = parent_node.metric
        print(f"============ _improve ============")
        print(f"Parent MSE: {current_mse}")
        
        if current_mse < OVERFITTING_THRESHOLD:
            print(f"Overfitting detected (MSE: {current_mse} < {OVERFITTING_THRESHOLD})")
            system_prompt = "You are an ML engineer. Reduce overfitting in this regression model."
            user_prompt = [
                "ISSUE: Model is overfitted (MSE too low).",
                "FIX:",
                "- Increase Ridge alpha",
                "- Reduce feature complexity",
                "OUTPUT:",
                "- Print MSE with sys.stdout.flush()",
                "- Save predictions to submission.csv"
            ]
        else:
            print(f"No overfitting detected (MSE: {current_mse})")
            system_prompt = "You are a Ridge regression specialist. Optimize this model."
            user_prompt = [
                f"Current MSE: {current_mse}. Improve without overfitting.",
                "PREVIOUS CODE:",
                f"```python\n{str(parent_node.code)}\n```",
                "OPTIMIZATION:",
                "- Tune Ridge alpha",
                "- IMPORTANT: Ensure train/test data consistency",
                "  * Remove 'id' column before transformations",
                "  * Keep identical feature sets between train/test",
                "- Simple feature engineering and preprocessing",
                "OUTPUT:",
                "- Print MSE",
                "- Save predictions to submission.csv",
                "- Use n_jobs=2"
            ]
    
        system_message = system_prompt
        user_message = "\n".join(user_prompt)
        plan, code = self.plan_and_code_query(system_message=system_message, user_message=user_message)
        return Node(plan=plan, code=code, parent=parent_node)

    def _debug(self, parent_node: Node) -> Node:
        system_prompt = "You are a Python debugging expert for Ridge regression models."
        
        user_prompt = [
            "ERROR CODE:",
            f"```python\n{parent_node.code}\n```",
            "ERROR OUTPUT:",
            f"```\n{parent_node.term_out[:300]}...\n```",
            "FIX:",
            "- Ensure Ridge regression with alpha ~10.23 works correctly",
            "- Verify StandardScaler is applied before Ridge",
            "- Fix any errors in feature engineering",
            "- Ensure cross-validation is implemented properly",
            "OUTPUT:",
            "- Print MSE with sys.stdout.flush()",
            "- Save predictions to submission.csv"
        ]

        system_message = system_prompt
        user_message = "\n".join(user_prompt)
        plan, code = self.plan_and_code_query(system_message=system_message, user_message=user_message)
        return Node(plan=plan, code=code, parent=parent_node)


    def update_data_preview(
        self,
    ):
        self.data_preview = data_preview_generate(cfg.data_dir)

    def step(self, exec_callback: ExecCallbackType):
        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview()

        parent_node = self.search_policy()

        if parent_node is None:
            print("Drafting new solution...")
            result_node = self._draft()
        elif parent_node.is_buggy:
            print("Debugging existing solution...")
            result_node = self._debug(parent_node)
        else:
            print("Improving existing solution...")
            result_node = self._improve(parent_node)

        print("Executing code...")
        try:
            # 設置最大執行時間
            MAX_EXEC_TIME = 300  # 最長執行 300 秒
            start_time = time.time()
            
            # 執行程式並保存結果
            exec_result = exec_callback(result_node.code, True)
            
            # 檢查執行時間
            exec_time = time.time() - start_time
            if exec_time > MAX_EXEC_TIME:
                print(f"Execution time is too long: {exec_time}s")
                result_node.is_buggy = True
                result_node.analysis = f"Execution time is too long: {exec_time}s, maybe stuck in an infinite loop"
                result_node.metric = 999.0  # 設為較大值以避免被選為最佳方案
            else:
                # 正常解析結果
                self.parse_exec_result(node=result_node, exec_result=exec_result)
                
        except Exception as e:
            # 捕獲任何執行錯誤
            print(f"Execution error: {type(e).__name__}: {str(e)}")
            result_node.is_buggy = True
            result_node.analysis = f"Execution error: {str(e)}"
            result_node.metric = 999.0
        finally:
            # 強制清理資源
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 終止所有可能的 Python 子進程
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    # 檢查是否為 Python 進程
                    if proc.info['name'] == 'python3' and proc.pid != os.getpid():
                        proc_cmdline = ' '.join(proc.info['cmdline'] or [])
                        # 只終止那些可能由我們的代碼生成的 Python 進程
                        if 'runfile.py' in proc_cmdline:
                            print(f"stop subprocess: {proc.pid}")
                            proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
                
        print("Execution completed.")
        self.journal.append(result_node)

    def parse_exec_result(self, node: Node, exec_result: ExecutionResult):
        node.absorb_exec_result(exec_result)
        print("----------- node terminal output: -----------")
        print(node.term_out)
        print("----------- end of node terminal output -----------")

        # Direct extraction of MSE from terminal output
        mse_match = re.search(r"MSE:?\s*([\d\.]+)", node.term_out, re.IGNORECASE)
        if mse_match:
            try:
                extracted_mse = float(mse_match.group(1))
                print(f"Directly extracted MSE: {extracted_mse}")
                
                # Check if MSE is suspiciously low (indicates overfitting)
                if extracted_mse < OVERFITTING_THRESHOLD:
                    print("WARNING: Extremely low MSE indicates severe overfitting")
                    node.is_buggy = True
                    node.analysis = "Detected severe overfitting (MSE too close to zero)"
                    node.metric = 999.0  # Set to high value to ensure it doesn't get selected as best
                else:
                    node.metric = extracted_mse
                    
            except ValueError:
                print("Failed to convert extracted MSE to float")

        system_prompt = "You are a highly skilled AI assistant with expertise in debugging and evaluating Python code execution. \
                        Your task is to analyze the execution output of a machine learning task, extract key evaluation results, \
                        and determine whether the code contains errors. Identify exceptions, compute relevant metrics if possible, \
                        and summarize the performance."

        # ================  Extract Evaluation Results from Execution Output  ================
        # 修改 parse_exec_result 方法中的 user_prompt 部分
        user_prompt = f"""
            The task is:
            {self.cfg.task_goal}
        
            Code implementation (Step {node.step}, Stage: {node.stage_name}):
            {wrap_code(node.code)}
        
            Execution output:
            {wrap_code(node.term_out, lang="")}
            
            Please analyze the output and provide your response in the following JSON format:
            {{
                "analysis": {{
                    "summary": "Brief summary of what is actually visible in the output",
                    "contains_mse": true/false,
                    "error_type": "Type of error if explicitly shown in output, otherwise null",
                    "error_line": -1,
                    "error_explanation": "Explanation of the actual error if present, otherwise null"
                }},
                "metric": 0.0,
                "is_buggy": true
            }}
            
            Where:
            1. **analysis.summary**: Brief factual overview based ONLY on what is visible in the output
            2. **analysis.contains_mse**: Boolean (true/false) indicating if "MSE: [number]" appears in the output
            3. **analysis.error_type**: ONLY include if an actual error appears in the output
            4. **error_line**: Line number if available, otherwise -1
            5. **error_explanation**: Explanation ONLY for actual errors in the output
            6. **metric**: The exact MSE value if "MSE: [number]" is present, otherwise 0.0
            7. **is_buggy**: Set to true if there are actual errors in the output, otherwise false
        """

        system_message = system_prompt
        user_message = " ".join(user_prompt)

        response = generate_response(
            myModel,
            _messages=[
                {'role': 'system', "content": system_message},
                {'role': 'user', "content": user_message}
            ]
        )

        # ================  Evaluation  ================
        print("----------- Evaluation Response: -----------")
        print(response)
        print("----------- End of Evaluation Response -----------")
        
        # 解析 LLM 回應

        
        # 嘗試解析為 JSON (如果回應是 JSON 格式)
        try:
            # 去除可能的 markdown 程式碼區塊標記
            json_text = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_text:
                cleaned_response = json_text.group(1).strip()
            else:
                cleaned_response = response.strip()
            
            # 解析 JSON
            json_response = json.loads(cleaned_response)
            
            # 處理 analysis 物件，整合錯誤資訊到 analysis 字串中
            if isinstance(json_response.get("analysis"), dict):
                analysis_dict = json_response.get("analysis", {})
                summary = analysis_dict.get("summary", "")
                error_type = analysis_dict.get("error_type")
                error_line = analysis_dict.get("error_line", -1)
                error_explanation = analysis_dict.get("error_explanation")
                
                # 整合所有資訊到一個字串
                analysis_parts = []
                if summary:
                    analysis_parts.append(f"Summary: {summary}")
                if error_type:
                    analysis_parts.append(f"Error Type: {error_type}")
                    if error_line and error_line != -1:
                        analysis_parts.append(f"Error Line: {error_line}")
                    if error_explanation:
                        analysis_parts.append(f"Error Explanation: {error_explanation}")
                
                node.analysis = "\n".join(analysis_parts) if analysis_parts else "No analysis provided."
            else:
                # 如果 analysis 是字串而非字典，直接使用
                node.analysis = json_response.get("analysis", "No analysis provided.")
            
            node.metric = float(json_response.get("metric", 0.0))
            node.is_buggy = json_response.get("is_buggy", True)
        except (json.JSONDecodeError, TypeError):
            # 如果不是 JSON，使用正則表達式解析
            
            # 提取 MSE
            mse_match = re.search(r"MSE:?\s*([\d\.]+)", response, re.IGNORECASE)
            if mse_match:
                try:
                    node.metric = float(mse_match.group(1))
                except ValueError:
                    node.metric = 0.0
            else:
                node.metric = 0.0
            
            # 提取分析
            analysis_parts = []
            error_match = re.search(r"Error Analysis:(.*?)(?:###|$)", response, re.DOTALL | re.IGNORECASE)
            if error_match:
                analysis_parts.append(error_match.group(1).strip())
            
            summary_match = re.search(r"Summary:(.*?)(?:###|$)", response, re.DOTALL | re.IGNORECASE)
            if summary_match:
                analysis_parts.append(summary_match.group(1).strip())
            
            node.analysis = "\n".join(analysis_parts) if analysis_parts else "No analysis provided."
            
            # 判斷程式碼是否有問題 (修改後不再將沒有 MSE 視為錯誤)
            node.is_buggy = (
                node.exc_type is not None or  # 程式碼執行出錯
                # node.metric == 0.0 or  # 移除這行
                "error" in response.lower() or  # 回應中包含 "error"
                "exception" in response.lower()  # 回應中包含 "exception"
            )

        print("Analysis:", node.analysis)
        print("Metric:", node.metric)
        print("Is Buggy:", node.is_buggy)
        
        # node.analysis = response.get("summary", "No analysis provided.")
        # node.metric = response.get("metric", None)

        # # 判斷是否為 buggy：
        # node.is_buggy = (
        #     response.get("is_buggy", False)  # LLM 判定
        #     or node.exc_type is not None  # 有異常
        #     or node.metric is None  # 無效 metric
        # )

        # # 預防 metric 仍為 None
        # if node.metric is None:
        #     node.metric = 0.0


"""### Text Processing"""

import json
import re

def wrap_code(code: str, lang="python") -> str:
    """Wraps code with three backticks."""
    return f"```{lang}\n{code}\n```"


def is_valid_python_script(script):
    """Check if a script is a valid Python script."""
    try:
        compile(script, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def extract_jsons(text):
    """Extract all JSON objects from the text. Caveat: This function cannot handle nested JSON objects."""
    json_objects = []

    # Find {} by regular expression
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)

    # Try to transform string into json objects
    for match in matches:
        try:
            json_obj = json.loads(match)
            json_objects.append(json_obj)
        except json.JSONDecodeError:
            pass

    return json_objects

def trim_long_string(string, threshold=3100, k=1500):
    # Check if the length of the string is longer than the threshold
    if len(string) > threshold:
        # Output the first k and last k characters
        first_k_chars = string[:k]
        last_k_chars = string[-k:]

        truncated_len = len(string) - 2 * k

        return f"{first_k_chars}\n ... [{truncated_len} characters truncated] ... \n{last_k_chars}"
    else:
        return string

def extract_code(text):
    """Extract python code blocks from the text."""
    parsed_codes = []

    # When code is in a text or python block
    matches = re.findall(r"```(python)?\n*(.*?)\n*```", text, re.DOTALL)
    for match in matches:
        code_block = match[1]
        parsed_codes.append(code_block)

    # When the entire text is code or backticks of the code block is missing
    if len(parsed_codes) == 0:
        matches = re.findall(r"^(```(python)?)?\n?(.*?)\n?(```)?$", text, re.DOTALL)
        if matches:
            code_block = matches[0][2]
            parsed_codes.append(code_block)

    # validate the parsed codes
    valid_code_blocks = [
        c for c in parsed_codes if is_valid_python_script(c)
    ]
    return "\n\n".join(valid_code_blocks)

def extract_text_up_to_code(s):
    """Extract (presumed) natural language text up to the start of the first code block."""
    if "```" not in s:
        return ""
    return s[: s.find("```")].strip()

"""### Feature Selection"""

import json
from pathlib import Path

import humanize
import pandas as pd


def preview_csv(p: Path) -> str:
    """Generate a textual preview of a csv file, highlighting useful features for prediction."""
    
    df = pd.read_csv(p)
    out = []
    
    out.append(f"-> {str(p)} has {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # Feature categorization based on dataset analysis
    useful_features = {
        "Health Indicators": ["cli_day1", "ili_day1", "hh_cmnty_cli_day1", "nohh_cmnty_cli_day1", "wnohh_cmnty_cli_day1"],
        "Behavior Indicators": ["wearing_mask_7d_day1", "wshop_indoors_day1", "wrestaurant_indoors_day1", "public_transit_day1", "wlarge_event_indoors_day1"],
        "Belief Indicators": ["wbelief_masking_effective_day1", "wbelief_distancing_effective_day1"],
        "Environmental Indicators": ["wothers_masked_public_day1", "wothers_distanced_public_day1", "wcovid_vaccinated_friends_day1"],
        "Mental Indicators": ["wworried_catch_covid_day1", "worried_finances_day1"],
        "Target Variable": ["tested_positive_day1", "tested_positive_day2"],
    }
    
    # Filter out non-useful features like state one-hot encoding
    dataset_features = set(df.columns.tolist())
    relevant_features = {key: [col for col in cols if col in dataset_features] for key, cols in useful_features.items()}
    
    # Generate output
    for category, cols in relevant_features.items():
        if cols:
            out.append(f"{category}: {', '.join(cols)}")
    
    return "\n".join(out)

def data_preview_generate(base_path):
    """
    Generate a textual preview of a directory
    """

    result = []
    files = [p for p in Path(base_path).iterdir()]
    for f in sorted(files):
        result.append(preview_csv(f))

    result = "\n\n".join(result)

    return result

"""## Config"""

import torch
import random
import numpy as np

# DO NOT MODIFY THIS CELL
class Config:
    """
    A recursive configuration class that converts a dictionary into an object
    with attributes accessible using dot notation.
    """

    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)

def set_seed(seed=531):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed()

config = {
    # experiment configurations
    "exp_name": "ML2025_HW2",
    "data_dir":  Path("ML2025Spring-hw2-public").resolve(),

    # the description of the task
    "task_goal": "Given the survey results from the past two days in a specific state in the U.S.,\
                  predict the probability of testing positive on day 3. \
                  The evaluation metric is Mean Squared Error (MSE).",

    "agent": {
        # the number of iterations
        "steps": 3,
        "search": {
            # decide whether to debug or improve
            "debug_prob": 0.8,
            # the number of draft generated before improving/debugging
            "num_drafts": 3,
        },
    },
}

cfg = Config(config)

"""## Main"""

def main():

    def exec_callback(*args, **kwargs):
        res = interpreter.run(*args, **kwargs)
        return res

    journal = Journal()
    agent = Agent(
        cfg=cfg,
        journal=journal,
    )

    interpreter = Interpreter()

    global_step = len(journal)
    while global_step < cfg.agent.steps:
        print(f"\n==================== Step {global_step + 1} ====================")
        # run agent
        agent.step(exec_callback=exec_callback)
        # save results for this iteration
        save_run(cfg, journal)
        # get currect step
        global_step = len(journal)


    # Kill created child process
    interpreter.cleanup_session()


if __name__ == "__main__":
    main()
