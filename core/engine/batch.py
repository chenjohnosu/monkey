import os
import re
import sys
import datetime
import shlex
from typing import Dict, List, Optional, Any
from core.engine.logging import debug, info, warning, error, LogManager
from core.engine.utils import ensure_dir

class BatchProcessor:
    def __init__(self, command_processor):
        self.command_processor = command_processor
        self.config = command_processor.config
        self.variables = {}
        self.current_line = 0
        self.current_file = None
        self.error_count = 0
        self.hpc_mode = self.config.get('system.hpc_mode', False)
        self.exit_on_error = self.config.get('batch.exit_on_error', True)
        debug(self.config, "Batch processor initialized")

    def process_file(self, filepath: str) -> bool:
        debug(self.config, f"Processing batch file: {filepath}")

        if not os.path.exists(filepath):
            error(f"Batch file not found: {filepath}")
            return False

        try:
            self.current_file = filepath
            self.current_line = 0
            self.error_count = 0

            # Ensure we have a log file for batch output
            log_file = LogManager.get_log_file_path()
            if not log_file:
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                workspace = self.config.get('workspace.default', 'default')
                logs_dir = os.path.join('logs', workspace)
                ensure_dir(logs_dir)
                log_file = os.path.join(logs_dir, f"batch_{timestamp}.txt")
                LogManager.add_file_handler(log_file)
                info(f"Set up batch log file: {log_file}")

            # Read the file
            with open(filepath, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            # Process variable declarations first
            self._process_variable_declarations(lines)

            # Then execute commands
            self._execute_commands(lines)

            # Report results
            if self.error_count > 0:
                warning(f"Batch processing completed with {self.error_count} errors")
                return False

            info(f"Batch processing completed successfully")
            return True

        except Exception as e:
            error(f"Error processing batch file: {str(e)}")
            import traceback
            debug(self.config, traceback.format_exc())
            return False

    def _process_variable_declarations(self, lines: List[str]) -> None:
        for i, line in enumerate(lines):
            line = line.strip()
            if self._is_variable_declaration(line):
                parts = line.split('=', 1)
                if len(parts) == 2:
                    var_name = parts[0].strip()
                    var_value = parts[1].strip()

                    # Remove quotes if present
                    if var_value.startswith('"') and var_value.endswith('"'):
                        var_value = var_value[1:-1]
                    elif var_value.startswith("'") and var_value.endswith("'"):
                        var_value = var_value[1:-1]

                    # Handle variable expansion within values
                    var_value = self._expand_variables(var_value)

                    # Store the variable
                    self.variables[var_name] = var_value
                    debug(self.config, f"Batch variable set: {var_name}={var_value}")

    def _execute_commands(self, lines: List[str]) -> None:
        i = 0
        while i < len(lines):
            self.current_line = i + 1
            line = lines[i].strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                i += 1
                continue
                
            # Skip variable declarations
            if self._is_variable_declaration(line):
                i += 1
                continue
                
            # Process control flow statements
            if line.startswith('if '):
                i = self._process_conditional(line, lines, i)
                continue
                
            if line.startswith('exit'):
                self._process_exit(line)
                return
                
            # Process command
            if line.startswith('/'):
                expanded_line = self._expand_variables(line)
                success = self._execute_command(expanded_line)
                
                if not success and self.exit_on_error:
                    error(f"Command failed at line {self.current_line}, exiting batch processing")
                    if self.hpc_mode:
                        sys.exit(1)
                    return
            else:
                warning(f"Invalid command format at line {self.current_line}: {line}")
                self.error_count += 1
                
            # Flush log file
            if LogManager.file_handler:
                LogManager.file_handler.flush()
                
            i += 1

    def _process_exit(self, line: str) -> None:
        parts = line.split()
        if len(parts) > 1:
            try:
                exit_code = int(parts[1])
                info(f"Batch file requested exit with code {exit_code}")
                if self.hpc_mode:
                    sys.exit(exit_code)
            except ValueError:
                warning(f"Invalid exit code: {parts[1]}")

    def _is_variable_declaration(self, line: str) -> bool:
        return re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*.+', line) is not None

    def _expand_variables(self, line: str) -> str:
        result = line
        
        # Find and replace ${VAR_NAME} format
        var_pattern = r'\${([a-zA-Z_][a-zA-Z0-9_]*)}'
        matches = re.findall(var_pattern, line)
        
        for var_name in matches:
            if var_name in self.variables:
                placeholder = f"${{{var_name}}}"
                result = result.replace(placeholder, self.variables[var_name])
                
        # Also support $VAR_NAME format
        var_pattern = r'(?<!\$)\$([a-zA-Z_][a-zA-Z0-9_]*)'
        matches = re.findall(var_pattern, line)
        
        for var_name in matches:
            if var_name in self.variables:
                placeholder = f"${var_name}"
                result = result.replace(placeholder, self.variables[var_name])
                
        return result

    def _execute_command(self, command: str) -> bool:
        try:
            info(f"Executing batch command: {command}")
            
            # Use shlex to handle quoted arguments correctly
            tokens = shlex.split(command)
            cmd = tokens[0].lower()
            args = tokens[1:]
            
            # Process command properly with all arguments
            if cmd.startswith('/'):
                self.command_processor._handle_system_command(cmd[1:], args)
            else:
                warning(f"Invalid command format: {command}")
                self.error_count += 1
                return False
                
            return True
        except Exception as e:
            error(f"Error executing command: {command}")
            error(f"Error details: {str(e)}")
            self.error_count += 1
            return False

    def _process_conditional(self, line: str, lines: List[str], current_index: int) -> int:
        # File existence condition
        match = re.match(r'if\s+exists\s+(.+)\s+then', line)
        if match:
            file_path = match.group(1)
            file_path = self._expand_variables(file_path)

            condition_met = os.path.exists(file_path)
            debug(self.config, f"Condition 'exists {file_path}': {condition_met}")
            
            # Find the matching 'endif' or 'else'
            i = current_index + 1
            while i < len(lines):
                if lines[i].strip() == 'endif':
                    # If condition is false, skip to after endif
                    return i + 1 if not condition_met else current_index + 1
                elif lines[i].strip() == 'else':
                    # If condition is true, skip to after else
                    # If condition is false, execute starting at next line
                    if condition_met:
                        # Find the matching endif
                        j = i + 1
                        while j < len(lines):
                            if lines[j].strip() == 'endif':
                                return j + 1
                            j += 1
                    else:
                        return i + 1
                i += 1

        # Error level condition
        match = re.match(r'if\s+errorlevel\s+(\d+)\s+then', line)
        if match:
            error_level = int(match.group(1))
            condition_met = self.error_count >= error_level
            debug(self.config, f"Condition 'errorlevel {error_level}': {condition_met}")
            
            # Find the matching 'endif' or 'else'
            i = current_index + 1
            while i < len(lines):
                if lines[i].strip() == 'endif':
                    return i + 1 if not condition_met else current_index + 1
                elif lines[i].strip() == 'else':
                    if condition_met:
                        j = i + 1
                        while j < len(lines):
                            if lines[j].strip() == 'endif':
                                return j + 1
                            j += 1
                    else:
                        return i + 1
                i += 1

        # Default: move to next line
        return current_index + 1