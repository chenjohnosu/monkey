"""
Batch processing functionality for document analysis toolkit
"""

import os
import re
import sys
from typing import Dict, List, Optional, Any
from core.engine.logging import debug, info, warning, error

class BatchProcessor:
    def __init__(self, command_processor):
        self.command_processor = command_processor
        self.config = command_processor.config
        self.variables = {}  # For variable substitution
        self.current_line = 0
        self.current_file = None
        self.error_count = 0
        self.hpc_mode = self.config.get('system.hpc_mode', False)
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

            # Read the file
            with open(filepath, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            # First pass - process variable declarations
            self._process_variable_declarations(lines)

            # Second pass - execute commands
            for i, line in enumerate(lines):
                self.current_line = i + 1

                # Skip empty lines and comments
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Skip variable declarations (already processed)
                if self._is_variable_declaration(line):
                    continue

                # Process flow control statements
                if line.startswith('if '):
                    self._process_conditional(line, lines, i)
                    continue

                if line.startswith('exit'):
                    parts = line.split()
                    if len(parts) > 1:
                        try:
                            exit_code = int(parts[1])
                            info(f"Batch file requested exit with code {exit_code}")
                            sys.exit(exit_code)
                        except ValueError:
                            warning(f"Invalid exit code: {parts[1]}")
                    # Exit with success if no code specified
                    info("Batch file requested exit")
                    return True

                # Process regular command
                expanded_line = self._expand_variables(line)
                if expanded_line.startswith('/'):
                    success = self._execute_command(expanded_line)
                    if not success and self._is_exit_on_error():
                        error(f"Command failed at line {self.current_line}, exiting batch processing")
                        return False
                else:
                    warning(f"Invalid command format at line {self.current_line}: {line}")
                    self.error_count += 1

            # Return success based on error count
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

                    # Store the variable
                    self.variables[var_name] = var_value
                    debug(self.config, f"Batch variable set: {var_name}={var_value}")

    def _is_variable_declaration(self, line: str) -> bool:
        return re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*.+', line) is not None

    def _expand_variables(self, line: str) -> str:
        result = line

        # Find and replace variables in the format ${VAR_NAME}
        for var_name, var_value in self.variables.items():
            result = result.replace(f"${{{var_name}}}", var_value)

        return result

    def _execute_command(self, command: str) -> bool:
        try:
            info(f"Executing batch command: {command}")
            self.command_processor.process_command(command)
            return True
        except Exception as e:
            error(f"Error executing command: {command}")
            error(f"Error details: {str(e)}")
            self.error_count += 1
            return False

    def _process_conditional(self, line: str, lines: List[str], current_index: int) -> int:
        # Example: if exists ${file} then
        match = re.match(r'if\s+exists\s+(.+)\s+then', line)
        if match:
            file_path = match.group(1)
            file_path = self._expand_variables(file_path)

            if os.path.exists(file_path):
                # Condition is true, continue with next line
                return current_index + 1
            else:
                # Find the matching 'endif' or 'else'
                for i in range(current_index + 1, len(lines)):
                    if lines[i].strip() == 'endif':
                        return i + 1
                    elif lines[i].strip() == 'else':
                        return i + 1

        # Process other conditionals like error checks
        match = re.match(r'if\s+errorlevel\s+(\d+)\s+then', line)
        if match:
            error_level = int(match.group(1))

            if self.error_count >= error_level:
                # Condition is true, continue with next line
                return current_index + 1
            else:
                # Find the matching 'endif' or 'else'
                for i in range(current_index + 1, len(lines)):
                    if lines[i].strip() == 'endif':
                        return i + 1
                    elif lines[i].strip() == 'else':
                        return i + 1

        # Default to next line
        return current_index + 1

    def _is_exit_on_error(self) -> bool:
        exit_on_error = self.variables.get('EXIT_ON_ERROR', 'true').lower()

        # In HPC mode, always exit on error unless explicitly set to false
        if self.hpc_mode and exit_on_error != 'false':
            return True

        return exit_on_error == 'true'