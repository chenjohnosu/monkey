"""
Output formatting and saving
"""

import os
import json
import datetime
from core.engine.utils import timestamp_filename, ensure_dir
from core.engine.logging import debug

class OutputManager:
    """Manages output formatting and saving"""
    
    def __init__(self, config):
        """Initialize the output manager"""
        self.config = config
        self.buffer = None
        self.session_file = None
        ensure_dir('logs')
        debug(config, "Output manager initialized")
    
    def add_to_buffer(self, query, response, docs=None):
        """
        Add a query-response pair to the buffer
        
        Args:
            query (str): User query
            response (str): System response
            docs (list, optional): Retrieved documents
        """
        debug(self.config, "Adding to output buffer")
        
        self.buffer = {
            'timestamp': datetime.datetime.now().isoformat(),
            'query': query,
            'response': response,
            'documents': docs if docs else []
        }
    
    def save_buffer(self, workspace):
        """
        Save the current buffer to a file
        
        Args:
            workspace (str): Current workspace
            
        Returns:
            str: Path to the saved file, or None if buffer is empty
        """
        debug(self.config, "Saving output buffer")
        
        if not self.buffer:
            return None
        
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join('logs', workspace)
        ensure_dir(logs_dir)
        
        # Get output format
        output_format = self.config.get('system.output_format')
        
        # Generate filename
        filename = timestamp_filename('query', output_format)
        filepath = os.path.join(logs_dir, filename)
        
        # Save file based on format
        if output_format == 'json':
            self._save_json(filepath, self.buffer)
        elif output_format == 'md':
            self._save_markdown(filepath, self.buffer)
        else:  # Default to txt
            self._save_text(filepath, self.buffer)
        
        return filepath
    
    def start_session_saving(self, workspace):
        """
        Start saving session output to a file

        Args:
            workspace (str): Current workspace
        """
        debug(self.config, "Starting session saving")

        # Create logs directory if it doesn't exist
        logs_dir = os.path.join('logs', workspace)
        ensure_dir(logs_dir)

        # Get output format
        output_format = self.config.get('system.output_format')

        # Generate filename
        filename = timestamp_filename('session', output_format)
        filepath = os.path.join(logs_dir, filename)

        # Open file for writing
        self.session_file = {
            'path': filepath,
            'format': output_format,
            'file': open(filepath, 'w', encoding='utf-8')
        }

        # Write header
        if output_format == 'json':
            self.session_file['file'].write('{\n')
            self.session_file['file'].write(f'  "session_start": "{datetime.datetime.now().isoformat()}",\n')
            self.session_file['file'].write('  "workspace": "' + workspace + '",\n')
            self.session_file['file'].write('  "interactions": [\n')
            self.session_file['first'] = True
        elif output_format == 'md':
            self.session_file['file'].write(f'# Session Log - {workspace}\n\n')
            self.session_file['file'].write(f'Session started: {datetime.datetime.now().isoformat()}\n\n')
        else:  # Default to txt
            self.session_file['file'].write(f'=== Session Log - {workspace} ===\n\n')
            self.session_file['file'].write(f'Session started: {datetime.datetime.now().isoformat()}\n\n')

    def stop_session_saving(self):
        """
        Stop saving session output

        Returns:
            str: Path to the saved file
        """
        debug(self.config, "Stopping session saving")

        if not self.session_file:
            return None

        # Write footer and close file
        if self.session_file['format'] == 'json':
            self.session_file['file'].write('\n  ],\n')
            self.session_file['file'].write(f'  "session_end": "{datetime.datetime.now().isoformat()}"\n')
            self.session_file['file'].write('}\n')
        else:
            self.session_file['file'].write('\n\n')
            self.session_file['file'].write(f'Session ended: {datetime.datetime.now().isoformat()}\n')

        filepath = self.session_file['path']
        self.session_file['file'].close()
        self.session_file = None

        return filepath

    def _save_json(self, filepath, data):
        """Save data as JSON"""
        with open(filepath, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=2)

    def _save_markdown(self, filepath, data):
        """Save query-response as Markdown"""
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(f'# Query Log\n\n')
            file.write(f'Timestamp: {data["timestamp"]}\n\n')
            file.write(f'## Query\n\n')
            file.write(f'{data["query"]}\n\n')
            file.write(f'## Response\n\n')
            file.write(f'{data["response"]}\n\n')

            if data.get('documents'):
                file.write(f'## Referenced Documents\n\n')
                for i, doc in enumerate(data['documents']):
                    file.write(f'### Document {i+1}: {doc["metadata"]["source"]}\n\n')
                    file.write(f'```\n{doc["content"][:500]}{"..." if len(doc["content"]) > 500 else ""}\n```\n\n')

    def _save_text(self, filepath, data):
        """Save query-response as plain text"""
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(f'=== Query Log ===\n\n')
            file.write(f'Timestamp: {data["timestamp"]}\n\n')
            file.write(f'--- Query ---\n\n')
            file.write(f'{data["query"]}\n\n')
            file.write(f'--- Response ---\n\n')
            file.write(f'{data["response"]}\n\n')

            if data.get('documents'):
                file.write(f'--- Referenced Documents ---\n\n')
                for i, doc in enumerate(data['documents']):
                    file.write(f'Document {i+1}: {doc["metadata"]["source"]}\n\n')
                    file.write(f'{doc["content"][:500]}{"..." if len(doc["content"]) > 500 else ""}\n\n')

    def print_formatted(self, output_type, content, **kwargs):
        """
        Print formatted output to console based on output type

        Args:
            output_type (str): Type of output ('debug', 'header', 'subheader', 'command',
                              'feedback', 'analysis', 'list', 'kv')
            content (str or dict): Content to display
            **kwargs: Additional formatting parameters
        """
        from core.engine.utils import (
            format_debug, format_header, format_subheader, format_mini_header,
            format_command, format_feedback, format_key_value, format_list_item,
            format_analysis_result, format_code_block
        )

        # Default formatted output to the original content
        formatted = str(content)

        # Format based on output type
        if output_type == 'debug':
            formatted = format_debug(content)
        elif output_type == 'header':
            formatted = format_header(content)
        elif output_type == 'subheader':
            formatted = format_subheader(content)
        elif output_type == 'mini_header':
            formatted = format_mini_header(content)
        elif output_type == 'command':
            color = kwargs.get('color', None)
            formatted = format_command(content)
        elif output_type == 'feedback':
            success = kwargs.get('success', True)
            formatted = format_feedback(content, success)
        elif output_type == 'analysis':
            title = kwargs.get('title', 'Result')
            formatted = format_analysis_result(title, content)
        elif output_type == 'list':
            indent = kwargs.get('indent', 0)
            formatted = format_list_item(content, indent)
        elif output_type == 'kv':
            key = kwargs.get('key', '')
            indent = kwargs.get('indent', 0)
            formatted = format_key_value(key, content, indent)
        elif output_type == 'code':
            indent = kwargs.get('indent', 0)
            formatted = format_code_block(content, indent)

        # Print the formatted output
        print(formatted)

        # Add to session file if active
        if self.session_file:
            self._write_to_session(formatted)

    def _write_to_session(self, text):
        """
        Write text to the active session file

        Args:
            text (str): Text to write to session
        """
        if not self.session_file:
            return

        try:
            # Write text based on the output format
            if self.session_file['format'] == 'json':
                # For JSON, we need to properly handle the commas and formatting
                if self.session_file.get('first', True):
                    self.session_file['file'].write('    {\n')
                    self.session_file['first'] = False
                else:
                    self.session_file['file'].write(',\n    {\n')

                # Escape any quotes in the text for JSON compatibility
                escaped_text = text.replace('"', '\\"').replace('\n', '\\n')

                # Write the content as a JSON object
                self.session_file['file'].write(f'      "timestamp": "{datetime.datetime.now().isoformat()}",\n')
                self.session_file['file'].write(f'      "content": "{escaped_text}"\n')
                self.session_file['file'].write('    }')
            else:
                # For text and markdown, just write the text with a timestamp
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.session_file['file'].write(f"[{timestamp}] {text}\n")

            # Make sure content is flushed to disk
            self.session_file['file'].flush()

        except Exception as e:
            # Log any errors but don't crash the application
            print(f"Error writing to session file: {str(e)}")
            print(f"Error writing to session file: {str(e)}")