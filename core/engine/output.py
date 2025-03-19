"""
Output formatting and saving
"""

import os
import json
import datetime
from core.engine.utils import debug_print, timestamp_filename, ensure_dir

class OutputManager:
    """Manages output formatting and saving"""
    
    def __init__(self, config):
        """Initialize the output manager"""
        self.config = config
        self.buffer = None
        self.session_file = None
        ensure_dir('logs')
        debug_print(config, "Output manager initialized")
    
    def add_to_buffer(self, query, response, docs=None):
        """
        Add a query-response pair to the buffer
        
        Args:
            query (str): User query
            response (str): System response
            docs (list, optional): Retrieved documents
        """
        debug_print(self.config, "Adding to output buffer")
        
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
        debug_print(self.config, "Saving output buffer")
        
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
    
    def save_theme_analysis(self, workspace, results, method, output_format=None):
        """
        Save theme analysis results to a file
        
        Args:
            workspace (str): Current workspace
            results (dict): Analysis results
            method (str): Analysis method used
            output_format (str, optional): Output format override
            
        Returns:
            str: Path to the saved file
        """
        debug_print(self.config, "Saving theme analysis results")
        
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join('logs', workspace)
        ensure_dir(logs_dir)
        
        # Get output format
        if not output_format:
            output_format = self.config.get('system.output_format')
        
        # Generate filename
        filename = timestamp_filename(f'themes_{method}', output_format)
        filepath = os.path.join(logs_dir, filename)
        
        # Create data structure
        data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'workspace': workspace,
            'method': method,
            'results': results
        }
        
        # Save file based on format
        if output_format == 'json':
            self._save_json(filepath, data)
        elif output_format == 'md':
            self._save_markdown_themes(filepath, data)
        else:  # Default to txt
            self._save_text_themes(filepath, data)
        
        return filepath
    
    def start_session_saving(self, workspace):
        """
        Start saving session output to a file
        
        Args:
            workspace (str): Current workspace
        """
        debug_print(self.config, "Starting session saving")
        
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
        debug_print(self.config, "Stopping session saving")
        
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
        # Determine output format
        output_format = kwargs.get('format', self.config.get('system.output_format'))

        # Import utility functions
        from core.engine.utils import (
            format_debug, format_header, format_subheader, format_mini_header,
            format_command, format_feedback, format_key_value, format_list_item,
            format_analysis_result, format_code_block,
            format_md_header, format_md_subheader, format_md_mini_header,
            format_md_command, format_md_feedback, format_md_analysis
        )

        # Format based on output type and format
        if output_format == 'md':
            # Markdown formatting
            if output_type == 'debug':
                formatted = format_debug(content)
            elif output_type == 'header':
                formatted = format_md_header(content)
            elif output_type == 'subheader':
                formatted = format_md_subheader(content)
            elif output_type == 'mini_header':
                formatted = format_md_mini_header(content)
            elif output_type == 'command':
                formatted = format_md_command(content)
            elif output_type == 'feedback':
                success = kwargs.get('success', True)
                formatted = format_md_feedback(content, success)
            elif output_type == 'analysis':
                items = kwargs.get('items', [])
                formatted = format_md_analysis(content, items)
            elif output_type == 'list':
                indent = kwargs.get('indent', 0)
                formatted = f"{'  ' * indent}- {content}"
            elif output_type == 'kv':
                key = kwargs.get('key', '')
                formatted = f"**{key}**: {content}"
            elif output_type == 'code':
                formatted = f"```\n{content}\n```"
            else:
                formatted = content
        else:
            # Plain text formatting with colors
            if output_type == 'debug':
                formatted = format_debug(content)
            elif output_type == 'header':
                formatted = format_header(content)
            elif output_type == 'subheader':
                formatted = format_subheader(content)
            elif output_type == 'mini_header':
                formatted = format_mini_header(content)
            elif output_type == 'command':
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
            else:
                formatted = content

        # Print the formatted output
        print(formatted)

        # Add to buffer if saving session
        if self.session_file:
            # For session file, store without colors
            from re import sub
            plain_text = sub(r'\033\[\d+(;\d+)?m', '', formatted) if '\033[' in formatted else formatted
            self._write_to_session(plain_text)

    def _save_markdown_themes(self, filepath, data):
        """Enhanced save theme analysis as Markdown"""
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(f'# Theme Analysis - {data["workspace"]}\n\n')
            file.write(f'**Timestamp:** {data["timestamp"]}\n')
            file.write(f'**Method:** {data["method"]}\n\n')

            for method_key, result in data['results'].items():
                file.write(f'## {result["method"]}\n\n')

                if 'entity_count' in result:
                    file.write(f'**Total entities:** {result["entity_count"]}\n')
                if 'variance_explained' in result:
                    file.write(f'**Variance explained:** {result["variance_explained"]}%\n')
                if 'clusters' in result:
                    file.write(f'**Number of clusters:** {result["clusters"]}\n')
                if 'error' in result:
                    file.write(f'**Error:** {result["error"]}\n')

                file.write('\n')

                for theme in result['themes']:
                    if 'name' in theme:
                        file.write(f'### {theme["name"]}\n\n')
                        if 'frequency' in theme:
                            file.write(f'**Frequency:** {theme["frequency"]}\n\n')
                        if 'keywords' in theme:
                            file.write(f'**Keywords:** {", ".join(theme["keywords"])}\n\n')
                        if 'score' in theme:
                            file.write(f'**Score:** {theme["score"]}\n\n')
                        if 'centrality' in theme:
                            file.write(f'**Centrality:** {theme["centrality"]}\n\n')
                        if 'document_count' in theme:
                            file.write(f'**Documents:** {theme["document_count"]}\n\n')
                        if 'documents' in theme and isinstance(theme['documents'], list):
                            file.write('**Document files:**\n\n')
                            for doc in theme['documents']:
                                if isinstance(doc, str):
                                    file.write(f'- {doc}\n')
                                elif isinstance(doc, dict) and 'source' in doc:
                                    file.write(f'- {doc["source"]}\n')
                            file.write('\n')
                        if 'description' in theme and theme['description']:
                            file.write('**Summary:**\n\n')
                            file.write(f'```\n{theme["description"]}\n```\n\n')
                    elif 'keyword' in theme:
                        file.write(f'### Keyword: {theme["keyword"]}\n\n')
                        file.write(f'**Score:** {theme["score"]}\n')
                        file.write(f'**Documents:** {theme["documents"]}\n\n')

    def _save_text_themes(self, filepath, data):
        """Enhanced save theme analysis as plain text"""
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(f"THEME ANALYSIS - {data['workspace']}\n")
            file.write(f"Timestamp: {data['timestamp']}\n")
            file.write(f"Method: {data['method']}\n\n")

            for method_key, result in data['results'].items():
                file.write(f"{result['method']}\n")

                if 'entity_count' in result:
                    file.write(f"Total entities: {result['entity_count']}\n")
                if 'variance_explained' in result:
                    file.write(f"Variance explained: {result['variance_explained']}%\n")
                if 'clusters' in result:
                    file.write(f"Number of clusters: {result['clusters']}\n")
                if 'error' in result:
                    file.write(f"Error: {result['error']}\n")

                file.write("\n")

                for theme in result['themes']:
                    if 'name' in theme:
                        file.write(f">> {theme['name']} <<\n")
                        if 'frequency' in theme:
                            file.write(f"Frequency: {theme['frequency']}\n")
                        if 'keywords' in theme:
                            file.write(f"Keywords: {', '.join(theme['keywords'])}\n")
                        if 'score' in theme:
                            file.write(f"Score: {theme['score']}\n")
                        if 'centrality' in theme:
                            file.write(f"Centrality: {theme['centrality']}\n")
                        if 'document_count' in theme:
                            file.write(f"Documents: {theme['document_count']}\n")
                        if 'documents' in theme and isinstance(theme['documents'], list):
                            file.write("Document files:\n")
                            for doc in theme['documents']:
                                if isinstance(doc, str):
                                    file.write(f"  • {doc}\n")
                                elif isinstance(doc, dict) and 'source' in doc:
                                    file.write(f"  • {doc['source']}\n")
                            file.write("\n")
                        if 'description' in theme and theme['description']:
                            file.write("Summary:\n")
                            file.write(f"{theme['description']}\n\n")
                    elif 'keyword' in theme:
                        file.write(f"Keyword: {theme['keyword']}\n")
                        file.write(f"Score: {theme['score']}\n")
                        file.write(f"Documents: {theme['documents']}\n\n")