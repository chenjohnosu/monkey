"""
Output formatting and saving
"""

import os
import json
import datetime
from core.engine.utils import timestamp_filename, ensure_dir
from core.engine.logging import debug

class OutputManager:
    def __init__(self, config):
        self.config = config
        self.buffer = None
        self.session_file = None
        self.batch_mode = config.get('system.hpc_mode', False)
        self.hpc_mode = config.get('system.hpc_mode', False)
        ensure_dir('logs')
        debug(config, "Output manager initialized")

    def add_to_buffer(self, query, response, docs=None):
        debug(self.config, "Adding to output buffer")

        self.buffer = {
            'timestamp': datetime.datetime.now().isoformat(),
            'query': query,
            'response': response,
            'documents': docs if docs else []
        }

    def save_buffer(self, workspace):
        debug(self.config, "Saving output buffer")

        if not self.buffer:
            return None

        # Create logs directory if it doesn't exist
        logs_dir = os.path.join('logs', workspace)
        ensure_dir(logs_dir)

        # Get output format
        output_format = self.config.get('system.output_format', 'txt')

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
        debug(self.config, "Starting session saving")

        # Create logs directory if it doesn't exist
        logs_dir = os.path.join('logs', workspace)
        ensure_dir(logs_dir)

        # Get output format
        output_format = self.config.get('system.output_format', 'txt')

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
            self.session_file['file'].write('  "batch_mode": ' + str(self.batch_mode).lower() + ',\n')
            self.session_file['file'].write('  "interactions": [\n')
            self.session_file['first'] = True
        elif output_format == 'md':
            self.session_file['file'].write(f'# Session Log - {workspace}\n\n')
            self.session_file['file'].write(f'Session started: {datetime.datetime.now().isoformat()}\n')
            if self.batch_mode:
                self.session_file['file'].write('Running in batch mode\n')
            self.session_file['file'].write('\n')
        else:  # Default to txt
            self.session_file['file'].write(f'=== Session Log - {workspace} ===\n\n')
            self.session_file['file'].write(f'Session started: {datetime.datetime.now().isoformat()}\n')
            if self.batch_mode:
                self.session_file['file'].write('Running in batch mode\n')
            self.session_file['file'].write('\n')

    def stop_session_saving(self):
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
        with open(filepath, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=2)

    def _save_markdown(self, filepath, data):
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
        debug(self.config, f"Formatted output type: {output_type}")

        # Format based on output type
        if output_type == 'debug':
            formatted = f"[DEBUG] {content}"
        elif output_type == 'header':
            formatted = f"\n{content}\n{'=' * len(content)}"
        elif output_type == 'subheader':
            formatted = f"\n{content}\n{'-' * len(content)}"
        elif output_type == 'mini_header':
            formatted = f"\n{content}:"
        elif output_type == 'command':
            formatted = f"> {content}"
        elif output_type == 'feedback':
            success = kwargs.get('success', True)
            prefix = "✓" if success else "✗"
            formatted = f"{prefix} {content}"
        elif output_type == 'analysis':
            title = kwargs.get('title', 'Result')
            formatted = f"{title}: {content}"
        elif output_type == 'list':
            indent = kwargs.get('indent', 0)
            indent_str = " " * indent
            formatted = f"{indent_str}• {content}"
        elif output_type == 'kv':
            key = kwargs.get('key', '')
            indent = kwargs.get('indent', 0)
            indent_str = " " * indent
            formatted = f"{indent_str}{key}: {content}"
        elif output_type == 'code':
            indent = kwargs.get('indent', 0)
            indent_str = " " * indent
            lines = content.split('\n')
            formatted_lines = [f"{indent_str}{line}" for line in lines]
            formatted = '\n'.join(formatted_lines)
        else:
            formatted = str(content)

        # Print the formatted output
        print(formatted)

        # Add to session file if active
        if self.session_file:
            self._write_to_session(formatted)

    def _write_to_session(self, text):
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
            # Log any errors but don't crash
            debug(self.config, f"Error writing to session file: {str(e)}")

    def save_theme_analysis(self, workspace, results, method, output_format):
        debug(self.config, "Saving theme analysis results")

        # Create logs directory
        logs_dir = os.path.join('logs', workspace)
        ensure_dir(logs_dir)

        # Generate filename with timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"themes_{method}_{timestamp}.{output_format}"
        filepath = os.path.join(logs_dir, filename)

        # Save based on format
        if output_format == 'json':
            self._save_json(filepath, {
                'timestamp': datetime.datetime.now().isoformat(),
                'method': method,
                'results': results
            })
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"=== THEME ANALYSIS RESULTS ===\n\n")
                f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
                f.write(f"Method: {method}\n\n")

                for result_type, result in results.items():
                    f.write(f"--- {result.get('method', result_type.upper())} ---\n\n")

                    # Show themes
                    themes = result.get('themes', [])
                    if themes:
                        f.write(f"Found {len(themes)} themes:\n\n")

                        for i, theme in enumerate(themes):
                            f.write(f"Theme {i+1}: {theme.get('name', 'Unnamed')}\n")

                            if 'keywords' in theme:
                                f.write(f"  Keywords: {', '.join(theme['keywords'])}\n")

                            if 'score' in theme:
                                f.write(f"  Score: {theme['score']}\n")

                            if 'document_count' in theme:
                                f.write(f"  Documents: {theme['document_count']}\n")

                            f.write("\n")
                    else:
                        f.write("No themes found\n\n")

        return filepath

    def save_topic_analysis(self, workspace, results, method, output_format):
        debug(self.config, "Saving topic analysis results")

        # Create logs directory
        logs_dir = os.path.join('logs', workspace)
        ensure_dir(logs_dir)

        # Generate filename with timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"topics_{method}_{timestamp}.{output_format}"
        filepath = os.path.join(logs_dir, filename)

        # Save based on format
        if output_format == 'json':
            self._save_json(filepath, {
                'timestamp': datetime.datetime.now().isoformat(),
                'method': method,
                'results': results
            })
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"=== TOPIC ANALYSIS RESULTS ===\n\n")
                f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
                f.write(f"Method: {method}\n\n")

                for result_type, result in results.items():
                    f.write(f"--- {result.get('method', result_type.upper())} ---\n\n")

                    # Show topics
                    topics = result.get('topics', [])
                    if topics:
                        f.write(f"Found {len(topics)} topics:\n\n")

                        for i, topic in enumerate(topics):
                            f.write(f"Topic {i+1}: {topic.get('name', 'Unnamed')}\n")

                            if 'keywords' in topic:
                                f.write(f"  Keywords: {', '.join(topic['keywords'])}\n")

                            if 'document_count' in topic:
                                f.write(f"  Documents: {topic['document_count']}\n")

                            f.write("\n")
                    else:
                        f.write("No topics found\n\n")

        return filepath

    def save_sentiment_analysis(self, workspace, results, method, output_format):
        debug(self.config, "Saving sentiment analysis results")

        # Create logs directory
        logs_dir = os.path.join('logs', workspace)
        ensure_dir(logs_dir)

        # Generate filename with timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"sentiment_{method}_{timestamp}.{output_format}"
        filepath = os.path.join(logs_dir, filename)

        # Save based on format
        if output_format == 'json':
            self._save_json(filepath, {
                'timestamp': datetime.datetime.now().isoformat(),
                'method': method,
                'results': results
            })
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"=== SENTIMENT ANALYSIS RESULTS ===\n\n")
                f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
                f.write(f"Method: {method}\n\n")

                for result_type, result in results.items():
                    f.write(f"--- {result.get('method', result_type.upper())} ---\n\n")

                    # Show sentiment distribution
                    if 'sentiment_distribution' in result:
                        f.write(f"Sentiment Distribution:\n")
                        for sentiment, count in result['sentiment_distribution'].items():
                            f.write(f"  {sentiment}: {count}\n")
                        f.write("\n")

                    # Show document sentiments
                    document_sentiments = result.get('document_sentiments', [])
                    if document_sentiments:
                        f.write(f"Document Sentiments:\n\n")

                        for i, doc in enumerate(document_sentiments[:10]):  # First 10 only
                            f.write(f"  {doc['source']}: {doc['sentiment']} (score: {doc['score']:.2f})\n")

                        if len(document_sentiments) > 10:
                            f.write(f"  ... and {len(document_sentiments) - 10} more\n")

                        f.write("\n")

        return filepath