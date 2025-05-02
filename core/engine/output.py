import os
import sys
import json
import datetime
from core.engine.utils import timestamp_filename, ensure_dir
from core.engine.logging import debug

class OutputManager:
    def __init__(self, config):
        self.config = config
        self.buffer = None
        self.session_file = None
        self.batch_mode = config.get('system.batch_mode', False)
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
                # For text, just write the text with a timestamp
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.session_file['file'].write(f"[{timestamp}] {text}\n")

            # Make sure content is flushed to disk
            self.session_file['file'].flush()

        except Exception as e:
            # Log any errors but don't crash
            debug(self.config, f"Error writing to session file: {str(e)}")

    def print(self, text, newline=True):
        """Simple print function that respects batch mode"""
        if newline:
            print(text)
        else:
            print(text, end='')
            
        # Add to session log if active
        if self.session_file:
            self._write_to_session(text)
            
        # Ensure output is flushed in HPC/batch mode
        if self.batch_mode or self.hpc_mode:
            sys.stdout.flush()

    def save_analysis_results(self, analysis_type, workspace, results):
        """Save analysis results to file"""
        debug(self.config, f"Saving {analysis_type} analysis results")

        # Create logs directory
        logs_dir = os.path.join('logs', workspace)
        ensure_dir(logs_dir)

        # Get output format
        output_format = self.config.get('system.output_format', 'txt')

        # Generate filename with timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{analysis_type}_{timestamp}.{output_format}"
        filepath = os.path.join(logs_dir, filename)

        # Save based on format
        if output_format == 'json':
            self._save_json(filepath, {
                'timestamp': datetime.datetime.now().isoformat(),
                'type': analysis_type,
                'results': results
            })
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"=== {analysis_type.upper()} ANALYSIS RESULTS ===\n\n")
                f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n\n")
                
                # Format results by type
                if analysis_type == 'themes':
                    self._write_theme_results(f, results)
                elif analysis_type == 'topics':
                    self._write_topic_results(f, results)
                elif analysis_type == 'sentiment':
                    self._write_sentiment_results(f, results)
                else:
                    # Generic format for other types
                    f.write(str(results))

        return filepath

    def _write_theme_results(self, file, results):
        """Write theme analysis results to text file"""
        for method, result in results.items():
            file.write(f"--- {result.get('method', method.upper())} ---\n\n")
            
            themes = result.get('themes', [])
            if themes:
                file.write(f"Found {len(themes)} themes:\n\n")
                
                for i, theme in enumerate(themes):
                    file.write(f"Theme {i+1}: {theme.get('name', 'Unnamed')}\n")
                    
                    if 'keywords' in theme:
                        file.write(f"  Keywords: {', '.join(theme['keywords'])}\n")
                        
                    if 'score' in theme:
                        file.write(f"  Score: {theme['score']}\n")
                        
                    if 'document_count' in theme:
                        file.write(f"  Documents: {theme['document_count']}\n")
                        
                    file.write("\n")
            else:
                file.write("No themes found\n\n")
                
    def _write_topic_results(self, file, results):
        """Write topic analysis results to text file"""
        for method, result in results.items():
            file.write(f"--- {result.get('method', method.upper())} ---\n\n")
            
            topics = result.get('topics', [])
            if topics:
                file.write(f"Found {len(topics)} topics:\n\n")
                
                for i, topic in enumerate(topics):
                    file.write(f"Topic {i+1}: {topic.get('name', 'Unnamed')}\n")
                    
                    if 'keywords' in topic:
                        file.write(f"  Keywords: {', '.join(topic['keywords'])}\n")
                        
                    if 'document_count' in topic:
                        file.write(f"  Documents: {topic['document_count']}\n")
                        
                    file.write("\n")
            else:
                file.write("No topics found\n\n")
                
    def _write_sentiment_results(self, file, results):
        """Write sentiment analysis results to text file"""
        for method, result in results.items():
            file.write(f"--- {result.get('method', method.upper())} ---\n\n")
            
            # Show sentiment distribution
            if 'sentiment_distribution' in result:
                file.write(f"Sentiment Distribution:\n")
                for sentiment, count in result['sentiment_distribution'].items():
                    file.write(f"  {sentiment}: {count}\n")
                file.write("\n")
                
            # Show document sentiments
            document_sentiments = result.get('document_sentiments', [])
            if document_sentiments:
                file.write(f"Document Sentiments:\n\n")
                
                for i, doc in enumerate(document_sentiments[:10]):  # First 10 only
                    file.write(f"  {doc['source']}: {doc['sentiment']} (score: {doc['score']:.2f})\n")
                    
                if len(document_sentiments) > 10:
                    file.write(f"  ... and {len(document_sentiments) - 10} more\n")
                    
                file.write("\n")