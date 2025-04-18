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
        debug(self.config, "Saving theme analysis results")
        
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

        # Default formatted output to the original content
        formatted = str(content)

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

        # Print the formatted output
        print(formatted)

        # Add to buffer if saving session
        if hasattr(self, 'session_file') and self.session_file:
            # Use the already-formatted text (which now includes all necessary formatting)
            # For session file, store without colors
            from re import sub
            plain_text = sub(r'\033\[\d+(;\d+)?m', '', formatted) if '\033[' in formatted else formatted

            # Call the _write_to_session method
            if hasattr(self, '_write_to_session'):
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

    def save_sentiment_analysis(self, workspace, results, method, output_format=None):
        """
        Save sentiment analysis results to a file

        Args:
            workspace (str): Current workspace
            results (dict): Analysis results
            method (str): Analysis method used
            output_format (str, optional): Output format override

        Returns:
            str: Path to the saved file
        """
        debug(self.config, "Saving sentiment analysis results")

        # Create logs directory if it doesn't exist
        logs_dir = os.path.join('logs', workspace)
        ensure_dir(logs_dir)

        # Get output format
        if not output_format:
            output_format = self.config.get('system.output_format')

        # Generate filename
        filename = timestamp_filename(f'sentiment_{method}', output_format)
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
            self._save_markdown_sentiment(filepath, data)
        else:  # Default to txt
            self._save_text_sentiment(filepath, data)

        return filepath

    def save_topic_analysis(self, workspace, results, method, output_format=None):
        """
        Save topic modeling results to a file

        Args:
            workspace (str): Current workspace
            results (dict): Analysis results
            method (str): Analysis method used
            output_format (str, optional): Output format override

        Returns:
            str: Path to the saved file
        """
        debug(self.config, "Saving topic modeling results")

        # Create logs directory if it doesn't exist
        logs_dir = os.path.join('logs', workspace)
        ensure_dir(logs_dir)

        # Get output format
        if not output_format:
            output_format = self.config.get('system.output_format')

        # Generate filename
        filename = timestamp_filename(f'topic_{method}', output_format)
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
            self._save_markdown_topic(filepath, data)
        else:  # Default to txt
            self._save_text_topic(filepath, data)

        return filepath

    def _save_markdown_sentiment(self, filepath, data):
        """Save sentiment analysis as Markdown"""
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(f'# Sentiment Analysis - {data["workspace"]}\n\n')
            file.write(f'Timestamp: {data["timestamp"]}\n')
            file.write(f'Method: {data["method"]}\n\n')

            for method_key, result in data['results'].items():
                file.write(f'## {result["method"]}\n\n')

                # Write sentiment distribution if available
                if "sentiment_distribution" in result:
                    file.write('### Sentiment Distribution\n\n')
                    file.write('| Sentiment | Count |\n')
                    file.write('|-----------|-------|\n')
                    for sentiment, count in result["sentiment_distribution"].items():
                        file.write(f'| {sentiment.capitalize()} | {count} |\n')
                    file.write('\n')

                # Write overall distribution if available
                if "overall_distribution" in result:
                    file.write('### Overall Distribution (%)\n\n')
                    file.write('| Sentiment | Percentage |\n')
                    file.write('|-----------|------------|\n')
                    for sentiment, percentage in result["overall_distribution"].items():
                        file.write(f'| {sentiment.capitalize()} | {percentage}% |\n')
                    file.write('\n')

                # Write sentiment trends if available
                if "sentiment_trends" in result:
                    file.write('### Sentiment Trends\n\n')
                    for trend in result["sentiment_trends"]:
                        file.write(f'#### Aspect: {trend["aspect"]}\n\n')
                        file.write(f'- Dominant Sentiment: {trend["dominant_sentiment"]}\n')
                        file.write(f'- Document Count: {trend["document_count"]}\n')
                        file.write('- Distribution:\n')
                        for sentiment, percentage in trend["sentiment_distribution"].items():
                            if percentage > 0:
                                file.write(f'  - {sentiment.capitalize()}: {percentage:.1f}%\n')
                        file.write('\n')

                # Write document analysis if available
                if "document_analysis" in result:
                    file.write('### Document Analysis\n\n')
                    for doc in result["document_analysis"]:
                        file.write(f'#### Document: {doc["source"]}\n\n')
                        file.write(f'- Overall Sentiment: {doc["overall_sentiment"]}\n')
                        file.write(f'- Emotional Tone: {doc["emotional_tone"]}\n')

                        if doc.get("aspects"):
                            file.write('- Key Aspects:\n')
                            for aspect in doc["aspects"]:
                                file.write(
                                    f'  - {aspect["aspect"]}: {aspect["sentiment"]} (confidence: {aspect["confidence"]:.2f})\n')

                        if doc.get("summary"):
                            file.write(f'- Summary: {doc["summary"]}\n')

                        file.write('\n')

    def _save_text_sentiment(self, filepath, data):
        """Save sentiment analysis as plain text"""
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(f'=== Sentiment Analysis - {data["workspace"]} ===\n\n')
            file.write(f'Timestamp: {data["timestamp"]}\n')
            file.write(f'Method: {data["method"]}\n\n')

            for method_key, result in data['results'].items():
                file.write(f'--- {result["method"]} ---\n\n')

                # Write sentiment distribution if available
                if "sentiment_distribution" in result:
                    file.write('Sentiment Distribution:\n')
                    for sentiment, count in result["sentiment_distribution"].items():
                        file.write(f'  {sentiment.capitalize()}: {count}\n')
                    file.write('\n')

                # Write overall distribution if available
                if "overall_distribution" in result:
                    file.write('Overall Distribution (%):\n')
                    for sentiment, percentage in result["overall_distribution"].items():
                        file.write(f'  {sentiment.capitalize()}: {percentage}%\n')
                    file.write('\n')

                # Write sentiment trends if available
                if "sentiment_trends" in result:
                    file.write('Sentiment Trends:\n')
                    for trend in result["sentiment_trends"]:
                        file.write(f'  Aspect: {trend["aspect"]}\n')
                        file.write(f'    Dominant Sentiment: {trend["dominant_sentiment"]}\n')
                        file.write(f'    Document Count: {trend["document_count"]}\n')
                        file.write('    Distribution:\n')
                        for sentiment, percentage in trend["sentiment_distribution"].items():
                            if percentage > 0:
                                file.write(f'      {sentiment.capitalize()}: {percentage:.1f}%\n')
                        file.write('\n')

                # Write document analysis if available
                if "document_analysis" in result:
                    file.write('Document Analysis:\n')
                    for doc in result["document_analysis"]:
                        file.write(f'  Document: {doc["source"]}\n')
                        file.write(f'    Overall Sentiment: {doc["overall_sentiment"]}\n')
                        file.write(f'    Emotional Tone: {doc["emotional_tone"]}\n')

                        if doc.get("aspects"):
                            file.write('    Key Aspects:\n')
                            for aspect in doc["aspects"]:
                                file.write(
                                    f'      {aspect["aspect"]}: {aspect["sentiment"]} (confidence: {aspect["confidence"]:.2f})\n')

                        if doc.get("summary"):
                            file.write(f'    Summary: {doc["summary"]}\n')

                        file.write('\n')

    def _save_markdown_topic(self, filepath, data):
        """Save topic modeling results as Markdown"""
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(f'# Topic Modeling - {data["workspace"]}\n\n')
            file.write(f'Timestamp: {data["timestamp"]}\n')
            file.write(f'Method: {data["method"]}\n\n')

            for method_key, result in data['results'].items():
                file.write(f'## {result["method"]}\n\n')

                # Show language if specified
                if "language" in result:
                    file.write(f'Language: {result["language"]}\n\n')

                # Show any errors
                if "error" in result:
                    file.write(f'**Error**: {result["error"]}\n\n')
                    continue

                # Display topics
                file.write(f'Found {len(result["topics"])} topics:\n\n')

                for topic in result['topics']:
                    file.write(f'### {topic["name"]}\n\n')

                    if "keywords" in topic:
                        file.write(f'**Keywords**: {", ".join(topic["keywords"])}\n\n')

                    if "document_count" in topic:
                        file.write(f'**Documents**: {topic["document_count"]}\n\n')
                    elif "count" in topic:
                        file.write(f'**Count**: {topic["count"]}\n\n')

                    if "score" in topic:
                        file.write(f'**Score**: {topic["score"]:.2f}\n\n')

                    # Display top documents for this topic
                    if "document_distribution" in topic:
                        doc_dist = topic["document_distribution"]
                        if doc_dist:
                            file.write('**Top documents**:\n\n')
                            for doc in sorted(doc_dist, key=lambda x: x.get("probability", 0), reverse=True)[:5]:
                                prob_str = f" ({doc['probability']:.2f})" if "probability" in doc else ""
                                file.write(f'- {doc["source"]}{prob_str}\n')

                            if len(doc_dist) > 5:
                                file.write(f'- ... and {len(doc_dist) - 5} more\n')

                            file.write('\n')

                    # Display topic summary if available
                    if "summary" in topic and topic["summary"]:
                        file.write(f'**Summary**: {topic["summary"]}\n\n')

    def _save_text_topic(self, filepath, data):
        """Save topic modeling results as plain text"""
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(f'=== Topic Modeling - {data.get("workspace", "Unknown Workspace")} ===\n\n')
            file.write(f'Timestamp: {data.get("timestamp", "Unknown")}\n')
            file.write(f'Method: {data.get("method", "Unknown")}\n\n')

            # Handle case where results might be empty or malformed
            results = data.get('results', {})
            if not results:
                file.write('No topic modeling results found.\n')
                return

            # Track total topics and total errors
            total_topics = 0
            total_errors = 0

            for key, result in results.items():
                # Convert to dictionary if not already
                if not isinstance(result, dict):
                    result = {'error': str(result)}

                # Handle cases where result might be a string or have unexpected structure
                if not result:
                    total_errors += 1
                    file.write(f'--- Result {key}: Empty Result ---\n\n')
                    continue

                # Safely get method, defaulting to key if not present
                method = result.get('method', key)
                file.write(f'--- {method} ---\n\n')

                # Show language if specified
                language = result.get('language', 'Unknown')
                file.write(f'Language: {language}\n\n')

                # Show any errors
                if "error" in result:
                    total_errors += 1
                    file.write(f'Error: {result["error"]}\n\n')
                    continue

                # Get topics, defaulting to empty list
                topics = result.get('topics', [])
                file.write(f'Found {len(topics)} topics:\n\n')
                total_topics += len(topics)

                # Handle case of no topics
                if not topics:
                    file.write('  No specific topics found.\n\n')
                    continue

                for topic_index, topic in enumerate(topics, 1):
                    # Convert to dictionary if not already
                    if not isinstance(topic, dict):
                        topic = {'name': f'Topic {topic_index}', 'description': str(topic)}

                    # Safely get topic name or generate one
                    topic_name = topic.get('name', f'Topic {topic_index}')
                    file.write(f'{topic_name}\n')

                    # Keywords (max 10)
                    keywords = topic.get('keywords', [])
                    if keywords:
                        file.write(f'  Keywords: {", ".join(str(k) for k in keywords[:10])}\n')
                        if len(keywords) > 10:
                            file.write(f'    ... and {len(keywords) - 10} more keywords\n')

                    # Coherence score if available
                    coherence = topic.get('coherence')
                    if coherence is not None:
                        file.write(f'  Coherence: {float(coherence):.2f}\n')

                    # Document count and type
                    doc_count = (topic.get('document_count') or
                                 topic.get('count') or
                                 len(topic.get('documents', [])) or
                                 len(topic.get('document_distribution', [])) or
                                 0)
                    if doc_count is not None:
                        file.write(f'  Documents: {doc_count}\n')

                    # Score
                    score = topic.get('score')
                    if score is not None:
                        file.write(f'  Score: {float(score):.3f}\n')

                    # Top documents
                    documents = topic.get('documents', [])
                    top_docs = topic.get('document_distribution', [])

                    # Prioritize document distribution if available
                    if top_docs:
                        file.write('  Top Documents:\n')
                        sorted_docs = sorted(top_docs, key=lambda x: x.get("probability", 0), reverse=True)[:5]
                        for doc in sorted_docs:
                            prob_str = f" ({doc.get('probability', 1):.2f})" if 'probability' in doc else ""
                            file.write(f'    - {doc.get("source", "Unknown")}{prob_str}\n')
                        if len(top_docs) > 5:
                            file.write(f'    - ... and {len(top_docs) - 5} more\n')
                    elif documents:
                        file.write('  Top Documents:\n')
                        for doc in documents[:5]:
                            file.write(f'    - {doc}\n')
                        if len(documents) > 5:
                            file.write(f'    - ... and {len(documents) - 5} more\n')

                    # Summary
                    summary = topic.get('summary') or topic.get('description')
                    if summary:
                        file.write(f'  Summary: {summary}\n')

                    file.write('\n')

            # Footer with summary statistics
            file.write('=== Analysis Summary ===\n')
            file.write(f'Total Topics: {total_topics}\n')
            file.write(f'Total Errors: {total_errors}\n')

    def _write_to_session(self, text):
        """
        Write text to the active session file

        Args:
            text (str): Text to write to session
        """
        import datetime  # Ensure datetime is imported

        if not hasattr(self, 'session_file') or not self.session_file:
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
            import traceback
            print(traceback.format_exc())