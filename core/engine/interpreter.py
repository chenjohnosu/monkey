"""
Analysis interpreter using LLMs to explain results
"""

import os
import datetime
from core.engine.logging import debug

class AnalysisInterpreter:
    """Interprets analysis results using LLMs"""

    def __init__(self, config, storage_manager=None, output_manager=None, text_processor=None):
        """Initialize the interpreter"""
        self.config = config
        self.storage_manager = storage_manager
        self.output_manager = output_manager
        self.text_processor = text_processor

        # Initialize LLM connector
        try:
            from core.connectors.connector_factory import ConnectorFactory
            self.factory = ConnectorFactory(config)
            self.llm_connector = self.factory.get_llm_connector()
            debug(config, "LLM connector initialized for analysis interpretation")
        except Exception as e:
            debug(config, f"Error initializing LLM connector: {str(e)}")
            self.factory = None
            self.llm_connector = None

        debug(config, "Analysis interpreter initialized")

    def interpret_analysis(self, workspace, analysis_type, query=None):
        """
        Interpret analysis results using LLM

        Args:
            workspace (str): Target workspace
            analysis_type (str): Type of analysis ('themes', 'topics', 'sentiment', 'session')
            query (str, optional): Specific question to ask

        Returns:
            str: Interpretation results
        """
        debug(self.config, f"Interpreting {analysis_type} analysis for workspace '{workspace}'")

        # Map analysis_type to the standardized form expected by the helper methods
        # This will ensure compatibility with both singular and plural forms
        analysis_type_map = {
            'themes': 'themes',
            'theme': 'themes',
            'topics': 'topics',
            'topic': 'topics',
            'sentiment': 'sentiment',
            'session': 'session'
        }

        # Normalize the analysis type
        normalized_type = analysis_type_map.get(analysis_type.lower(), analysis_type.lower())

        # Dispatch to the appropriate analysis method
        if normalized_type == 'themes':
            return self._interpret_theme_analysis(workspace, query)
        elif normalized_type == 'topics':
            return self._interpret_topic_analysis(workspace, query)
        elif normalized_type == 'sentiment':
            return self._interpret_sentiment_analysis(workspace, query)
        elif normalized_type == 'session':
            return self._interpret_session_analysis(workspace, query)
        else:
            return f"Interpretation not available for analysis type: {analysis_type}"

    def _interpret_theme_analysis(self, workspace, query=None):
        """
        Interpret theme analysis results

        Args:
            workspace (str): Target workspace
            query (str, optional): Specific question to ask

        Returns:
            str: Interpretation of theme analysis
        """
        debug(self.config, f"Interpreting theme analysis for workspace '{workspace}'")

        # Find the most recent theme analysis results
        results_file = self._find_latest_analysis_file(workspace, 'themes')
        if not results_file:
            return "No theme analysis results found for this workspace."

        # Read the results file
        analysis_content = self._read_analysis_file(results_file)
        if not analysis_content:
            return "Error reading theme analysis results."

        # Generate interpretation with LLM
        return self._generate_interpretation(analysis_content, 'theme analysis', query)

    def _interpret_topic_analysis(self, workspace, query=None):
        """
        Interpret topic modeling results

        Args:
            workspace (str): Target workspace
            query (str, optional): Specific question to ask

        Returns:
            str: Interpretation of topic modeling
        """
        debug(self.config, f"Interpreting topic modeling for workspace '{workspace}'")

        # Find the most recent topic analysis results
        results_file = self._find_latest_analysis_file(workspace, 'topic')
        if not results_file:
            return "No topic modeling results found for this workspace."

        # Read the results file
        analysis_content = self._read_analysis_file(results_file)
        if not analysis_content:
            return "Error reading topic modeling results."

        # Generate interpretation with LLM
        return self._generate_interpretation(analysis_content, 'topic modeling', query)

    def _interpret_sentiment_analysis(self, workspace, query=None):
        """
        Interpret sentiment analysis results

        Args:
            workspace (str): Target workspace
            query (str, optional): Specific question to ask

        Returns:
            str: Interpretation of sentiment analysis
        """
        debug(self.config, f"Interpreting sentiment analysis for workspace '{workspace}'")

        # Find the most recent sentiment analysis results
        results_file = self._find_latest_analysis_file(workspace, 'sentiment')
        if not results_file:
            return "No sentiment analysis results found for this workspace."

        # Read the results file
        analysis_content = self._read_analysis_file(results_file)
        if not analysis_content:
            return "Error reading sentiment analysis results."

        # Generate interpretation with LLM
        return self._generate_interpretation(analysis_content, 'sentiment analysis', query)

    def _interpret_session_analysis(self, workspace, query=None):
        """
        Interpret query session logs

        Args:
            workspace (str): Target workspace
            query (str, optional): Specific question to ask

        Returns:
            str: Interpretation of session logs
        """
        debug(self.config, f"Interpreting session logs for workspace '{workspace}'")

        # Find the most recent session log
        results_file = self._find_latest_analysis_file(workspace, 'session')
        if not results_file:
            return "No query session logs found for this workspace."

        # Read the session log
        session_content = self._read_analysis_file(results_file)
        if not session_content:
            return "Error reading session logs."

        # Generate interpretation with LLM
        return self._generate_interpretation(session_content, 'query session', query)

    def _find_latest_analysis_file(self, workspace, analysis_type):
        """
        Find the most recent analysis results file

        Args:
            workspace (str): Target workspace
            analysis_type (str): Type of analysis

        Returns:
            str: Path to the analysis file, or None if not found
        """
        debug(self.config, f"Finding latest {analysis_type} analysis file for '{workspace}'")

        try:
            # Create logs directory path
            logs_dir = os.path.join('logs', workspace)
            if not os.path.exists(logs_dir):
                debug(self.config, f"Logs directory not found: {logs_dir}")
                return None

            # Pattern matching for different analysis types
            patterns = {
                'themes': ['themes_', 'theme_'],
                'topic': ['topic_'],
                'sentiment': ['sentiment_'],
                'session': ['session_']
            }

            # Get pattern for this analysis type
            type_patterns = patterns.get(analysis_type, [f"{analysis_type}_"])

            # Find all matching files
            matching_files = []
            for filename in os.listdir(logs_dir):
                # Check if the filename matches any of the patterns for this analysis type
                if any(pattern in filename for pattern in type_patterns):
                    filepath = os.path.join(logs_dir, filename)
                    if os.path.isfile(filepath):
                        # Get file modification time
                        mod_time = os.path.getmtime(filepath)
                        matching_files.append((filepath, mod_time))

            # Sort by modification time (newest first)
            matching_files.sort(key=lambda x: x[1], reverse=True)

            # Return the newest file
            if matching_files:
                debug(self.config, f"Found latest file: {matching_files[0][0]}")
                return matching_files[0][0]

            debug(self.config, f"No matching files found for {analysis_type} analysis")
            return None

        except Exception as e:
            debug(self.config, f"Error finding analysis file: {str(e)}")
            return None

    def _read_analysis_file(self, filepath):
        """
        Read content from an analysis file

        Args:
            filepath (str): Path to the analysis file

        Returns:
            str: File content, or None if error
        """
        debug(self.config, f"Reading analysis file: {filepath}")

        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
            return content
        except Exception as e:
            debug(self.config, f"Error reading file: {str(e)}")
            return None

    def _generate_interpretation(self, analysis_content, analysis_type, query=None):
        """
        Generate interpretation using LLM

        Args:
            analysis_content (str): Analysis content to interpret
            analysis_type (str): Type of analysis for context
            query (str, optional): Specific question to ask

        Returns:
            str: Interpretation from LLM
        """
        debug(self.config, f"Generating interpretation for {analysis_type}")

        # Verify LLM connector is available
        if not self.llm_connector:
            return f"LLM connector not available for generating {analysis_type} interpretation."

        try:
            # Limit content length for LLM processing (avoid token limits)
            max_content_length = 6000
            if len(analysis_content) > max_content_length:
                debug(self.config, f"Content too large ({len(analysis_content)} chars), truncating")
                shortened_content = analysis_content[:max_content_length] + "\n...[content truncated]..."
            else:
                shortened_content = analysis_content

            # Create prompt based on whether a specific question was asked
            if query:
                prompt = f"""You are an expert document analysis assistant.
The user has performed {analysis_type} on a document collection and has a specific question:

{query}

Below is the analysis output:

{shortened_content}

Please provide a detailed, insightful answer to the user's question based only on the information in the analysis output."""
            else:
                prompt = f"""You are an expert document analysis assistant.
The user has performed {analysis_type} on a document collection.
Below is the analysis output:

{shortened_content}

Please interpret these results and provide insights about:
1. The main findings and patterns in the analysis
2. What these results reveal about the document collection
3. Additional perspectives or implications that might not be immediately obvious
4. Suggestions for further analysis or investigation

Organize your response in a clear, structured way focusing on the most significant aspects of the analysis."""

            # Generate interpretation with LLM
            debug(self.config, "Sending analysis to LLM for interpretation")
            model = self.config.get('llm.default_model')
            interpretation = self.llm_connector.generate(prompt, model=model, max_tokens=1000)

            return interpretation.strip()

        except Exception as e:
            debug(self.config, f"Error generating interpretation: {str(e)}")
            import traceback
            debug(self.config, traceback.format_exc())
            return f"Error generating interpretation: {str(e)}"