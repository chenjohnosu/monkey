"""
LLM-assisted interpretation for analysis results
"""

import os
import types
import datetime
from core.engine.logging import debug_print


def setup_llm_theme_analysis(command_processor):
    """
    Set up LLM-assisted theme analysis

    Args:
        command_processor: The CLI command processor
    """
    debug_print(command_processor.config, "Setting up LLM-assisted theme analysis...")

    # Import diagnostic functions from the diagnostics module
    try:
        from core.engine.diagnostics import (
            enhance_theme_analysis_with_diagnostics,
            _generate_entity_diagnostics,
            _generate_network_diagnostics,
            _generate_keyword_diagnostics,
            _output_enhanced_diagnostics,
            _group_similar_keywords,
            send_diagnostics_to_llm
        )

        # Create the diagnostic theme analyzer if needed
        if not hasattr(command_processor, 'diagnostic_theme_analyzer'):
            # Create a copy of the theme analyzer with diagnostic capabilities
            from core.modes.themes import ThemeAnalyzer

            command_processor.diagnostic_theme_analyzer = ThemeAnalyzer(
                command_processor.config,
                command_processor.storage_manager,
                command_processor.output_manager,
                command_processor.text_processor
            )

            # Add the diagnostic methods to the analyzer
            setattr(command_processor.diagnostic_theme_analyzer, "enhance_theme_analysis_with_diagnostics",
                    types.MethodType(enhance_theme_analysis_with_diagnostics,
                                     command_processor.diagnostic_theme_analyzer))

            setattr(command_processor.diagnostic_theme_analyzer, "_generate_entity_diagnostics",
                    types.MethodType(_generate_entity_diagnostics, command_processor.diagnostic_theme_analyzer))

            setattr(command_processor.diagnostic_theme_analyzer, "_generate_network_diagnostics",
                    types.MethodType(_generate_network_diagnostics, command_processor.diagnostic_theme_analyzer))

            setattr(command_processor.diagnostic_theme_analyzer, "_generate_keyword_diagnostics",
                    types.MethodType(_generate_keyword_diagnostics, command_processor.diagnostic_theme_analyzer))

            setattr(command_processor.diagnostic_theme_analyzer, "_output_enhanced_diagnostics",
                    types.MethodType(_output_enhanced_diagnostics, command_processor.diagnostic_theme_analyzer))

            setattr(command_processor.diagnostic_theme_analyzer, "_group_similar_keywords",
                    types.MethodType(_group_similar_keywords, command_processor.diagnostic_theme_analyzer))

            setattr(command_processor.diagnostic_theme_analyzer, "send_diagnostics_to_llm",
                    types.MethodType(send_diagnostics_to_llm, command_processor.diagnostic_theme_analyzer))

            debug_print(command_processor.config, "Created diagnostic theme analyzer with enhanced methods")

        debug_print(command_processor.config, "LLM-assisted theme analysis setup successful")

    except Exception as e:
        debug_print(command_processor.config, f"Error during LLM theme analysis setup: {str(e)}")
        import traceback
        traceback.print_exc()
        print("LLM-assisted theme analysis could not be initialized properly. Some features may be limited.")

def run_themes_with_llm_interpretation(self, workspace, method='all'):
    """
    Run themes analysis with LLM-assisted interpretation

    Args:
        workspace (str): The workspace to analyze
        method (str): Analysis method ('all', 'nfm', 'net', 'key', 'lsa', 'cluster')
    """
    debug_print(self.config, f"Running LLM-assisted theme analysis on workspace '{workspace}' with method '{method}'")

    # Check if diagnostic theme analyzer is properly initialized
    if not hasattr(self, 'diagnostic_theme_analyzer'):
        print("Initializing enhanced theme analyzer with diagnostic capabilities...")
        try:
            # Try to set up the analyzer again
            from core.engine.interpreter import setup_llm_theme_analysis
            setup_llm_theme_analysis(self)

            if not hasattr(self, 'diagnostic_theme_analyzer'):
                print("Error: Failed to initialize diagnostic theme analyzer. Falling back to standard analysis.")
                self.theme_analyzer.analyze(workspace, method)
                return

        except Exception as e:
            print(f"Error initializing enhanced theme analyzer: {str(e)}")
            import traceback
            traceback.print_exc()
            print("Falling back to standard analysis")
            self.theme_analyzer.analyze(workspace, method)
            return

    print(f"Running enhanced theme analysis on workspace '{workspace}' with method '{method}'...")

    # First run the standard analysis to ensure all components are initialized properly
    standard_results = self.theme_analyzer.analyze(workspace, method)

    # Then run the enhanced version with diagnostic output
    try:
        results = self.diagnostic_theme_analyzer.enhance_theme_analysis_with_diagnostics(workspace, method,
                                                                                         standard_results)

        # Determine output format
        output_format = self.config.get('system.output_format', 'json')

        # Generate consistent timestamp - this should match what _output_enhanced_diagnostics is using
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create path to the diagnostics file
        diagnostics_path = os.path.join('logs', workspace, f"diagnostics_{method}_{timestamp}.{output_format}")

        # Ask user if they want to generate LLM interpretation
        print("\n=== LLM INTERPRETATION OPTIONS ===")
        print("Would you like the LLM to interpret the analysis results?")
        print("1. Yes - Generate general interpretation")
        print("2. Yes - Ask a specific question about the results")
        print("3. No - Skip LLM interpretation")

        choice = input("Enter choice (1-3): ").strip()

        if choice == '1':
            # Generate a general interpretation for the consolidated results
            if os.path.exists(diagnostics_path):
                print(f"\nGenerating LLM interpretation for {method.upper()} analysis...")
                interpretation = self.diagnostic_theme_analyzer.send_diagnostics_to_llm(diagnostics_path)

                # Display interpretation
                self.output_manager.print_formatted('header', f"LLM INTERPRETATION FOR {method.upper()}")
                print(interpretation)
            else:
                print(f"Warning: Diagnostics file not found at {diagnostics_path}")
                # Try a fallback approach - look for any diagnostics file with similar pattern
                fallback_files = [f for f in os.listdir(os.path.join('logs', workspace))
                                  if f.startswith(f"diagnostics_{method}_") and f.endswith(f".{output_format}")]

                if fallback_files:
                    # Use the most recent file
                    fallback_path = os.path.join('logs', workspace, sorted(fallback_files)[-1])
                    print(f"Using alternative diagnostics file: {fallback_path}")
                    interpretation = self.diagnostic_theme_analyzer.send_diagnostics_to_llm(fallback_path)

                    # Display interpretation
                    self.output_manager.print_formatted('header', f"LLM INTERPRETATION FOR {method.upper()}")
                    print(interpretation)
                else:
                    print("No diagnostics files found. Cannot generate interpretation.")

        elif choice == '2':
            # Check if the diagnostics file exists
            if not os.path.exists(diagnostics_path):
                print(f"Warning: Diagnostics file not found at {diagnostics_path}")
                # Try fallback approach
                fallback_files = [f for f in os.listdir(os.path.join('logs', workspace))
                                  if f.startswith(f"diagnostics_{method}_") and f.endswith(f".{output_format}")]

                if fallback_files:
                    diagnostics_path = os.path.join('logs', workspace, sorted(fallback_files)[-1])
                    print(f"Using alternative diagnostics file: {diagnostics_path}")
                else:
                    print("No diagnostics files found. Cannot generate interpretation.")
                    return

            # Get the question
            question = input("\nEnter your question about the analysis: ").strip()

            if question:
                # Send for interpretation
                print(f"\nSending question to LLM about {method.upper()} analysis...")
                interpretation = self.diagnostic_theme_analyzer.send_diagnostics_to_llm(
                    diagnostics_path, query=question)

                # Display interpretation
                self.output_manager.print_formatted('header', f"LLM ANSWER FOR {method.upper()}")
                print(interpretation)

        print("\nAnalysis complete. Diagnostics and interpretations saved to the logs directory.")

    except Exception as e:
        print(f"Error during enhanced theme analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        print("Analysis could not be completed successfully.")
