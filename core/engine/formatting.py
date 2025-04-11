"""
Formatting utilities for consistent output appearance
"""

import re

class Colors:
    """ANSI color codes for CLI output"""
    RESET = "\033[0m"
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    GRAY = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Styles
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    DIM = "\033[2m"

    # Backgrounds
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"

class Formatter:
    """Formatting utilities for consistent output appearance"""

    @staticmethod
    def format_header(text, for_tui=False):
        """Format a section header"""
        if for_tui:
            return f"[bold magenta]{text}[/bold magenta]"
        else:
            return f"\n{Colors.BOLD}{Colors.MAGENTA}{text}{Colors.RESET}"

    @staticmethod
    def format_subheader(text, for_tui=False):
        """Format a subsection header"""
        if for_tui:
            return f"[cyan]{text}[/cyan]"
        else:
            return f"\n{Colors.CYAN}{text}{Colors.RESET}"

    @staticmethod
    def format_mini_header(text, for_tui=False):
        """Format a mini header"""
        if for_tui:
            return f"[yellow]{text}[/yellow]"
        else:
            return f"\n{Colors.YELLOW}{text}{Colors.RESET}"

    @staticmethod
    def format_key_value(key, value, indent=0, for_tui=False):
        """Format a key-value pair"""
        spaces = " " * indent
        if for_tui:
            return f"{spaces}[bold white]{key}:[/bold white] {value}"
        else:
            return f"{spaces}{Colors.BRIGHT_WHITE}{key}:{Colors.RESET} {value}"

    @staticmethod
    def format_list_item(text, indent=0, for_tui=False):
        """Format a list item"""
        spaces = " " * indent
        if for_tui:
            return f"{spaces}• {text}"
        else:
            return f"{spaces}• {text}"

    @staticmethod
    def format_feedback(message, success=True, for_tui=False):
        """Format a feedback message"""
        if for_tui:
            if success:
                return f"[green]✓ {message}[/green]"
            else:
                return f"[red]✗ {message}[/red]"
        else:
            if success:
                return f"{Colors.GREEN}✓ {message}{Colors.RESET}"
            else:
                return f"{Colors.RED}✗ {message}{Colors.RESET}"

    @staticmethod
    def format_code_block(content, indent=0, for_tui=False):
        """Format a code block"""
        spaces = " " * indent
        lines = content.split('\n')
        if for_tui:
            formatted_lines = [f"{spaces}[dim]{line}[/dim]" for line in lines]
            return '\n'.join(formatted_lines)
        else:
            formatted_lines = [f"{spaces}{Colors.GRAY}{line}{Colors.RESET}" for line in lines]
            return '\n'.join(formatted_lines)

    @staticmethod
    def format_command(command, for_tui=False):
        """Format a command"""
        if for_tui:
            return f"[bright_blue]CMD> {command}[/bright_blue]"
        else:
            return f"{Colors.BRIGHT_BLUE}CMD> {command}{Colors.RESET}"

    @staticmethod
    def format_error(message, for_tui=False):
        """Format an error message"""
        if for_tui:
            return f"[bold red]{message}[/bold red]"
        else:
            return f"{Colors.BRIGHT_RED}{message}{Colors.RESET}"

    @staticmethod
    def format_warning(message, for_tui=False):
        """Format a warning message"""
        if for_tui:
            return f"[yellow]{message}[/yellow]"
        else:
            return f"{Colors.YELLOW}{message}{Colors.RESET}"

    @staticmethod
    def format_info(message, for_tui=False):
        """Format an info message"""
        if for_tui:
            return f"[white]{message}[/white]"
        else:
            return f"{Colors.WHITE}{message}{Colors.RESET}"

    @staticmethod
    def format_debug(message, for_tui=False):
        """Format a debug message"""
        if for_tui:
            return f"[dim]{message}[/dim]"
        else:
            return f"{Colors.GRAY}{message}{Colors.RESET}"

    @staticmethod
    def format_trace(message, for_tui=False):
        """Format a trace message"""
        if for_tui:
            return f"[dim]{message}[/dim]"
        else:
            return f"{Colors.DIM}{message}{Colors.RESET}"

    @staticmethod
    def strip_ansi(text):
        """Remove ANSI escape codes from text"""
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)