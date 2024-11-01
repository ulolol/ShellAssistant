import re
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import cmd
import shlex
import time
import toml
from typing import List, Dict


def format_markdown(text):
    """Format text with basic Markdown-like syntax to ANSI escape codes for shell display."""
    # ANSI escape codes for formatting
    BOLD_CYAN = '\033[1;96m'   # Bold and Cyan
    ITALIC_MAGENTA = '\033[3;95m'  # Italic and Magenta
    UNDERLINE_GREEN = '\033[4;92m' # Underline and Green
    RESET_BLUE = '\033[0;94m'  # Reset and Blue

    # Replace Markdown-like syntax with ANSI escape codes, ensuring correct precedence
    text = re.sub(r'__(.*?)__', f'{UNDERLINE_GREEN}\\1{RESET_BLUE}', text)       # Underline -> Green
    text = re.sub(r'_(.*?)_', f'{ITALIC_MAGENTA}\\1{RESET_BLUE}', text)          # Italic -> Magenta    
    text = re.sub(r'\*\*(.*?)\*\*', f'{BOLD_CYAN}\\1{RESET_BLUE}', text)         # Bold -> Cyan
    text = re.sub(r'\*(.*?)\*', f'{ITALIC_MAGENTA}\\1{RESET_BLUE}', text, flags=re.DOTALL)          # Italic -> Magenta

    return text


class ChatBotShell(cmd.Cmd):
    intro = 'Welcome to the ChatBot Shell. Type your query and press enter. Type help or ? to list commands.\n'
    prompt = 'chatbot> '
    
    def __init__(self):
        super().__init__()
        self.wrapper = GeminiWebWrapper()
        self.num_results = 5
        self.history = []  # Store chat history
        self.reset_shell_formatting()


    def reset_shell_formatting(self):
        """Reset the shell formatting at startup."""
        RESET_BLUE = '\033[0;94m'  # Reset and Blue
        print(RESET_BLUE, end='')  # Reset formatting


    def default(self, arg):
        """Handle user input as a query."""
        try:
            query = arg.strip()
            if not query:
                print("Hello .. ")
                return

            # Check if user explicitly requests a web search
            if query.lower().startswith("search "):
                query = query[len("search "):]
                print("Checking the InterWebs for you ... ")
                response = self.perform_web_search(query)
            else:
                # Use Gemini model for normal conversation
                #print("\nGenerating response using Gemini...")
                response = self.query_gemini_with_history(query)
                print("\nResponse:")
                print(format_markdown(response))

            # Save query and response to history
            self.history.append({"query": query, "response": response})

        except Exception as e:
            print(f"Error: {str(e)}")

    def perform_web_search(self, query):
        """Perform a web search and return a response."""
        print(f"\nSearching web using Gemini 1.5 Flash...")
        context = self.wrapper.generate_context(query, self.num_results)

        print("\nGenerating response with context...")
        response = self.wrapper.query_gemini(query, context)
        print("\nResponse:")
        print(format_markdown(response))
        print()  # Extra newline for readability
        return response

    def query_gemini_with_history(self, query):
        """Query the Gemini model with the current session history."""
        # Create a context string from chat history
        context = "\n".join(
            [f"User: {entry['query']}\nBot: {entry['response']}" for entry in self.history]
        )
        # Include the current query in the context
        context += f"\nUser: {query}\n"
        return self.wrapper.query_gemini(query, context)

    def do_history(self, arg):
        """Display the chat session history."""
        print("\nChat Session History:")
        for i, entry in enumerate(self.history, 1):
            print(f"{i}. Query: {entry['query']}")
            print(f"   Response: {entry['response']}\n")

    def do_config(self, arg):
        """Show or set default configuration. Usage: config [show|set <param> <value>]"""
        args = shlex.split(arg)
        if not args or args[0] == 'show':
            print(f"Current configuration:")
            print("Current model: Gemini 1.5 Flash")
            print(f"Number of results: {self.num_results}")
        elif args[0] == 'set' and len(args) >= 3:
            param = args[1]
            value = args[2]
            if param == 'results':
                self.num_results = int(value)
            print(f"Updated {param} to {value}")
        else:
            print("Invalid config command. Use 'config show' or 'config set <param> <value>'")

    def do_exit(self, arg):
        """Exit the chatbot shell"""
        print("Goodbye!")
        return True

    def do_quit(self, arg):
        """Exit the chatbot shell"""
        return self.do_exit(arg)

    def do_EOF(self, arg):
        """Exit on Ctrl-D"""
        print()  # Empty line before exiting
        return True

    # Shortcuts for common commands
    do_q = do_quit

    def emptyline(self):
        """Do nothing on empty line"""
        pass


class GeminiWebWrapper:
    def __init__(self):
        self.api_key = self.load_api_key()
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-8b:generateContent"

    def load_api_key(self) -> str:
        """Load the Gemini API key from a TOML file."""
        try:
            config = toml.load("SearchShellGPT.toml")
            return config['gemini']['api_key']
        except Exception as e:
            raise RuntimeError("Failed to load API key from SearchShellGPT.toml") from e

    def search_web(self, query: str, num_results: int = 3) -> List[Dict]:
        """Search the web using Google and return results."""
        try:
            results = []
            for url in search(query, num_results=num_results):
                results.append({
                    'href': url,
                    'title': self._get_page_title(url),
                    'body': ''
                })
            return results
        except Exception as e:
            print(f"Error searching web: {e}")
            return []

    def _get_page_title(self, url: str) -> str:
        """Extract the title from a webpage."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.title.string if soup.title else url
            return title.strip()
        except Exception:
            return url

    def extract_content(self, url: str) -> str:
        """Extract main content from a webpage."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'iframe', 'noscript']):
                element.decompose()

            main_content = soup.find('main') or soup.find('article') or soup.find('div', {'class': ['content', 'main', 'article']})
            if main_content:
                text = main_content.get_text(separator='\n', strip=True)
            else:
                text = soup.get_text(separator='\n', strip=True)

            lines = [line.strip() for line in text.split('\n') if line.strip()]
            cleaned_text = '\n'.join(lines)

            return cleaned_text[:2000]
        except Exception as e:
            print(f"Error extracting content from {url}: {e}")
            return ""

    def generate_context(self, query: str, num_results: int = 3) -> str:
        """Generate context from web search results."""
        results = self.search_web(query, num_results)
        context = []

        for result in results:
            url = result['href']
            title = result['title']

            print(f"\nFetching content from: {url}")
            content = self.extract_content(url)

            if content:
                context_entry = (
                    f"Source: {title}\n"
                    f"URL: {url}\n\n"
                    f"Content:\n{content}\n"
                    f"{'='*50}\n"
                )
                context.append(context_entry)
            else:
                context_entry = (
                    f"Source: {title}\n"
                    f"URL: {url}\n"
                    f"{'='*50}\n"
                )
                context.append(context_entry)

            print("Ingesting Web responses ... ")
            time.sleep(1)

        return "\n".join(context)

    def query_gemini(self, prompt: str, context: str) -> str:
        """Query Gemini with the given prompt and context using direct API call."""
        try:
            full_prompt = (
                f"Converse with the User naturally." 
                f"Always try to incorporate emojis into your responses naturally and wherever possible."
                f"Always try to keep your responses interesting by using emojis."               
                f"If the user asks a search query, then please provide a comprehensive answer based on the context below, \
                    if available, otherwise use your own knowledge. "
                f"Please ensure the answer is detailed with points wherever necessary. "
                f"Please ensure that the answer is properly formatted for reading. "
                f"If the context doesn't contain relevant information, please state that clearly."
                f"Question: {prompt}\n\n"
                f"Context from previous conversations:\n\n{context}\n\n"
            )

            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key
            }

            data = {
                "contents": [{
                    "parts": [{
                        "text": full_prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 2048,
                }
            }

            response = requests.post(
                f"{self.api_url}?key={self.api_key}",
                headers=headers,
                json=data,
                timeout=30
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            # Extract the generated text from the response
            try:
                return response_data['candidates'][0]['content']['parts'][0]['text']
            except (KeyError, IndexError) as e:
                return f"Error parsing Gemini response: {str(e)}"

        except requests.exceptions.RequestException as e:
            return f"Error querying Gemini API: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"

def main():
    ChatBotShell().cmdloop()

if __name__ == "__main__":
    main()
