#!/usr/bin/env python3
import cmd
import asyncio
import re
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import json
import aiohttp
import toml
import time
from typing import List, Dict

def format_markdown(text):
    """Format text with basic Markdown-like syntax to ANSI escape codes for shell display."""
    BOLD_CYAN = '\033[1;96m'
    ITALIC_MAGENTA = '\033[3;95m'
    UNDERLINE_GREEN = '\033[4;92m'
    RESET_BLUE = '\033[0;94m'
    BOLD_ITALIC_MAGENTA = '\033[1;3;95m' # Bold, Italic and Magenta

    # Bold, Italic and Magenta
    text = re.sub(r'_\*_(.*?)_\*_', f'{BOLD_ITALIC_MAGENTA}\\1{RESET_BLUE}', text,  flags=re.DOTALL) # Bold, Italic -> Magenta
    text = re.sub(r'__(.*?)__', f'{UNDERLINE_GREEN}\\1{RESET_BLUE}', text)
    text = re.sub(r'_(.*?)_', f'{ITALIC_MAGENTA}\\1{RESET_BLUE}', text)
    text = re.sub(r'\*\*(.*?)\*\*', f'{BOLD_CYAN}\\1{RESET_BLUE}', text)
    text = re.sub(r'\*(.*?)\*', f'{ITALIC_MAGENTA}\\1{RESET_BLUE}', text, flags=re.DOTALL)
    
    return text

class OpenAIWebWrapper:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.api_key = self.load_api_key()

    def load_api_key(self) -> str:
        """Load the OpenAI API key from a TOML file."""
        try:
            config = toml.load("SearchShellGPT.toml")
            return config['openai']['api_key']
        except Exception as e:
            raise RuntimeError("Failed to load API key from config.toml") from e

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
                'User-Agent': 'Mozilla/5.0'
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
                'User-Agent': 'Mozilla/5.0'
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

    

    async def query_openai_async(self, prompt: str, context: str, chat_history: list = [], callback=None) -> str:
        """Query OpenAI with the given prompt and context using aiohttp, and handle streaming response."""
        try:
            messages = [{"role": "system", "content": "You are a helpful assistant."}]

            # Add chat history to messages
            messages.extend(chat_history)
            if context:
                messages.append({"role": "system", "content": f"Previous Chat History:\n{context}"})

            # Add current prompt and context to messages
            full_prompt = (
                f"Context from web searches:\n\n{context}\n\n"
                f"Question: {prompt}\n\n"
                f"Provide a comprehensive answer based on the context above."
                f"Converse with the User naturally." 
                f"Always try to incorporate emojis into your responses naturally and wherever possible."
                f"Always try to keep your responses interesting by using emojis."               
                f"If the user asks a search query, then please provide a comprehensive answer based on the context below, \
                    if available, otherwise use your own knowledge. "
                f"Please ensure the answer is detailed with points wherever necessary. "
                f"Please ensure that the answer is properly formatted for reading. "
                f"If the context doesn't contain relevant information, please state that clearly."
            )

            messages.append({"role": "user", "content": full_prompt})

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    'https://api.openai.com/v1/chat/completions',
                    headers={'Authorization': f'Bearer {self.api_key}'},
                    json={
                        "model": self.model_name,
                        "messages": messages,
                        "max_tokens": 2048,
                        "stream": True  # Ensure the API supports this
                    }
                ) as response:
                    response_content = ""

                    async for line in response.content:
                        line_content = line.decode('utf-8').strip()
                        if not line_content or line_content == "[DONE]":
                            continue
                        
                        if line_content.startswith("data: "):
                            line_content = line_content[len("data: "):]
                        
                        # Parse the JSON content
                        #try:
                        json_data = json.loads(line_content)
                        delta_content = json_data['choices'][0]['delta'].get('content', '')
                        response_content += delta_content                            
                        # Use the callback to handle streaming data
                        if callback:
                            callback(delta_content)
                            
                        #except json.JSONDecodeError as e:
                            #print(f"Error decoding JSON: {e}")

            return response_content.strip()
        except Exception as e:
            return f"Error querying OpenAI: {e}"


class ConversationShell(cmd.Cmd):
    intro = 'Welcome to the AI Conversation Shell. Type your message or use "search <query>" to search for information.\nType help or ? to list commands.\n'
    prompt = 'ai> '

    def __init__(self):
        super().__init__()
        self.wrapper = OpenAIWebWrapper(model_name="gpt-4o-mini")
        self.num_results = 5
        self.chat_history = []
        self.reset_shell_formatting()
        self.buffer = []

    def reset_shell_formatting(self):
        """Reset the shell formatting at startup."""
        RESET_BLUE = '\033[0;94m'  # Reset and Blue
        print(RESET_BLUE, end='')  # Reset formatting

    def display_streaming_output(self, content):
        """Callback function to display streaming content."""
        # Accumulate content in the buffer
        self.buffer.append(content)

        # Check if the buffer contains a complete logical segment
        if content.endswith(('.', '!', '?', '\n')):
            # Join buffer and apply markdown formatting
            full_content = ''.join(self.buffer)
            formatted_content = format_markdown(full_content)
            print(formatted_content, end='', flush=True)

            # Clear the buffer after printing
            self.buffer.clear()

    def do_search(self, arg):
        """Search and add context to chat history."""
        if arg.strip():
            print(f"\nChecking the InterWebs for {arg}...")
            context = self.wrapper.generate_context(arg, self.num_results)
            if not context.strip():
                print("No relevant information found from the search.")
                return

            self.chat_history.append({"role": "system", "content": f"Web Search Context:\n{context}"})
            asyncio.run(self.handle_streaming_response(f"Summarize the following context:\n{context}"))
        else:
            print("Please provide a search query.")

    def default(self, line):
        """Handle general conversation input."""
        if line.startswith("search "):
            self.do_search(line[7:])
        else:
            self.chat_history.append({"role": "user", "content": line})
            asyncio.run(self.handle_streaming_response(line))

    async def handle_streaming_response(self, prompt):
        """Handle streaming response from OpenAI."""
        print(format_markdown("\n_*_Response:_*_"))
        try:
            response_content = await self.wrapper.query_openai_async(
                prompt, "", self.chat_history, callback=self.display_streaming_output
            )
            self.chat_history.append({"role": "assistant", "content": response_content})
            #print()  # Add a newline after streaming is done
        except Exception as e:
            print(f"Error querying OpenAI: {e}")
        print("\n\n")

    def do_exit(self, arg):
        """Exit the conversation shell."""
        print(format_markdown("_*_Goodbye!_*_"))
        return True

    def do_quit(self, arg):
        """Exit the conversation shell."""
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

def main():
    ConversationShell().cmdloop()

if __name__ == "__main__":
    main()
