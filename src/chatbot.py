"""Chatbot module for conversational RAG with context management.

This module maintains conversation history and uses the RAG pipeline
to answer queries while keeping context from previous messages.
"""

import pandas as pd
from typing import List, Dict, Any, Tuple
from rag_pipeline import answer_query
from llm import generate_with_multiple_input
from config import OPENAI_MODEL, CONTEXT_WINDOW
from tracer import tracer

class ChatBot:
    """
    Conversational chatbot that maintains context and uses RAG pipeline.
    
    Features:
    - Maintains conversation history
    - Limits context window to recent messages
    - Integrates with RAG pipeline for FAQ/Product queries
    - Logs interactions for analysis
    """
    
    def __init__(
        self,
        generator_function=None,
        model: str = None,
        context_window: int = CONTEXT_WINDOW,
        simplified: bool = False
    ):
        """
        Initialize the chatbot.
        
        Args:
            generator_function: Custom function to generate responses (default: answer_query)
            model: Model to use for responses (default: from config)
            context_window: Number of recent messages to keep in context
            simplified: Use simplified RAG mode (faster, cheaper)
        """
        # System prompt defines the chatbot's role
        self.system_prompt = {
            'role': 'system',
            'content': "You are a friendly assistant from Fashion Forward Hub. It is a clothing store selling a variety of items. Your job is to answer questions related to FAQ or Products."
        }
        
        # Initial greeting
        self.initial_message = {
            'role': 'assistant',
            'content': "Hi! How can I help you today?"
        }
        
        # Generator function (defaults to our RAG pipeline)
        if generator_function is None:
            # Default generator that accepts all parameters
            def default_generator(query, **kwargs):
                # Don't pass simplified from self, let it come from kwargs
                return answer_query(query, model=model, **kwargs)
            self.generator_function = default_generator
        else:
            self.generator_function = generator_function
        
        # Store simplified (but don't use it in generator anymore)
        self.simplified = simplified
            
        # Initialize conversation with system prompt and greeting
        self.conversation: List[Dict[str, str]] = [self.system_prompt, self.initial_message]
        
        # Configuration
        self.context_window = context_window
        self.model = model
        self.simplified = simplified
        
        # Logging
        self.logging_dataset = pd.DataFrame(columns=['query', 'result', 'total_tokens', 'kwargs'])
    
    def chat(
        self,
        prompt: str,
        role: str = 'user',
        return_stats: bool = False,
        use_rag: bool = True,
        retriever_type: str = "semantic",
        simplified: bool = False,
        top_k: int = 20,
        alpha: float = 0.5,
        k: int = 60,
        use_reranker: bool = False,
        rerank_query: str = None
    ) -> Dict[str, Any]:
        """Handle a single round of conversation."""

        with tracer.start_as_current_span("chatbot_turn") as turn_span:
            turn_span.set_attribute("query", prompt)
            turn_span.set_attribute("use_rag", use_rag)
            
            if use_rag:
                turn_span.set_attribute("retriever_type", retriever_type)
                turn_span.set_attribute("simplified", simplified)
                turn_span.set_attribute("top_k", top_k)
        
            # Get recent context
            recent_context = self.conversation[-self.context_window:]
            
            # Step 1: Generate RAG response parameters
            params_dict, rag_tokens = self.generator_function(
                prompt,
                use_rag=use_rag,
                retriever_type=retriever_type,
                simplified=simplified,
                top_k=top_k,
                alpha=alpha,
                k=k,
                use_reranker=use_reranker,
                rerank_query=rerank_query
            )
            
            # Step 2: Build messages with appropriate context
            messages_for_llm = recent_context + [
                {"role": "user", "content": params_dict['prompt']}
            ]

            with tracer.start_as_current_span("final_llm_call") as span:
                span.set_attribute("model", params_dict.get('model', self.model))
                span.set_attribute("num_messages", len(messages_for_llm))
            
                # Step 3: Generate response with context
                response = generate_with_multiple_input(
                    messages=messages_for_llm,
                    model=params_dict.get('model', self.model),
                    temperature=params_dict.get('temperature', 1.0),
                    top_p=params_dict.get('top_p', 1.0),
                    max_tokens=params_dict.get('max_tokens', 500)
                )

                # Log LLM tokens
                span.set_attribute("llm_tokens", response['total_tokens'])
                span.set_attribute("response_length", len(response['content']))
            
            # Extract response content
            content = response['content']
            llm_tokens = response['total_tokens']
            total_tokens = rag_tokens + llm_tokens
            
            # Update conversation history
            self.conversation.append({"role": "user", "content": prompt})
            self.conversation.append({"role": "assistant", "content": content})
            
            # Log interaction
            self._log_interaction(prompt, content, total_tokens, params_dict)

            turn_span.set_attribute("rag_tokens", rag_tokens)
            turn_span.set_attribute("llm_tokens", llm_tokens)
            turn_span.set_attribute("total_tokens", total_tokens)
        
            if return_stats:
                return {
                    "content": content,
                    "role": "assistant",
                    "rag_tokens": rag_tokens,
                    "llm_tokens": llm_tokens,
                    "total_tokens": total_tokens
                }
            
            return {"content": content, "role": "assistant"}
        
    def _log_interaction(self, query: str, result: str, total_tokens: int, kwargs: dict):
        """Log the interaction for analysis."""
        new_row = pd.DataFrame([{
            'query': query,
            'result': result,
            'total_tokens': total_tokens,
            'kwargs': str(kwargs)
        }])
        self.logging_dataset = pd.concat([self.logging_dataset, new_row], ignore_index=True)
    
    def clear_conversation(self):
        """Reset conversation to initial state."""
        self.conversation = [self.system_prompt, self.initial_message]
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history."""
        return self.conversation.copy()
    
    def export_logs(self, filepath: str = "chat_logs.csv"):
        """Export conversation logs to CSV."""
        self.logging_dataset.to_csv(filepath, index=False)
        print(f"✅ Logs exported to {filepath}")


def start_terminal_chat(simplified: bool = False):
    """
    Start an interactive terminal chat session.
    
    Args:
        simplified: Use simplified RAG mode
    """
    chatbot = ChatBot(simplified=simplified)
    
    print("=" * 70)
    print("Fashion Forward Hub - Chat Assistant")
    print("=" * 70)
    print(chatbot.initial_message['content'])
    print("\nType 'quit', 'exit', or 'end' to stop the conversation.")
    print("Type 'clear' to reset the conversation.")
    print("Type 'stats' to see token usage for last response.")
    print("=" * 70)
    print()
    
    last_stats = None
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'end']:
                print("\n👋 Thanks for chatting! Goodbye!")
                break
            
            if user_input.lower() == 'clear':
                chatbot.clear_conversation()
                print("🔄 Conversation reset!")
                print(chatbot.initial_message['content'])
                continue
            
            if user_input.lower() == 'stats' and last_stats:
                print(f"\n📊 Last Response Statistics:")
                print(f"   RAG tokens: {last_stats['rag_tokens']}")
                print(f"   LLM tokens: {last_stats['llm_tokens']}")
                print(f"   Total tokens: {last_stats['total_tokens']}")
                continue
            
            # Get response
            print("\n🤔 Assistant is thinking...")
            response = chatbot.chat(user_input, return_stats=True)
            last_stats = response
            
            print(f"\n🤖 Assistant: {response['content']}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.\n")
    
    # Export logs
    chatbot.export_logs()
    print(f"\n📊 Total interactions: {len(chatbot.logging_dataset)}")


if __name__ == "__main__":
    """
    Run interactive chat from command line.
    
    Usage:
        python src/chatbot.py              # Standard mode
        python src/chatbot.py --simplified # Simplified mode (faster/cheaper)
    """
    import sys
    
    simplified = '--simplified' in sys.argv
    
    if simplified:
        print("🚀 Starting chat in SIMPLIFIED mode (faster, lower cost)\n")
    else:
        print("🚀 Starting chat in STANDARD mode (more accurate)\n")
    
    start_terminal_chat(simplified=simplified)