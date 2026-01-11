# client/utils/interaction_logger.py
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


class InteractionLogger:
    """Lightweight logger for agent interactions."""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.current_interaction: Optional[Dict[str, Any]] = None
        
    def start_interaction(self, user_question: str, conversation_history: List[Dict[str, str]], provider: str = "unknown"):
        """
        Start logging a new interaction.
        
        Args:
            user_question: The current user question
            conversation_history: Full conversation history up to this point
            provider: LLM provider being used
        """
        self.current_interaction = {
            "timestamp": datetime.now().isoformat(),
            "provider": provider,
            "user_question": user_question,
            "conversation_history": conversation_history,  # Full chat history
            "start_time": time.time(),
            "end_time": None,
            "execution_time_seconds": None,
            "tool_calls": [],  # List of {tool, query, raw_response, similarity_scores}
            "agent_reasoning": "",  # Complete reasoning chain
            "final_answer": "",
            # Metrics computed later by post-processing script:
            "bert_score": None,
            "perplexity": None
        }
        
    def log_tool_call(self, tool_name: str, query: str, raw_response: str, similarity_scores: List[float] = None):
        """
        Log a complete tool call with its raw response.
        
        Args:
            tool_name: Name of the tool invoked
            query: Query string used
            raw_response: Raw text response from the tool (what LLM sees)
            similarity_scores: List of similarity scores from retrieved documents
        """
        if self.current_interaction is None:
            return
            
        avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else None
        
        tool_call_entry = {
            "tool": tool_name,
            "query": query,
            "raw_response": raw_response,  # This is the key addition!
            "similarity_scores": similarity_scores or [],
            "average_similarity": avg_similarity,
            "timestamp": datetime.now().isoformat()
        }
        self.current_interaction["tool_calls"].append(tool_call_entry)
        
    def log_reasoning(self, reasoning_text: str):
        """Append to agent reasoning chain."""
        if self.current_interaction is None:
            return
            
        if self.current_interaction["agent_reasoning"]:
            self.current_interaction["agent_reasoning"] += "\n" + reasoning_text
        else:
            self.current_interaction["agent_reasoning"] = reasoning_text
            
    def end_interaction(self, final_answer: str) -> Dict[str, Any]:
        """
        End interaction and save log.
        
        Args:
            final_answer: Final answer from agent
            
        Returns:
            The complete interaction data
        """
        if self.current_interaction is None:
            return {}
            
        self.current_interaction["end_time"] = time.time()
        self.current_interaction["execution_time_seconds"] = (
            self.current_interaction["end_time"] - self.current_interaction["start_time"]
        )
        self.current_interaction["final_answer"] = final_answer
        
        # Save to file
        self._save_log()
        
        # Return copy and clear
        interaction_data = self.current_interaction.copy()
        self.current_interaction = None
        
        return interaction_data
        
    def _save_log(self):
        """Save interaction to JSON file."""
        if self.current_interaction is None:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"interaction_{timestamp}.json"
        filepath = self.log_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.current_interaction, f, indent=2, ensure_ascii=False)
            
        print(f"✅ Log saved: {filepath}")
        
    def get_current_interaction(self) -> Optional[Dict[str, Any]]:
        """Get current interaction data."""
        return self.current_interaction