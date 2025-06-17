import streamlit as st
import json
import os
import uuid
import requests
from datetime import datetime
from typing import Dict, List, Any
import re
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI

from prompts import (
    routing_agent_prompt,
    keyword_extractor_prompt,
    profile_analyzer_prompt,
    job_fit_analyzer_prompt,
    content_enhancer_prompt,
    skill_gap_analyzer_prompt,
    history_manager_prompt,
    collaborative_agent_prompt,
    roadmap_generator_prompt,
    conversation_helper_prompt
)
# Load environment variables from .env file

# Configure logging for terminal output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('linkedin_optimizer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# API Configuration
APIFY_TOKEN = "apify_api_WQ32iCSl3BJGicZUDdDKbD0gdekpXF0aVuoq"
APIFY_ACTOR_ID = "dev_fusion~linkedin-profile-scraper"
GOOGLE_API_KEY = "AIzaSyBwQ5Bv8xuREBWEYtjtlHHZWn6IHBdULqY"

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY   
APIFY_ENDPOINT = (
    f"https://api.apify.com/v2/acts/{APIFY_ACTOR_ID}"
    f"/run-sync-get-dataset-items?token={APIFY_TOKEN}"
)

# Session storage directory
SESSIONS_DIR = "sessions"
os.makedirs(SESSIONS_DIR, exist_ok=True)

@st.cache_resource
def get_llm() -> ChatGoogleGenerativeAI:
    """Initialize and cache the Google Gemini LLM instance"""
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        api_key=GOOGLE_API_KEY
    )


class LinkedInOptimizer:
    def __init__(self) -> None:
        self.llm = get_llm()
        logger.info("Initializing LinkedInOptimizer with Gemini model")

        # Initialize all AI agent chains
        self.routing_chain = LLMChain(llm=self.llm, prompt=routing_agent_prompt)
        self.keyword_chain = LLMChain(llm=self.llm, prompt=keyword_extractor_prompt)
        self.profile_chain = LLMChain(llm=self.llm, prompt=profile_analyzer_prompt)
        self.jobfit_chain = LLMChain(llm=self.llm, prompt=job_fit_analyzer_prompt)
        self.content_chain = LLMChain(llm=self.llm, prompt=content_enhancer_prompt)
        self.skillgap_chain = LLMChain(llm=self.llm, prompt=skill_gap_analyzer_prompt)
        self.history_chain = LLMChain(llm=self.llm, prompt=history_manager_prompt)
        self.collaborative_chain = LLMChain(llm=self.llm, prompt=collaborative_agent_prompt)
        self.roadmap_chain = LLMChain(llm=self.llm, prompt=roadmap_generator_prompt)
        self.conversation_helper_chain = LLMChain(llm=self.llm, prompt=conversation_helper_prompt)
        logger.info("All LangChain agents initialized successfully")

    @staticmethod
    def validate_linkedin_url(profile_url: str) -> bool:
        """Validate LinkedIn profile URL format"""
        pattern = r"^https?://(?:www\.)?linkedin\.com/in/[a-zA-Z0-9\-]+/?$"
        is_valid = bool(re.match(pattern, profile_url.strip()))
        logger.info(f"LinkedIn URL validation: {profile_url} -> {is_valid}")
        return is_valid

    def extract_profile_data(self, profile_url: str) -> Dict[str, Any]:
        """Extract LinkedIn profile data using Apify scraper"""
        logger.info(f"Starting profile extraction for URL: {profile_url}")
        try:
            payload = {"profileUrls": [profile_url], "proxy": {"useApifyProxy": True}}
            r = requests.post(APIFY_ENDPOINT, json=payload, timeout=120)

            if r.status_code >= 400:
                logger.error(f"Apify API error: {r.status_code} - {r.text}")
                raise RuntimeError(f"Apify error {r.status_code}: {r.text}")

            items = r.json()
            if not items:
                logger.warning("Apify returned empty result set")
                raise ValueError("Apify returned an empty result set.")

            profile_data = items[0]

            # Validate and correct data attribution
            if "experiences" in profile_data:
                for exp in profile_data["experiences"]:
                    if "projects" in exp:
                        for project in exp["projects"]:
                            project["role"] = exp.get("title", "Unknown Role")

            logger.info(f"Successfully extracted profile data. Keys: {list(profile_data.keys())}")
            return profile_data

        except Exception as e:
            logger.error(f"Profile extraction failed: {str(e)}")
            raise

    def extract_keywords(self, job_description: str, profile_data: str = "", conversation_context: str = "", current_query: str = "") -> str:
        """Extract relevant keywords from job description using AI"""
        logger.info("Executing keyword extraction agent")
        try:
            result = self.keyword_chain.run({
                "job_description": job_description,
                "profile_data": profile_data,
                "conversation_context": conversation_context,
                "current_query": current_query
            })
            logger.info("Keyword extraction completed successfully")
            return result
        except Exception as e:
            logger.error(f"Keyword extraction failed: {str(e)}")
            st.error(f"Error extracting keywords: {e}")
            return ""
        
    def route_query(self, user_query: str, available_agents: str = None, conversation_context: str = "") -> List[str]:
        """Route user query to appropriate AI agents based on content"""
        logger.info(f"Routing query: {user_query[:100]}...")

        # Check if the user query is a greeting (improved detection)
        greeting_keywords = ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
        user_query_lower = user_query.lower().strip()
        
        # Check for exact greeting or greeting at start of sentence
        if (user_query_lower in greeting_keywords or 
            any(user_query_lower.startswith(greeting) for greeting in greeting_keywords)):
            logger.info("Greeting detected, routing to greeting agent")
            return ["greeting"]

        if available_agents is None:
            available_agents = "profile_analyzer, job_fit_analyzer, content_enhancer, skill_gap_analyzer, roadmap_generator"

        try:
            raw_response = self.routing_chain.run({
                "user_query": user_query,
                "available_agents": available_agents,
                "conversation_context": conversation_context
            })

            logger.info(f"Raw routing response: {raw_response}")

            # Clean and parse the response more robustly
            clean_response = raw_response.strip()
            
            # Remove markdown code blocks if present
            if clean_response.startswith('```'):
                lines = clean_response.split('\n')
                clean_response = '\n'.join(lines[1:-1]) if len(lines) > 2 else clean_response
            
            # Remove any remaining markdown formatting
            clean_response = clean_response.replace('```json', '').replace('```', '').strip()

            # Attempt to parse the response as JSON
            try:
                agents = json.loads(clean_response)
                if isinstance(agents, list) and len(agents) > 0:
                    logger.info(f"Successfully parsed agents: {agents}")
                    return agents
            except json.JSONDecodeError as json_error:
                logger.warning(f"JSON parsing failed: {json_error}")
                logger.warning(f"Attempting text extraction from: {clean_response}")

            # Fallback: extract agent names from text response
            agents = self._extract_agents_from_text(clean_response)
            logger.info(f"Extracted agents from text: {agents}")
            
            if agents:
                return agents
            else:
                logger.error("No valid agents found in routing response")
                raise ValueError("No valid agents identified in routing response")

        except Exception as e:
            logger.error(f"Routing error: {str(e)}")
            logger.error(f"Failed to route query: {user_query}")
            raise RuntimeError(f"Agent routing failed: {str(e)}")
    
    def execute_agent(self, agent_name: str, **kwargs) -> str:
        """Execute individual AI agent and return its response"""
        
        # Handle greeting agent first (before chain mapping)
        if agent_name == "greeting":
            return self.handle_greeting()

        chain_mapping = {
            "profile_analyzer": self.profile_chain,
            "job_fit_analyzer": self.jobfit_chain,
            "content_enhancer": self.content_chain,
            "keyword_analyzer": self.keyword_chain, 
            "skill_gap_analyzer": self.skillgap_chain,
            "history_manager": self.history_chain,
            "roadmap_generator": self.roadmap_chain,
            "conversation_helper": self.conversation_helper_chain
        }
        
        chain = chain_mapping.get(agent_name)
        if not chain:
            logger.error(f"Unknown agent requested: {agent_name}")
            return f"**Error**: Unknown agent `{agent_name}`"

        try:
            logger.info(f"Executing agent: {agent_name}")
            result = chain.run(kwargs)
            logger.info(f"Agent {agent_name} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{agent_name} execution failed: {e}")
            return f"**Error running {agent_name}:** {e}"
        
    def _extract_agents_from_text(self, text: str) -> List[str]:
        """Extract agent names from text when JSON parsing fails"""
        valid_agents = [
            "greeting",
            "conversation_helper",
            "profile_analyzer", 
            "job_fit_analyzer",
            "content_enhancer", 
            "skill_gap_analyzer",
            "roadmap_generator",
            "keyword_analyzer"
        ]

        found_agents = []
        text_lower = text.lower()
        
        for agent in valid_agents:
            # Check for exact match or underscore/hyphen variations
            if (agent in text_lower or 
                agent.replace('_', '-') in text_lower or
                agent.replace('_', ' ') in text_lower):
                found_agents.append(agent)

        # Remove duplicates while preserving order
        found_agents = list(dict.fromkeys(found_agents))
        
        return found_agents
    
    def handle_greeting(self) -> str:
        """Handle greeting from the user"""
        return "Hello! How can I assist you with optimizing your LinkedIn profile today?"
    
    
    def get_relevant_history(self, conversation_history: str, current_query: str, target_agent: str) -> str:
        """Extract relevant conversation history for current query context"""
        logger.info(f"Extracting relevant history for agent: {target_agent}")
        try:
            result = self.history_chain.run({
                "conversation_history": conversation_history,
                "current_query": current_query,
                "target_agent": target_agent,
            })
            logger.info("History extraction completed")
            return result
        except Exception as e:
            logger.error(f"History extraction error: {str(e)}")
            return ""

      
    async def execute_agents_async(self, agents: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Execute multiple agents concurrently for faster processing"""
        logger.info(f"Executing {len(agents)} agents asynchronously: {agents}")
        
        with ThreadPoolExecutor(max_workers=len(agents)) as executor:
            loop = asyncio.get_event_loop()
            tasks = []
            
            for agent in agents:
                task = loop.run_in_executor(
                    executor, 
                    self.execute_agent, 
                    agent, 
                    **kwargs
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            logger.info(f"All {len(agents)} agents completed execution")
            return results


class SessionManager:
    """Manages user chat sessions with persistent storage"""
    
    def __init__(self) -> None:
        self.sessions_dir = SESSIONS_DIR
        logger.info(f"SessionManager initialized with directory: {self.sessions_dir}")
    
    def create_new_session(self) -> str:
        """Create a new chat session and return session ID"""
        session_id = str(uuid.uuid4())
        logger.info(f"Creating new session: {session_id}")
        
        session_data = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "profile_url": "",
            "job_description": "",
            "profile_data": {},
            "keywords": "",
            "messages": []
        }
        
        session_file = os.path.join(self.sessions_dir, f"{session_id}.json")
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        logger.info(f"Session created successfully: {session_id}")
        return session_id
    
    def load_session(self, session_id: str) -> Dict[str, Any]:
        """Load session data from storage file"""
        session_file = os.path.join(self.sessions_dir, f"{session_id}.json")
        try:
            with open(session_file, 'r') as f:
                data = json.load(f)
                logger.info(f"Session loaded: {session_id}")
                return data
        except FileNotFoundError:
            logger.warning(f"Session not found: {session_id}")
            return None
    
    def save_session(self, session_data: Dict[str, Any]):
        """Save session data to storage file"""
        session_id = session_data["session_id"]
        session_file = os.path.join(self.sessions_dir, f"{session_id}.json")
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        logger.info(f"Session saved: {session_id}")
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Retrieve all session summaries for sidebar display"""
        sessions = []
        for filename in os.listdir(self.sessions_dir):
            if filename.endswith('.json'):
                session_id = filename.replace('.json', '')
                session_data = self.load_session(session_id)
                if session_data:
                    summary = {
                        "session_id": session_id,
                        "created_at": session_data.get("created_at", ""),
                        "message_count": len(session_data.get("messages", [])),
                        "has_profile": bool(session_data.get("profile_data")),
                        "has_jd": bool(session_data.get("job_description", "").strip())
                    }
                    sessions.append(summary)
        
        # Sort by creation date, newest first
        sessions.sort(key=lambda x: x["created_at"], reverse=True)
        logger.info(f"Retrieved {len(sessions)} sessions")
        return sessions


def initialize_streamlit():
    """Initialize Streamlit page configuration and session state"""
    st.set_page_config(
        page_title="LinkedIn Profile Optimizer",
        page_icon="💼",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize optimizer instance
    if 'linkedin_optimizer' not in st.session_state:
        st.session_state.linkedin_optimizer = LinkedInOptimizer()

    # Initialize session manager
    if 'session_manager' not in st.session_state:
        st.session_state.session_manager = SessionManager()

    # Initialize current session tracking
    if 'current_session_id' not in st.session_state:
        st.session_state.current_session_id = None


def render_sidebar():
    """Render sidebar with session management controls"""
    with st.sidebar:
        st.title("💼 LinkedIn Optimizer")
        st.markdown("---")
        
        # New chat session button
        if st.button("➕ New Chat", use_container_width=True):
            session_id = st.session_state.session_manager.create_new_session()
            st.session_state.current_session_id = session_id
            logger.info(f"New chat session created from UI: {session_id}")
            st.rerun()
        
        st.markdown("### 📝 Chat Sessions")
        
        # Display all existing sessions
        sessions = st.session_state.session_manager.get_all_sessions()
        
        if sessions:
            for session in sessions:
                # Create readable session name
                session_name = f"Chat {session['session_id'][:8]}"
                if session['message_count'] > 0:
                    session_name += f" ({session['message_count']} msgs)"
                
                # Session selection button
                if st.button(
                    session_name,
                    key=f"session_{session['session_id']}",
                    use_container_width=True,
                    type="primary" if st.session_state.current_session_id == session['session_id'] else "secondary"
                ):
                    st.session_state.current_session_id = session['session_id']
                    logger.info(f"Switched to session: {session['session_id']}")
                    st.rerun()
                
                # Show session status indicators
                status_icons = []
                if session['has_profile']:
                    status_icons.append("👤")  # Profile loaded
                if session['has_jd']:
                    status_icons.append("📄")  # Job description added
                
                if status_icons:
                    st.caption(" ".join(status_icons))
        else:
            st.info("No chat sessions yet. Create a new chat to get started!")
            
        # Debug information (optional)
        if st.checkbox("Show Debug Info"):
            st.write("Current Session:", st.session_state.current_session_id)
            st.write("Total Sessions:", len(sessions))


def render_input_panel():
    """Render input panel for LinkedIn URL and job description"""
    st.markdown("### 🔗 LinkedIn Profile & Job Description")
    
    # Get current session data for pre-filling inputs
    current_session = None
    if st.session_state.current_session_id:
        current_session = st.session_state.session_manager.load_session(
            st.session_state.current_session_id
        )
    
    # LinkedIn profile URL input
    profile_url = st.text_input(
        "LinkedIn Profile URL",
        value=current_session.get("profile_url", "") if current_session else "",
        placeholder="https://linkedin.com/in/username",
        help="Paste the LinkedIn profile URL you want to optimize"
    )
    
    # Job description input
    job_description = st.text_area(
        "Job Description",
        value=current_session.get("job_description", "") if current_session else "",
        height=300,
        placeholder="Paste the job description for the role you're targeting...",
        help="Paste the complete job description including requirements and responsibilities"
    )
    
    # Auto-save inputs to current session when changed
    if current_session and (
        profile_url != current_session.get("profile_url", "") or 
        job_description != current_session.get("job_description", "")
    ):
        current_session["profile_url"] = profile_url
        current_session["job_description"] = job_description
        st.session_state.session_manager.save_session(current_session)
        logger.info(f"Updated session inputs: URL={bool(profile_url)}, JD={bool(job_description)}")
    
    return profile_url, job_description


def render_chat_interface():
    """Render main chat interface with message history"""
    st.markdown("### 💬 Chat Assistant")
    
    # Ensure a session is selected
    if not st.session_state.current_session_id:
        st.info("👈 Please select or create a chat session from the sidebar to get started.")
        return
    
    # Load current session data
    current_session = st.session_state.session_manager.load_session(
        st.session_state.current_session_id
    )
    
    if not current_session:
        st.error("Session not found. Please create a new session.")
        return
    
    # Display conversation history
    messages = current_session.get("messages", [])
    
    for message in messages:
        with st.chat_message("user"):
            st.write(message["user_query"])
        
        with st.chat_message("assistant"):
            st.markdown(message["ai_response"])
    
    # Chat input field
    if prompt := st.chat_input("Ask me anything about your LinkedIn profile optimization..."):
        logger.info(f"User input received: {prompt[:100]}...")
        
        # Handle first message (requires profile extraction)
        if len(messages) == 0:
            profile_url = current_session.get("profile_url", "").strip()
            job_description = current_session.get("job_description", "").strip()
            
            # Validate required inputs
            if not profile_url:
                st.error("❌ Please provide a LinkedIn profile URL before starting the chat.")
                return
            
            if not st.session_state.linkedin_optimizer.validate_linkedin_url(profile_url):
                st.error("❌ Please provide a valid LinkedIn profile URL (e.g., https://linkedin.com/in/username)")
                return
            
            if not job_description:
                st.error("❌ Please provide a job description before starting the chat.")
                return
            
            # Process first query with data extraction
            with st.spinner("🔄 Extracting profile data and analyzing..."):
                process_first_query(current_session, prompt)
        else:
            # Process follow-up queries
            with st.spinner("🤔 Processing your query..."):
                process_followup_query(current_session, prompt)


def process_first_query(session_data: Dict[str, Any], user_query: str):
    """Process the first query with profile extraction and initial analysis"""
    logger.info("Processing first query - starting data extraction")
    
    try:
        optimizer = st.session_state.linkedin_optimizer
        
        # Extract LinkedIn profile data
        logger.info("Extracting LinkedIn profile data...")
        profile_data = optimizer.extract_profile_data(session_data["profile_url"])
        
        if not profile_data:
            st.error("Failed to extract profile data. Please check the LinkedIn URL.")
            return
        
        # Extract keywords from job description with proper parameters
        logger.info("Analyzing job description for keywords...")
        keywords = optimizer.extract_keywords(
            job_description=session_data["job_description"],
            profile_data=json.dumps(profile_data, indent=2),
            conversation_context="",
            current_query=user_query
        )
        
        # Cache extracted data in session
        session_data["profile_data"] = profile_data
        session_data["keywords"] = keywords
        logger.info("Cached profile data and keywords in session")
        
        # Route query to appropriate agents
        logger.info("Routing query to appropriate agents...")
        selected_agents = optimizer.route_query(user_query, conversation_context="")
        
        logger.info(f"Selected agents: {', '.join(selected_agents)}")
        
        # Execute selected agents synchronously
        ai_response = execute_agents_sync(
            selected_agents,
            session_data,
            user_query,
            ""
        )
        
        # Save conversation to session
        save_message_to_session(session_data, user_query, ai_response)
        
        # Display new messages
        display_chat_messages(user_query, ai_response)
        
        logger.info("First query processing completed successfully")
        st.rerun()
        
    except Exception as e:
        error_msg = f"Error processing first query: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)

def process_followup_query(session_data: Dict[str, Any], user_query: str):
    """Process follow-up queries with conversation context"""
    logger.info("Processing follow-up query")
    
    try:
        optimizer = st.session_state.linkedin_optimizer
        
        # Get conversation history for context
        conversation_history = format_conversation_history(session_data["messages"])
        
        # Extract relevant context for current query
        relevant_context = ""
        if conversation_history:
            primary_agent = None  # Let the system determine the best agent
            relevant_context = optimizer.get_relevant_history(
                conversation_history, user_query, primary_agent
            )
        
        # Route query with conversation context
        selected_agents = optimizer.route_query(user_query, conversation_context=relevant_context)
        logger.info(f"Selected agents: {', '.join(selected_agents)}")
        
        # Execute selected agents
        ai_response = execute_agents_sync(
            selected_agents,
            session_data,
            user_query,
            relevant_context
        )
        
        # Save conversation to session
        save_message_to_session(session_data, user_query, ai_response)
        
        # Display new messages
        display_chat_messages(user_query, ai_response)
        
        logger.info("Follow-up query processing completed successfully")
        st.rerun()
        
    except Exception as e:
        error_msg = f"Error processing follow-up query: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)


def execute_agents_sync(
    selected_agents: List[str],
    session_data: Dict[str, Any],
    user_query: str,
    relevant_context: str
) -> str:
    """Execute selected agents synchronously and merge their responses"""
    optimizer = st.session_state.linkedin_optimizer

    # Prepare common input parameters for all agents
    common_params = {
        "profile_data": json.dumps(session_data["profile_data"], indent=2),
        "job_description": session_data["job_description"],
        "keywords": session_data["keywords"],
        "conversation_context": relevant_context,
        "current_query": user_query,
    }

    # Execute each selected agent and collect outputs
    agent_outputs = []
    for agent in selected_agents:
        logger.info(f"Executing agent: {agent}")
        if agent == "conversation_helper":
            conversation_helper_params = {
                "conversation_context": relevant_context,
                "current_query": user_query,
                "session_data": json.dumps({
                    "messages": session_data.get("messages", []),
                    "session_id": session_data.get("session_id", ""),
                    "created_at": session_data.get("created_at", "")
                }, indent=2)
            }
            raw_output = optimizer.execute_agent(agent, **conversation_helper_params).strip()
        else:
            raw_output = optimizer.execute_agent(agent, **common_params).strip()
            
        agent_outputs.append(f"{agent}:\n{raw_output}")

    # Return single agent output directly, or merge multiple outputs
    if len(agent_outputs) == 1:
        # Single agent: return output without agent label
        return agent_outputs[0].split("\n", 1)[1].strip()

    # Multiple agents: use collaborative agent to merge responses
    logger.info("Merging multiple agent outputs using collaborative agent")
    merged_response = optimizer.collaborative_chain.run({
        "user_query": user_query,
        "agent_outputs": "\n\n".join(agent_outputs),
        "context": relevant_context,
    })

    return merged_response.strip()


def save_message_to_session(session_data: Dict[str, Any], user_query: str, ai_response: str):
    """Save user query and AI response to session history"""
    message_id = str(uuid.uuid4())
    new_message = {
        "message_id": message_id,
        "user_query": user_query,
        "ai_response": ai_response,
        "timestamp": datetime.now().isoformat()
    }
    
    session_data["messages"].append(new_message)
    st.session_state.session_manager.save_session(session_data)
    logger.info(f"Message saved to session: {message_id}")
    

def display_chat_messages(user_query: str, ai_response: str):
    """Display user query and AI response in chat format"""
    with st.chat_message("user"):
        st.write(user_query)
    
    with st.chat_message("assistant"):
        st.markdown(ai_response)


def format_conversation_history(messages: List[Dict[str, Any]]) -> str:
    """Format recent conversation history for context awareness"""
    if not messages:
        return ""
    
    history = []
    # Use last 5 messages for context (prevents token limit issues)
    for msg in messages[-5:]:
        history.append(f"User: {msg['user_query']}")
        # Truncate AI responses to 500 chars for context efficiency
        history.append(f"Assistant: {msg['ai_response'][:500]}...")
    
    return "\n".join(history)


def main() -> None:
    """Main application entry point"""
    logger.info("Starting LinkedIn Optimizer application")
    
    # Initialize Streamlit configuration and state
    initialize_streamlit()
    
    # Create three-column layout
    col1, col2, col3 = st.columns([0.5, 5, 1])
    
    # Render application components
    render_sidebar()
    
    with col1:
        st.empty()  # Left spacer column
    
    with col2:
        render_chat_interface()  # Main chat interface
    
    with col3:
        render_input_panel()  # LinkedIn URL and job description inputs


if __name__ == "__main__":
    main()