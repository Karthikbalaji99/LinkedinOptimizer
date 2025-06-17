# LinkedIn Profile Optimizer

LinkedIn Profile Optimizer is a Streamlit application designed to help users optimize their LinkedIn profiles for better visibility and alignment with job descriptions. The application uses AI agents to analyze profiles, extract keywords, enhance content, and provide actionable insights.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Code Structure](#code-structure)
- [Contributing](#contributing)
- [License](#license)
- [Changes to v1.1](#changes-to-v1.1)

## Features

- **Profile Analysis**: Analyze LinkedIn profiles for completeness, structure, and quality.
- **Job Fit Analysis**: Compare user profiles against job descriptions to assess compatibility.
- **Content Enhancement**: Rewrite and optimize profile content for better impact.
- **Keyword Extraction**: Extract relevant keywords from job descriptions.
- **Skill Gap Analysis**: Identify missing skills and provide learning recommendations.
- **Session Management**: Manage multiple chat sessions with persistent storage.

## Installation

To run the LinkedIn Profile Optimizer locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd linkedin-profile-optimizer
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   Create a `.env` file in the root directory and add your API keys:
   ```env
   APIFY_TOKEN=your_apify_token
   GOOGLE_API_KEY=your_google_api_key
   ```

5. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Create a New Chat Session**:
   - Click on the "New Chat" button in the sidebar to start a new session.

2. **Enter LinkedIn Profile URL and Job Description**:
   - Provide the LinkedIn profile URL and the job description you are targeting.

3. **Interact with the Chat Assistant**:
   - Ask questions about your LinkedIn profile optimization, job fit, content enhancement, keyword extraction, and skill gap analysis.

4. **Manage Sessions**:
   - Use the sidebar to switch between different chat sessions or create new ones.

## Configuration

The application requires the following API keys:

- **Apify API Token**: Used for extracting LinkedIn profile data.
- **Google API Key**: Used for accessing the Google Generative AI model.

Ensure these keys are set in your environment variables or `.env` file.

## Code Structure

The codebase is organized as follows:

- `app.py`: Main application file containing the Streamlit interface and core logic.
- `prompts.py`: Contains prompt templates for various AI agents.
- `sessions/`: Directory for storing session data.

### Key Classes and Functions

- `LinkedInOptimizer`: Main class handling LinkedIn profile analysis using AI agents.
- `SessionManager`: Manages user chat sessions with persistent storage.
- `initialize_streamlit()`: Initializes Streamlit page configuration and session state.
- `render_sidebar()`: Renders the sidebar with session management controls.
- `render_input_panel()`: Renders input fields for LinkedIn URL and job description.
- `render_chat_interface()`: Renders the main chat interface with message history.
- `process_first_query()`: Processes the first query with profile extraction and initial analysis.
- `process_followup_query()`: Processes follow-up queries with conversation context.

## Changes to v1.1

### 1. **New Agents and Prompts Added**

#### **Conversation Helper (`conversation_helper`)**
- **Description**: 
  - Added a new agent to handle meta-queries related to the conversation history, such as "What did we discuss?", "What was my first request?", and "Show me our conversation history."
- **Reason**: 
  - This addresses feedback regarding the tool's inability to handle and provide specific answers about past interactions.

#### **Roadmap Generator (`roadmap_generator`)**
- **Description**: 
  - Introduced a new agent that generates personalized career development roadmaps, including milestones, timelines, and specific learning resources.
- **Reason**: 
  - The previous version provided generic career advice. This agent creates a tailored roadmap that is actionable and specific to the user's profile and target job role.

#### **Greeting Agent (`greeting`)**
- **Description**: 
  - This new agent responds to user greetings such as "Hi", "Hello", "Good morning", etc., to make the tool more conversational.
- **Reason**: 
  - This improves the user experience by making the interaction feel more natural and friendly.

### 2. **Routing Logic Enhancements**

#### **Meta-Query Handling**
- **Description**: 
  - Updated routing logic to specifically detect meta-queries related to conversation history and route them to the `conversation_helper` agent.
- **Reason**: 
  - Previously, the tool failed to respond to queries about past interactions. This update ensures that history-related queries are routed correctly.

#### **Improved Routing for Career Roadmap Queries**
- **Description**: 
  - Routing logic now detects when users inquire about career roadmaps and routes these queries to the `roadmap_generator` agent.
- **Reason**: 
  - This change ensures that queries about personalized career development are handled by the correct agent, delivering tailored roadmaps.

#### **Format Detection**
- **Description**: 
  - Added logic to detect user preferences for response formats (e.g., lists, bullet points) and ensure agents return responses in the requested format.
- **Reason**: 
  - Users wanted more structured and actionable outputs. This ensures that responses are formatted according to the user's preferences.

### 3. **Profile Content Enhancements**

#### **Role Attribution Accuracy**
- **Description**: 
  - Updated the `content_enhancer` agent to maintain correct attribution of projects to their respective job roles (e.g., internship projects are not moved to full-time roles).
- **Reason**: 
  - Feedback indicated that the tool was misattributing internship projects to full-time roles. This change ensures that content is attributed correctly.

#### **Improved Content Enhancement for LinkedIn Profile**
- **Description**: 
  - Enhanced the profile analyzer and content enhancer agents to focus on results-driven descriptions, quantified achievements, and accurate role attribution.
- **Reason**: 
  - Previous content suggestions were too generic. This update ensures that content is more specific, tailored, and actionable for profile optimization.

### 4. **Structured Learning Pathways**

#### **Detailed Skill Gap Analysis**
- **Description**: 
  - Enhanced the `skill_gap_analyzer` agent to provide specific, actionable learning resources (e.g., courses, platforms, instructors).
- **Reason**: 
  - Feedback indicated that skill gap analysis was too generalized. This change provides users with clear, detailed learning paths to fill skill gaps.

#### **Milestones and Timelines in Career Roadmap**
- **Description**: 
  - The `roadmap_generator` now includes specific milestones, timelines, and actionable steps for career progression.
- **Reason**: 
  - The previous roadmap was too vague. This update ensures that users receive concrete, measurable milestones with timelines to track their career development.

### 5. **Improved Query Routing Logic**

#### **Meta-Query Handling for Conversation History**
- **Description**: 
  - Enhanced routing logic to route conversation history queries to the `conversation_helper` agent.
- **Reason**: 
  - Ensures that users' requests for past conversation details are addressed accurately and promptly.

#### **Improved Detection for Career Roadmap Queries**
- **Description**: 
  - Routing logic now specifically detects when users ask for career roadmaps and routes those queries to the `roadmap_generator`.
- **Reason**: 
  - Ensures that queries about personalized career development plans are handled correctly and generate tailored roadmaps.

### 6. **User Feedback and Interactions**

#### **Greeting Logic**
- **Description**: 
  - The tool now detects greetings and routes them to the `greeting` agent for a friendly and conversational response.
- **Reason**: 
  - Adds a conversational element to the interaction, making the tool more user-friendly and engaging.

### 7. **Improved Data Processing and Error Handling**

#### **Data Attribution and Handling**
- **Description**: 
  - Ensured that each project is attributed to the correct job role (e.g., internship projects remain with the internship role).
- **Reason**: 
  - This prevents the issue where projects were incorrectly moved between job roles, ensuring accurate and meaningful profile data.

### 8. **Optimizing Response Generation**

#### **Synthesis of Multiple Agent Outputs**
- **Description**: 
  - Added the `collaborative_agent` to merge responses from multiple agents into a cohesive, user-friendly message.
- **Reason**: 
  - Improves the final response by combining insights from multiple agents into a single, clear, and actionable answer.

---

