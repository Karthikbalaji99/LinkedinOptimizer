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


