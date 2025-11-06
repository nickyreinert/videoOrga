# AI Coding Agent Instructions

## Core Principles
- **Language**: ALWAYS use English for all code, comments, variables, and documentation
- **Function Size**: 10-20 lines maximum - create new functions when exceeded
- **Modularity**: Separate concerns into different files and folders
- **Clean Code**: Readable, minimal complexity, descriptive naming
- **No Redundancy**: Create reusable functions instead of duplicating code
- **Task Focus**: ONLY do what is specifically requested - no anticipation or forward thinking

## Code Standards

### Function Requirements
- Use descriptive, speaking names: `calculate_user_balance()`, `render_dashboard_template()`
- Add simple inline comments explaining purpose
- Group similar functions in same file
- Add function overview at top of JavaScript/Python files
- Keep Python files under 200 lines - split into multiple files when exceeded

### Error Handling
```python
def process_user_data(data):
    try:
        # function logic
        return result
    except SpecificError as e:
        log_message(f"Error in process_user_data: {e}", level="ERROR")
        return None
```

### Code Organization
```
project/
├── app.py
├── config.py
├── .env
├── requirements.txt
├── uv.lock
├── Dockerfile
├── docker-compose.yml
├── README.md
├── PROTOCOL.md
├── ARCHITECTURE.md
├── functions/
│   ├── ui/ui_operations.py
│   ├── auth/auth_handlers.py
│   ├── data/data_processors.py
│   └── api/api_endpoints.py
├── templates/
├── static/
├── tests/
│   ├── test_ui/
│   ├── test_auth/
│   └── test_data/
└── utils/
    └── logger.py
```

### Section Headers
**Python/JavaScript:**
```python
# -----------------------
# UI OPERATIONS FUNCTIONS
# -----------------------
```

**HTML:**
```html
<!-- -----------------------
     UI OPERATIONS ELEMENTS
     ----------------------- -->
```

## Technology Stack
- **Package Manager**: uv (instead of pip)
- **Backend**: Python with Flask
- **Database**: SQLAlchemy ORM (project-specific DB choice)
- **Templates**: Jinja2 with HTML
- **Frontend**: Pure JavaScript (clean, stable, fast)
- **Architecture**: Microservice-like with function calls/APIs (no shared global state)
- **Deployment**: Local + Docker support

## Configuration Management
- Use `.env` file for environment variables
- Load with `python-dotenv`
- Structure: `config.py` loads from `.env`
```python
# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY')
    DATABASE_URL = os.environ.get('DATABASE_URL')
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
```

## Security Standards
### Input Validation (ALWAYS implement)
```python
from flask import request
import bleach

def validate_and_sanitize_input(data, allowed_tags=[]):
    if not data:
        return None
    return bleach.clean(data, tags=allowed_tags, strip=True)
```

### Database Security
- Use parameterized queries via SQLAlchemy
- Never concatenate user input into SQL
- Validate all inputs before database operations

## API Response Standards
```python
def create_response(status, data=None, message=""):
    return {
        "status": status,  # "success" or "error"
        "data": data or {},
        "message": message
    }
```

## Testing Requirements
- **Structure**: Mirror main code in `/tests` folder
- **Naming**: `test_[module_name].py`
- **Coverage**: Unit tests for all functions
- **Framework**: pytest
```python
# tests/test_ui/test_ui_operations.py
def test_calculate_user_balance():
    result = calculate_user_balance(test_data)
    assert result == expected_value
```

## Logging System
Create centralized logging in `utils/logger.py`:
- **DEBUG OFF**: Print to STDOUT + send to frontend console
- **DEBUG ON**: Detailed development logging
- Two-level system with configurable debug mode

## Docker Support
- Always design for both local development and Docker deployment
- Include Docker configuration files (structure decided per project)
- Ensure code works in both environments
- Consider containerization when making architectural decisions

## Documentation Requirements

### README.md Format
- Brief explanations with plain bullet points
- No long paragraphs
- Clear structure
- Include: project structure, setup/installation steps (brief), key features, function organization

### ARCHITECTURE.md (New Requirement)
Brief overview of how code parts connect:
```markdown
# System Architecture

## Request Flow
- User request → Flask routes → API handlers → Data processors → Database
- Response: Database → Data processors → API handlers → Templates → User

## Key Components
- `/auth` - User authentication and sessions
- `/ui` - Template rendering and UI logic  
- `/data` - Database operations and data processing
- `/api` - REST endpoints and request handling

## Finding Code
- **Add feature**: Start in `/api` for new endpoints
- **UI changes**: Check `/ui` and `/templates`
- **Data logic**: Look in `/data` processors
- **Authentication**: All in `/auth` handlers
```

### PROTOCOL.md Change Log
Track all changes concisely:
```markdown
# Change Log
## TASK DESCRIPTION
- solution point 1
- solution point 2  
- solution point 3

## CURRENT TASK
- unfinished
```

## Frontend Standards (Pure JavaScript)
- No frameworks - use vanilla JS
- Keep functions small and focused
- Use modern ES6+ features
- Organize by feature in separate files
```javascript
// static/js/ui-operations.js
function handleUserClick(event) {
    // Clean, stable, fast implementation
}
```

## Implementation Workflow
1. Set up file structure + Docker files
2. Configure .env and config.py
3. Implement logging system
4. Create modular functions with tests
5. Add security validation
6. Generate documentation (README.md, ARCHITECTURE.md)
7. Update PROTOCOL.md
8. Ensure English throughout

## Restrictions
- NO emoticons or emojis in code/comments/documentation
- NO example text content unless requested
- NO removal of existing comments
- NO anticipating future needs
- NO shared global variables between functions
- NO direct SQL queries - always use SQLAlchemy ORM