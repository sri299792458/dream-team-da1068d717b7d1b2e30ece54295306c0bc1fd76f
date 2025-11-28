# Tool-Aware Agents: Optional On-Demand Paper Search

## Overview

Dream Team agents are **tool-aware with autonomy**: they know about available tools and **choose when to use them** based on their own judgment. The primary tool is **paper search** via Semantic Scholar API.

**Key Principle**: Agents are **not forced** to search - they decide whether they need papers to ground their reasoning.

## How It Works

### 1. Optional Tool Usage

When `research_api` is available, agents receive a prompt that tells them:
- They have access to a paper search tool
- They can **choose** whether to use it
- If they have enough expertise, they can skip searching

**Agent Decision Flow:**
```
1. Agent receives task/agenda
2. Agent evaluates: "Do I need to search papers or do I know enough?"
3a. If enough knowledge → Provide answer directly
3b. If needs papers → Use search_papers("query")
4. Framework detects search request and executes it
5. Agent receives papers and completes task
```

**Example 1: Agent chooses NOT to search**

```
Agent prompt: "You have access to search_papers(). Propose feature engineering approach."

Agent response: "Based on my ML expertise, I recommend creating rolling statistics
over time windows (24h, 7d, 30d) to capture temporal patterns. This is a standard
approach for time-series problems that I'm confident will work well here."

→ No search performed - agent used existing knowledge
```

**Example 2: Agent chooses TO search**

```
Agent prompt: "You have access to search_papers(). Propose shelf life prediction approach."

Agent response:
Thought: "I need to understand domain-specific approaches for perishable goods"
Action: search_papers("shelf life prediction food storage")

→ Framework executes search

Observation: Found 5 papers including:
  - "Arrhenius-based Models for Food Degradation..." (2021, 156 citations)
    Key insights: Temperature acceleration factor; Q10 model for shelf life

Agent final response: "Based on the literature, I recommend an Arrhenius-based approach
where shelf life degrades exponentially with temperature. We should compute temperature
acceleration factors (Q10 values) as features (Author et al., 2021)..."
```

### 2. When Tools Are Available

Tools are enabled by passing `research_api` to meeting constructors:

```python
meeting = TeamMeeting(
    save_dir="results/meetings",
    research_api=research_assistant.ss_api  # ✅ Enables paper search
)
```

If `research_api` is provided:
- Agents can use `use_react=True` to search papers during reasoning
- The LLM is prompted with search capabilities
- Papers are automatically added to agent knowledge bases

### 3. Agent Types and Tool Usage

#### Domain Experts (Data Scientists, etc.)
**Use ReAct with paper search** to ground their proposals:

```python
meeting.run(
    agent=data_scientist,
    task="Propose feature engineering approach",
    use_react=True  # Enable paper search during reasoning
)
```

The agent will:
1. Think about what to propose based on expertise
2. Search for papers to support/validate the idea
3. Cite papers in the final proposal

#### Coding Agents
**Use ReAct for internal reasoning** (no paper search needed):

```python
meeting.run(
    agent=coding_agent,
    task="Implement the feature engineering pipeline",
    use_react_coding=True  # Internal reasoning, no search
)
```

The agent will:
1. Think about implementation approach
2. Reason about edge cases and data flow
3. Generate code based on reasoning

### 4. Implementation Locations

All enabled in [orchestrator.py](src/dream_team/orchestrator.py):

| Method | Meeting Type | research_api? | use_react? | Purpose |
|--------|-------------|---------------|------------|---------|
| `_initial_exploration()` | Individual | ✅ | ✅ | PI explores problem with papers |
| `_team_planning_meeting()` | Team | ✅ | ✅ | Team discusses approach with papers |
| `_validate_approach()` | Individual | ✅ | ✅ | Validate with literature |
| `_analyze_error()` | Individual | ✅ | ❌ | Error analysis (mechanical) |
| `_trigger_evolution()` | Individual | ✅ | ❌ | Evolution planning |
| `_generate_code()` | Individual | ✅ | ReAct-coding | Code generation (reasoning) |
| `_fix_code_error()` | Individual | ✅ | ❌ | Code fixing |
| `_parse_execution_output()` | Individual | ❌ | ❌ | Parsing (mechanical) |
| `_bootstrap_recruit_team_member()` | Individual | ✅ | ✅ | Recruit with papers |

### 5. How Agents Know About Tools

Agents **receive explicit tool documentation** in their prompts:

```python
"""
You have access to a paper search tool if you need it:
- If you already have sufficient knowledge, complete the task directly
- If you need recent research or supporting evidence, you can search papers

**Available Tool**:
- search_papers(query): Search Semantic Scholar for recent papers (2018-2025)

You may:
1. Just provide your proposal if you have enough knowledge
2. Or use the tool format if you want to search:
   Thought: [Why I need to search]
   Action: search_papers("[your query]")
"""
```

**Key Design Choice**: Tool is presented as **optional**, not mandatory.

**Parsing**: The framework detects when agents choose to search:
```python
# Check if agent requested search
if 'search_papers(' in response.lower() or (
    'thought:' in response.lower() and
    'action:' in response.lower() and
    'search' in response.lower()
):
    # Execute search
    observation = search_and_observe(query)
    # Give results back to agent
```

### 6. Tool Output Integration

Papers found during search are:
- Analyzed by LLM to extract key insights
- Added to agent's knowledge base
- Included in "Observation" for next reasoning step

```python
observation = f"Found {len(papers_found)} relevant papers:
- Paper Title... (Authors et al., Year)
  Key insights: insight 1; insight 2
- Paper Title 2...
  Key insights: insight 3; insight 4
"
```

### 7. Benefits

✅ **Agent autonomy**: Agents decide when they need papers, not forced to search

✅ **Grounded reasoning**: When agents do search, they cite specific papers

✅ **Evolving knowledge**: Papers accumulate in knowledge bases across iterations

✅ **No hallucination**: Techniques backed by actual research when papers used

✅ **Efficient**: Only search when agent determines it's necessary

✅ **Cost effective**: Avoids unnecessary API calls when agent already knows the answer

✅ **Natural workflow**: Mimics how human researchers work - use papers when needed

## Example: End-to-End Flow

```
User: "Build a shelf life prediction model"

1. Initial Exploration (PI with ReAct):
   Thought: "Need to understand perishable goods modeling"
   Action: Search "shelf life prediction food storage"
   Observation: Found papers on thermal abuse, Arrhenius models...
   Final: "We should explore temperature-accelerated degradation models"

2. Team Planning Meeting (Team with ReAct):
   Data Scientist:
     Thought: "Feature engineering should capture temperature patterns"
     Action: Search "time series feature engineering temperature"
     Observation: Found papers on rolling statistics, lag features...
     Final: "Propose rolling mean temperature over 24h windows (Smith et al., 2021)"

   ML Engineer:
     Thought: "Gradient boosting works well for this"
     Action: Search "gradient boosting regression time series"
     Observation: Found papers on XGBoost for forecasting...
     Final: "Recommend XGBoost with custom loss function (Chen et al., 2020)"

   PI Synthesis:
     "Based on the team's research-backed proposals, we'll implement:
      1. Rolling temperature features (Smith et al., 2021)
      2. XGBoost regression (Chen et al., 2020)
      3. Custom evaluation focused on MAE"

3. Code Generation (Coding Agent with ReAct-coding):
   Thought: "Need to structure the pipeline with sklearn"
   Thought: "Handle missing values before feature engineering"
   Thought: "Use ColumnTransformer for separate feature types"
   Final: [generates code based on reasoning]

4. Execution & Iteration...
   (Continues with reflection, learning from errors, evolving agents)
```

## Comparison with TreeSearch

Dream Team's approach is **simpler and more agent-centric**:

| Aspect | TreeSearch | Dream Team |
|--------|-----------|------------|
| **Tool awareness** | Implicit (assumed from ReAct pattern) | **Explicit** (tool documented in prompt) |
| **Agent choice** | Forced ReAct loop | **Optional** - agent decides |
| **Complexity** | Complex tree/stage/parallel structures | Simple linear flow |
| **Transparency** | Hidden in framework | Agent sees tool documentation |
| **Control** | Framework-driven | **Agent-driven** |

**Dream Team advantages**:
- ✅ Agents have autonomy to skip search when not needed
- ✅ Explicit tool documentation in prompts
- ✅ Papers persist in knowledge bases across iterations
- ✅ Simpler to understand and debug
- ✅ Lower cost (no forced searches)

## Future Enhancements

Potential additions:
1. **More tools**: Code search, documentation lookup, data profiling tools
2. **Explicit tool docs**: Add tool documentation to prompts for clarity
3. **Tool selection**: Let agents choose which tool to use when multiple are available
4. **Tool chaining**: Use output of one tool as input to another
5. **Parallel tool use**: Search multiple sources simultaneously

For now, the simple ReAct pattern with Semantic Scholar is sufficient for research-driven agents.
