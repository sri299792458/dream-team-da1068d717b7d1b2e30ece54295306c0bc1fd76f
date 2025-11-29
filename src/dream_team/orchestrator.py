"""
Experiment orchestration for autonomous Dream Team operation.

Coordinates agents, code execution, and iterative improvement.
Uses mathematical framework for emergent evolution.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np

from .agent import Agent
from .executor import CodeExecutor, extract_code_from_text
from .meetings import TeamMeeting, IndividualMeeting
from .evolution_agent import EvolutionAgent, EvolutionDecision
from .research import get_research_assistant
from .utils import save_json
from .context import Reflection, ReflectionMemory
from .event_store import EventStore
from .analyzers import OutputAnalyzer, CodeAnalyzer
from .semantic_state import IterationRecord
from .context_builder import ContextBuilder


class ExperimentOrchestrator:
    """Orchestrates autonomous experimentation with evolving agents"""

    def __init__(
        self,
        team_lead: Agent,
        team_members: List[Agent],
        coding_agent: Agent,
        results_dir: Path,
        evolution_engine: Optional[Any] = None,
    ):
        """
        Initialize orchestrator.

        Args:
            team_lead: Lead agent who coordinates
            team_members: Other agents on the team (strategists, domain experts)
            coding_agent: Dedicated agent who implements code based on team discussions
            results_dir: Directory to save results
            evolution_engine: Engine for agent evolution (unused for now)
        """
        self.team_lead = team_lead
        self.team_members = team_members  # Can be empty initially - PI recruits after bootstrap
        self.coding_agent = coding_agent
        self.all_agents = [team_lead] + team_members
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.evolution_agent: Optional[EvolutionAgent] = None  # Will be initialized in run()
        self.evolution_decision: Optional[EvolutionDecision] = None
        self.executor: Optional[CodeExecutor] = None  # Created when run() is called
        self.research = get_research_assistant()

        # Get LLM for query generation and other tasks
        from .llm import get_llm
        self.llm = get_llm()

        # === Layer 1: Event store ===
        experiment_id = self.results_dir.name
        self.event_store = EventStore(
            experiment_id=experiment_id,
            storage_dir=self.results_dir / "events",
        )

        # === Layer 2: analyzers (semantic state builders) ===
        self.output_analyzer = OutputAnalyzer(self.llm)
        self.code_analyzer = CodeAnalyzer(self.llm)
        self.iteration_records: List[IterationRecord] = []

        # Reflexion: Memory of past reflections for learning
        self.reflection_memory = ReflectionMemory()

        # === Layer 3: Context builder ===
        self.context_builder = ContextBuilder(
            experiment_id=experiment_id,
            reflection_memory=self.reflection_memory,
        )

        self.iteration = 0
        self.best_metric: Optional[float] = None
        self.bootstrap_completed = len(team_members) > 0  # Skip bootstrap if team already exists
        self.column_schemas: Dict[str, List[str]] = {}  # Will be populated during bootstrap

        # These are set in run()
        self.problem_statement: Optional[str] = None
        self.target_metric: Optional[str] = None
        self.minimize_metric: bool = True

    # ======================================================================
    # Main run loop
    # ======================================================================

    def run(
        self,
        problem_statement: str,
        data_context: Dict[str, Any],
        target_metric: str,
        minimize_metric: bool = True,
        max_iterations: int = 5,
        target_score: Optional[float] = None,
        resume: bool = True,
    ) -> Dict[str, Any]:
        """
        Run autonomous experimentation.

        Args:
            problem_statement: Description of the challenge
            data_context: Dictionary with data (e.g., {'train_df': df, 'test_df': df})
            target_metric: Name of metric to optimize (e.g., 'mae', 'f1')
            minimize_metric: Whether lower is better
            max_iterations: Maximum iterations before stopping
            target_score: Optional target score to achieve
            resume: (currently ignored) placeholder for future resume-from-events

        Returns:
            Final experiment summary
        """
        print("=" * 60)
        print("🚀 AUTONOMOUS DREAM TEAM EXPERIMENT")
        print("=" * 60)
        print(f"\nProblem: {problem_statement[:100]}...")
        print(f"Target Metric: {target_metric} ({'minimize' if minimize_metric else 'maximize'})")
        print(f"Max Iterations: {max_iterations}")
        if target_score is not None:
            print(f"Target Score: {target_score}")
        print()

        # Initialize executor with data
        artifacts_dir = self.results_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        if data_context is None:
            data_context = {}
        data_context["artifacts_dir"] = artifacts_dir

        self.executor = CodeExecutor(data_context=data_context)

        # Store for prompts & evolution
        self.problem_statement = problem_statement
        self.target_metric = target_metric
        self.minimize_metric = minimize_metric

        # Resume: legacy JSON-based resume is gone; this is now a TODO for EventStore.
        start_iteration = 1
        if resume:
            print(
                "📝 Resume from previous runs is not yet implemented with the new event-store "
                "architecture. Starting fresh.\n"
            )

        # Bootstrap phase: PI explores problem and recruits team
        if not self.bootstrap_completed:
            self._bootstrap_exploration(problem_statement)
            self.bootstrap_completed = True
            print("\n" + "=" * 60)

        # Initialize evolution agent
        self._initialize_evolution_agent(problem_statement, target_metric, minimize_metric)
        print("Bootstrap complete. Starting team iterations...\n")

        # Main iteration loop
        for self.iteration in range(start_iteration, max_iterations + 1):
            print(f"\n{'=' * 60}")
            print(f"ITERATION {self.iteration}/{max_iterations}")
            print(f"{'=' * 60}\n")

            # Step 1: Team meeting to discuss approach
            approach = self._team_planning_meeting(problem_statement)

            # Step 2: Agent implements the approach (writes code)
            implementation = self._implement_approach(approach)

            # Step 3: Execute code and get results (with automatic error recovery)
            results = self._execute_with_retry(implementation, approach, max_retries=2)

            # Step 4: Extract metrics
            metrics = self._extract_metrics(results, target_metric)

            # ---- Layer 1: log raw execution event ----
            output_text = results.get("output", "") or ""
            error_text = results.get("error")
            traceback_text = results.get("traceback") or ""

            self.event_store.log_event(
                kind="execution",
                iteration=self.iteration,
                agent=self.coding_agent.title,
                payload={
                    "success": results["success"],
                    "metrics": metrics,
                    "description": results.get("description"),
                },
                large_data={
                    "output": output_text,
                    "code": implementation,
                    "traceback": traceback_text,
                },
            )

            # ---- Layer 2: semantic analysis ----
            output_analysis = self.output_analyzer.analyze(
                output=output_text,
                error=error_text,
                traceback=traceback_text,
            )
            code_analysis = self.code_analyzer.analyze(implementation)

            # Step 5.5: Reflexion - Reflect on iteration and extract learnings
            print(f"\n🤔 {self.team_lead.title} reflecting on iteration...\n")
            reflection = self._reflect_on_iteration(
                approach=approach,
                code=implementation,
                results=results,
                metrics=metrics,
            )

            # Log reflection event
            self.event_store.log_event(
                kind="reflection",
                iteration=self.iteration,
                agent=self.team_lead.title,
                payload={},
                large_data={"reflection": reflection},
            )

            # Parse and store reflection
            reflection_obj = Reflection.from_text(self.iteration, reflection)
            self.reflection_memory.add_reflection(reflection_obj)

            print("✓ Reflection recorded\n")

            # Step 6: Build IterationRecord (canonical semantic state)
            serializable_results = {
                "success": results["success"],
                "output": results["output"],
                "error": results.get("error"),
                "traceback": results.get("traceback"),
                "code": results["code"],
                "description": results["description"],
            }

            iter_record = IterationRecord(
                iteration=self.iteration,
                approach=approach,
                code=implementation,
                results=serializable_results,
                metrics=metrics,
                output_analysis=output_analysis,
                code_analysis=code_analysis,
                reflection=reflection,
            )

            self.iteration_records.append(iter_record)
            self.context_builder.set_iterations(self.iteration_records)

            # Step 7: update agents' KnowledgeBases from semantic state
            self._update_agent_knowledge_from_iteration(
                iter_record=iter_record,
                target_metric=target_metric,
                minimize=minimize_metric,
            )

            # Refine concept depths based on techniques actually used
            if self.evolution_agent is not None:
                self.evolution_agent.refine_concepts_from_code(
                    agent=self.coding_agent,
                    techniques=code_analysis.techniques,
                )

            # Optional: save a human-readable iteration summary for debugging
            iteration_summary = {
                "iteration": self.iteration,
                "approach": approach,
                "metrics": metrics,
                "agents_snapshot": [a.title for a in self.all_agents],
            }
            save_json(iteration_summary, self.results_dir / f"iteration_{self.iteration:02d}.json")

            # Update best metric BEFORE printing summary
            self._update_best_metric(metrics, target_metric, minimize_metric)

            # Iteration summary (console)
            print(f"\n{'=' * 60}")
            print(f"ITERATION {self.iteration} SUMMARY")
            print(f"{'=' * 60}")
            print(f"Status: {'✅ Success' if results['success'] else '❌ Failed'}")
            if metrics:
                for k, v in metrics.items():
                    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
            else:
                print("No metrics extracted")
            if self.best_metric is not None:
                print(f"Best {target_metric} so far: {self.best_metric:.4f}")
            print(f"{'=' * 60}\n")

            # Step 8: Check if goal achieved
            if self._check_goal_achieved(metrics, target_metric, target_score, minimize_metric):
                print(f"\n🎯 Target achieved! {target_metric}: {metrics.get(target_metric)}")
                break

            # Step 9: Evolution step
            should_evolve = self._check_mathematical_evolution(metrics, target_metric, minimize_metric)
            if should_evolve:
                self._evolve_team(problem_statement, metrics)

        # Final summary
        final_summary = self._generate_final_summary()
        save_json(final_summary, self.results_dir / "final_summary.json")

        # Persist event store (Layer 1 ground truth)
        event_store_path = self.event_store.save()
        print(f"📦 Event store saved: {event_store_path}")

        print(f"\n{'=' * 60}")
        print("✅ EXPERIMENT COMPLETE")
        print(f"{'=' * 60}")
        print(f"\nTotal Iterations: {self.iteration}")
        print(f"Best {target_metric}: {self.best_metric}")
        print(f"Results saved to: {self.results_dir}\n")

        return final_summary

    # ======================================================================
    # Bootstrap phase
    # ======================================================================

    def _bootstrap_exploration(self, problem_statement: str):
        """
        Bootstrap phase: PI explores problem and recruits team.

        The PI (team lead) starts alone, explores the data with coding agent,
        sees what the problem is about, then decides what expertise is needed
        and recruits team members.
        """
        print("\n" + "=" * 60)
        print("BOOTSTRAP: PI Initial Exploration")
        print("=" * 60)
        print(f"\n{self.team_lead.title} is exploring the problem alone...\n")

        # PI decides what initial exploration is needed
        exploration_task = f"""
You've received a new research problem. Before assembling a team, you need to understand what you're dealing with.

## Problem:
{problem_statement}

## Available Data:
{list(self.executor.data_context.keys())}

## Your Task:
Decide what initial exploration will help you understand:
1. What the data looks like (schemas, sizes, distributions)
2. What the challenge involves
3. What expertise you'll need on your team

In 2-3 sentences, describe what exploration code should be written.
"""

        meeting = IndividualMeeting(
            save_dir=str(self.results_dir / "meetings"),
            research_api=self.research.ss_api if hasattr(self, "research") else None,
        )
        exploration_plan = meeting.run(
            agent=self.team_lead,
            task=exploration_task,
            num_iterations=1,
        )

        print(f"\n{self.team_lead.title}'s exploration plan:\n{exploration_plan}\n")

        # Coding agent implements exploration
        print(f"💻 {self.coding_agent.title} implementing exploration...\n")

        code_task = f"""
The PI wants to do initial exploration. Write Python code to implement this:

## PI's Request:
{exploration_plan}

## Problem Statement (for reference):
{problem_statement}

## Available in execution context:
- Pre-imported libraries: pandas (pd), numpy (np), torch, pathlib.Path
- Variables: {list(self.executor.data_context.keys())}
  (You can use any of these variables directly in your code)

## Requirements:
- Inspect dataframes: print(df.info()), df.head(), df.describe(), df.columns
- ONLY print what you observe - no summaries, interpretations, or conclusions
- Use variables from "Available in execution context" above
- Suppress warnings if needed

Output ONLY the Python code, wrapped in ```python code blocks.
"""

        code_meeting = IndividualMeeting(
            save_dir=str(self.results_dir / "meetings"),
            research_api=self.research.ss_api if hasattr(self, "research") else None,
        )
        code_output = code_meeting.run(
            agent=self.coding_agent,
            task=code_task,
            num_iterations=1,
            use_react_coding=True,
        )

        code = extract_code_from_text(code_output)

        # Save exploration code
        code_file = self.results_dir / "code" / "iteration_00.py"
        code_file.parent.mkdir(exist_ok=True)
        code_file.write_text(code)

        # Execute exploration with retry on failure
        print("⚙️  Executing exploration...\n")
        results = self._execute_with_retry(
            code=code,
            approach=exploration_plan,
            max_retries=2,
        )

        # Log bootstrap execution as an event
        self.event_store.log_event(
            kind="execution",
            iteration=0,
            agent=self.coding_agent.title,
            payload={
                "phase": "bootstrap",
                "success": results["success"],
            },
            large_data={
                "output": results.get("output", "") or "",
                "code": code,
                "traceback": results.get("traceback") or "",
            },
        )

        if results["success"]:
            print("✅ Exploration successful!\n")
            print("Output:")
            print("-" * 60)
            print(results["output"])
            print("-" * 60)

            # Extract column schemas from explored dataframes
            print("\n📋 Extracting column schemas...")
            column_schemas = {}
            for df_name in ["batches_train", "batches_test", "products", "sites", "regions"]:
                df = self.executor.get_variable(df_name)
                if df is not None and hasattr(df, "columns"):
                    column_schemas[df_name] = list(df.columns)
                    print(f"   {df_name}: {len(df.columns)} columns - {list(df.columns)[:10]}...")

            self.column_schemas = column_schemas
        else:
            print("❌ Exploration failed after retries:")
            print(results["error"])

        # PI reviews results and recruits team using ReAct
        print(f"\n{self.team_lead.title} reviewing exploration results and recruiting team...\n")

        recruitment_task = f"""
Based on the problem and exploration results, decide what expertise you need on your team.

## Problem:
{problem_statement}

## Exploration Results:
{results['output'][:2000] if results['success'] else "Exploration failed, but you have the problem statement."}

## Your Task:
List 1-3 team members you want to recruit. For each, provide:
- Title (e.g., "ML Strategist", "Domain Expert", "Data Analyst")
- Expertise (what they should know)
- Role (what they'll contribute)

Be specific about the skills needed based on what you learned from exploration and research papers.

Format your response as a simple list, one team member per line.
"""

        recruitment_meeting = IndividualMeeting(
            save_dir=str(self.results_dir / "meetings"),
            research_api=self.research.ss_api if hasattr(self, "research") else None,
        )
        recruitment_plan = recruitment_meeting.run(
            agent=self.team_lead,
            task=recruitment_task,
            num_iterations=1,
            use_react=True,
        )

        print(f"Recruitment plan:\n{recruitment_plan}\n")

        recruited_agents = self._parse_and_recruit(recruitment_plan)

        self.team_members.extend(recruited_agents)
        self.all_agents = [self.team_lead] + self.team_members

        print(f"\n✅ Team assembled! {len(recruited_agents)} member(s) recruited:")
        for agent in recruited_agents:
            print(f"   - {agent.title}")

        # Optional: Save bootstrap summary for human inspection
        bootstrap_summary = {
            "iteration": 0,
            "phase": "bootstrap",
            "approach": exploration_plan,
            "results": {
                "success": results["success"],
                "output": results["output"] if results["success"] else results.get("error", ""),
                "error": results.get("error"),
                "traceback": results.get("traceback"),
                "code": code,
                "description": "Bootstrap exploration",
            },
            "metrics": {},
            "agents_snapshot": [self.team_lead.title, self.coding_agent.title],
            "recruitment_plan": recruitment_plan,
            "recruited_agents": [
                {
                    "title": a.title,
                    "expertise": a.expertise,
                    "role": a.role,
                    "goal": a.goal,
                }
                for a in recruited_agents
            ],
        }
        save_json(bootstrap_summary, self.results_dir / "iteration_00_bootstrap.json")

    # ======================================================================
    # Recruitment parsing, team meeting, coding, execution, error fixing
    # (unchanged in spirit, but now using ContextBuilder + no experiment_history)
    # ======================================================================

    def _parse_and_recruit(self, recruitment_plan: str) -> List[Agent]:
        """
        Parse PI's recruitment plan and create agents.

        Uses LLM to extract agent specifications from PI's plan.
        """
        # Use LLM to parse the recruitment plan and extract agent definitions
        parse_task = f"""
Parse this recruitment plan and extract agent specifications.

## Recruitment Plan:
{recruitment_plan}

## Your Task:
Extract the team members mentioned in the plan.
For each, provide:
- Title
- Expertise
- Role

Output ONLY a JSON list of objects, like this:
[
  {{
    "title": "Agent Title",
    "expertise": "Expertise description",
    "role": "Role description"
  }}
]
"""

        try:
            parsed_agents = self.llm.generate_json(parse_task, temperature=0.1)
            # Handle case where LLM returns a dict instead of list (e.g. {"agents": [...]})
            if isinstance(parsed_agents, dict):
                if "agents" in parsed_agents:
                    parsed_agents = parsed_agents["agents"]
                else:
                    # Try to find any list value
                    for val in parsed_agents.values():
                        if isinstance(val, list):
                            parsed_agents = val
                            break
            
            if not isinstance(parsed_agents, list):
                parsed_agents = []

        except Exception as e:
            print(f"   ⚠️  Parsing failed: {e}")
            parsed_agents = []

        # Create Agent objects
        agents = []
        for agent_data in parsed_agents:
            if isinstance(agent_data, dict) and "title" in agent_data and "expertise" in agent_data:
                agent = Agent(
                    title=agent_data.get("title", "Specialist"),
                    expertise=agent_data.get("expertise", "General expertise"),
                    goal=f"contribute specialized expertise to optimize the target metric",
                    role=agent_data.get("role", "Contribute to team success")
                )
                agents.append(agent)

        # Fallback: if parsing failed, create a generic ML specialist
        if not agents:
            print("   ⚠️  Could not parse recruitment plan, creating default ML Strategist")
            agents = [Agent(
                title="ML Strategist",
                expertise="machine learning, feature engineering, model selection, predictive modeling",
                goal="design effective predictive approaches",
                role="propose modeling strategies and analytical approaches"
            )]

        return agents

    def _team_planning_meeting(self, problem_statement: str) -> str:
        """Run team meeting to plan approach"""
        print("👥 Team planning meeting...\n")

        base_context = self.context_builder.for_team_meeting(
            target_metric=self.target_metric,
        )

        columns_summary = ""
        if self.column_schemas:
            columns_summary = "\n## AVAILABLE COLUMNS (use EXACT column names):\n"
            for df_name, cols in self.column_schemas.items():
                columns_summary += f"\n{df_name}: {cols}\n"

        agenda = f"""
**BE CONCISE.**

## Problem:
{problem_statement}

## Available Dataframes:
{list(self.executor.data_context.keys())}
{columns_summary}

{base_context}

## Roles:
- **Team Members**: Review recent iterations and propose what to do next.
- **Lead**: Synthesize into a decisive action plan.
"""

        meeting = TeamMeeting(
            save_dir=str(self.results_dir / "meetings"),
            research_api=self.research.ss_api if hasattr(self, "research") else None,
        )
        summary = meeting.run(
            team_lead=self.team_lead,
            team_members=self.team_members,
            agenda=agenda,
            num_rounds=1,
        )

        meeting.save(f"iteration_{self.iteration:02d}_team_meeting.json")

        self.event_store.log_event(
            kind="meeting",
            iteration=self.iteration,
            agent=self.team_lead.title,
            payload={"type": "team"},
        )

        return summary.get("summary", summary)

    def _implement_approach(self, approach: str) -> str:
        """Have coding agent write code to implement the approach"""
        print(f"💻 {self.coding_agent.title} implementing approach...\n")

        schema_info = self._format_column_schemas()

        impl_context = self.context_builder.for_coding(
            coding_agent=self.coding_agent,
            team_plan=approach,
            target_metric=self.target_metric,
            column_schemas=schema_info
        )

        task = f"""
Implement the team's plan.

{impl_context}

## EXECUTION ENVIRONMENT:
- The following variables are PRE-LOADED in the global scope: {list(self.executor.data_context.keys())}
- You MUST use these existing variables.
- **CRITICAL: DO NOT define any of the variables listed above. They already exist.**
- **CRITICAL: DO NOT generate dummy data. It will overwrite the real data.**
- Just start using the variables directly.

## XGBOOST NOTE:
- Environment: XGBoost 3.1.1 with CUDA.
- Do NOT use `tree_method="gpu_hist"` (it is invalid here).
- If you use XGBRegressor, use `device="cuda"` and optionally `tree_method="hist"` or leave `tree_method` default.

## Requirements:
- Use the EXACT column names from DataFrame Schemas above
- Use GPU when training models
- Import what you need (standard libraries)
- Compute {self.target_metric} and store in a variable if training a model
- Print important outputs: metrics, feature importance, model summaries
- Save trained models (joblib.dump, torch.save) if training is expensive
- Suppress verbose output (warnings.filterwarnings('ignore'), verbose=0/-1)

Output ONLY Python code in ```python``` blocks.
"""

        meeting = IndividualMeeting(
            save_dir=str(self.results_dir / 'meetings'),
            research_api=self.research.ss_api if hasattr(self, 'research') else None
        )
        code_output = meeting.run(
            agent=self.coding_agent,
            task=task,
            num_iterations=1,
            use_react_coding=True
        )

        meeting.save(f'iteration_{self.iteration:02d}_coding.json')

        self.event_store.log_event(
            kind="meeting",
            iteration=self.iteration,
            agent=self.coding_agent.title,
            payload={"type": "coding"}
        )

        code = extract_code_from_text(code_output)

        code_file = self.results_dir / 'code' / f'iteration_{self.iteration:02d}.py'
        code_file.parent.mkdir(exist_ok=True)
        code_file.write_text(code)

        return code

    def _execute_implementation(self, code: str) -> Dict[str, Any]:
        """Execute the generated code"""
        print("⚙️ Executing implementation...\n")

        result = self.executor.execute(
            code=code,
            description=f"Iteration {self.iteration} implementation"
        )

        if not result['success']:
            print(f"   ❌ Execution failed: {result['error']}\n")
            print(f"   Traceback:\n{result['traceback']}\n")

        return result

    def _execute_with_retry(self, code: str, approach: str, max_retries: int = 2) -> Dict[str, Any]:
        """
        Execute code with automatic error recovery.

        If execution fails, give the error to the agent and ask for a fix.
        Retry up to max_retries times.

        Args:
            code: Initial code to execute
            approach: The approach description (for context)
            max_retries: Maximum number of retry attempts

        Returns:
            Execution results (final attempt)
        """
        current_code = code
        attempt = 0

        while attempt <= max_retries:
            if attempt > 0:
                print(f"   🔄 Retry attempt {attempt}/{max_retries}\n")

            # Execute code
            result = self._execute_implementation(current_code)

            # If successful, return
            if result['success']:
                if attempt > 0:
                    print(f"   ✅ Fixed after {attempt} attempt(s)!\n")
                return result

            # Check if failure was due to missing package - install and retry automatically
            # This doesn't count against retry limit - it's just installing a dependency
            if 'missing_package' in result:
                package = result['missing_package']
                print(f"   📦 Missing package detected: {package}")
                if self.executor._install_package(package):
                    print(f"   🔄 Retrying after installing {package}...\n")
                    continue  # Retry with same code after installation (doesn't increment attempt)
                else:
                    print(f"   ⚠️ Failed to install {package}, asking agent to use alternative...\n")

            # If failed and we have retries left, ask agent to fix
            if attempt < max_retries:
                print(f"   ❌ Error: {result['error']}")
                print(f"   🔧 Asking agent to fix...\n")
                current_code = self._fix_code_error(
                    failed_code=current_code,
                    error=result['error'],
                    traceback=result.get('traceback', ''),
                    approach=approach
                )

                # Save the fixed code attempt
                code_file = self.results_dir / 'code' / f'iteration_{self.iteration:02d}_retry_{attempt+1}.py'
                code_file.parent.mkdir(exist_ok=True)
                code_file.write_text(current_code)
                print(f"   Fixed code saved to: {code_file}\n")

            attempt += 1
        # Max retries exhausted, return last failed result
        print(f"   ⚠️ Max retries ({max_retries}) exhausted. Moving on with failure.\n")
        return result

    def _fix_code_error(self, failed_code: str, error: str, traceback: str, approach: str) -> str:
        """
        Ask coding agent to fix code that failed execution.

        Args:
            failed_code: The code that failed
            error: Error message
            traceback: Full traceback
            approach: Original approach description

        Returns:
            Fixed code
        """
        print(f"   🔧 {self.coding_agent.title} fixing error...\n")

        fix_context = self.context_builder.for_error_fix(
            coding_agent=self.coding_agent,
            failed_code=failed_code,
            error=error,
            traceback=traceback,
            approach=approach
        )

        preloaded_vars = list(self.executor.data_context.keys())

        task = f"""
Your code failed with an error. Fix it.

{fix_context}

## Available in execution context:
- Pre-imported libraries: pandas, numpy, torch, pathlib
- Variables: {preloaded_vars}

## Constraints:
- Keep using these existing dataframes; do NOT recreate or overwrite them with dummy data
  (e.g. no "batches_train = pd.DataFrame(...)" or similar).
- If the error involves XGBoost and `tree_method="gpu_hist"`, fix it by using `device="cuda"`
  and/or a valid `tree_method` like `"hist"`, not by changing the data.

## Task:
Read the traceback carefully and fix the specific lines that failed.
Do NOT just repeat the same code.

Output ONLY the FIXED Python code in ```python``` blocks.
"""

        meeting = IndividualMeeting(
            save_dir=str(self.results_dir / 'meetings'),
            research_api=self.research.ss_api if hasattr(self, 'research') else None
        )
        code_output = meeting.run(
            agent=self.coding_agent,
            task=task,
            num_iterations=1,
            use_react_coding=True  # Bug fixing uses internal reasoning, not paper search
        )

        # Extract fixed code
        fixed_code = extract_code_from_text(code_output)

        return fixed_code

    # ======================================================================
    # Metrics, reflection, evolution, summary
    # ======================================================================

    def _format_column_schemas(self) -> str:
        """Format column schemas for display in prompts"""
        if not hasattr(self, 'column_schemas') or not self.column_schemas:
            return "No schema information available."

        result = ""
        for df_name, cols in self.column_schemas.items():
            result += f"{df_name}: {cols}\n"
        return result

    def _extract_metrics(self, results: Dict[str, Any], target_metric: str) -> Dict[str, float]:
        """Extract metrics from execution results, ensuring JSON-serializable values only"""
        raw_metrics = results.get('metrics', {})

        # Filter to only keep JSON-serializable numeric values
        metrics = {}
        for key, value in raw_metrics.items():
            try:
                # Only keep simple numeric types
                if isinstance(value, (int, float, np.integer, np.floating)):
                    metrics[key] = float(value)
                elif isinstance(value, (list, np.ndarray)):
                    # For arrays, take the mean
                    metrics[key] = float(np.mean(value))
            except (TypeError, ValueError, AttributeError):
                # Skip non-numeric or non-serializable values
                pass

        # Try to find target metric in variables if not in metrics
        if target_metric not in metrics:
            for key, value in results.get('variables', {}).items():
                if target_metric in key.lower():
                    try:
                        # Handle arrays (take mean)
                        if hasattr(value, '__iter__') and not isinstance(value, str):
                            metrics[target_metric] = float(np.mean(value))
                        else:
                            metrics[target_metric] = float(value)
                        break
                    except (TypeError, ValueError, AttributeError):
                        pass

        return metrics

    def _reflect_on_iteration(
        self,
        approach: str,
        code: str,
        results: Dict[str, Any],
        metrics: Dict[str, float],
    ) -> str:
        """
        Generate self-reflection on iteration (Reflexion: Shinn et al.)

        After each iteration, explicitly analyze what happened and extract
        structured learnings to guide future iterations.

        Args:
            approach: What was attempted
            code: Implementation
            results: Execution results
            metrics: Extracted metrics

        Returns:
            Reflection text with structured sections
        """
        # Prepare output for reflection (summarize if too long)
        output = results.get('output', '')
        if len(output) > 2000:
            output_preview = output[:2000] + "\n... (truncated)"
        else:
            output_preview = output

        error_section = ""
        if not results.get('success'):
            error_section = f"""
## Error
{results.get('error', 'Unknown error')}

## Traceback
{results.get('traceback', 'No traceback available')[:500]}
"""

        reflection_prompt = f"""You are reviewing iteration {self.iteration} of this research experiment.

## What Was Attempted
{approach}

## Code Implementation
```python
{code[:500]}...
```

## What Happened
Success: {results.get('success')}
Metrics: {metrics}

## Execution Output
{output_preview}
{error_section}

## Reflection Task
Analyze this iteration and extract concrete, actionable learnings.

Answer these questions:
1. **What worked?** - Specific techniques, patterns, or approaches that succeeded
2. **What failed?** - Specific mistakes, wrong assumptions, or bugs encountered
3. **Why did it fail?** - Root cause analysis (not just symptoms)
4. **What to try differently?** - Concrete suggestions for the next iteration
5. **What to avoid?** - Dead ends or approaches that won't work for this problem

Be specific and actionable. Focus on insights the team can actually use.

Format your response with these exact section headers:
**Worked**: [what succeeded]
**Failed**: [what didn't work]
**Why**: [root cause]
**Try next**: [specific suggestions]
**Avoid**: [what not to do]
"""

        reflection = self.llm.generate(
            reflection_prompt,
            system_instruction=self.team_lead.prompt,
            temperature=0.3  # More focused for reflection
        )

        return reflection

    def _check_goal_achieved(
        self,
        metrics: Dict[str, float],
        target_metric: str,
        target_score: Optional[float],
        minimize: bool,
    ) -> bool:
        """Check if target score achieved"""
        if not target_score or target_metric not in metrics:
            return False

        current = metrics[target_metric]

        if minimize:
            return current <= target_score
        else:
            return current >= target_score

    def _update_best_metric(
        self,
        metrics: Dict[str, float],
        target_metric: str,
        minimize: bool,
    ):
        """Update best metric seen so far"""
        if target_metric not in metrics:
            return

        current = metrics[target_metric]

        if self.best_metric is None:
            self.best_metric = current
        elif minimize and current < self.best_metric:
            self.best_metric = current
            print(f"\n✨ New best {target_metric}: {current:.4f}")
        elif not minimize and current > self.best_metric:
            self.best_metric = current
            print(f"\n✨ New best {target_metric}: {current:.4f}")

    def _generate_final_summary(self) -> Dict[str, Any]:
        """Generate final experiment summary based on semantic state"""
        return {
            "total_iterations": self.iteration,
            "best_metric": self.best_metric,
            "final_team": [
                {
                    "title": a.title,
                    "expertise": a.expertise,
                    "specialization_depth": a.specialization_depth,
                }
                for a in self.all_agents
            ],
            # Layer 2 semantic history
            "iterations": [rec.to_dict() for rec in self.iteration_records],
            # Executor-level summary from CodeExecutor
            "execution_summary": self.executor.summary() if self.executor else {},
        }

    # ========== Evolution Methods ==========

    def _initialize_evolution_agent(self, problem_statement: str, target_metric: str, minimize_metric: bool):
        """Initialize the evolution agent"""
        print("\n🧬 Initializing Evolution Agent...")
        
        self.evolution_agent = EvolutionAgent(
            llm=self.llm,
            target_team_size=(3, 6) # Default constraint
        )
        
        self.evolution_agent.initialize(
            agents=self.all_agents,
            problem_statement=problem_statement,
            target_metric=target_metric,
            minimize_metric=minimize_metric
        )
        print("   ✓ Evolution agent initialized")

    def _check_mathematical_evolution(
        self,
        metrics: Dict[str, float],
        target_metric: str,
        minimize: bool,
    ) -> bool:
        """Check if evolution is needed using EvolutionAgent"""
        if self.evolution_agent is None:
            return False

        current_val = metrics.get(target_metric)
        if current_val is None:
            return False

        # Record metric for this iteration (Layer 5 math state)
        self.evolution_agent.record_metric(current_val)

        print("\n🧠 Evolution Agent analyzing team dynamics...")
        self.evolution_decision = self.evolution_agent.step()
        decision = self.evolution_decision

        # Update ContextBuilder with coverage/gaps for next prompts
        coverage, gaps = self.evolution_agent.get_coverage_and_gaps()
        self.context_builder.set_evolution_state(gaps=gaps, coverage=coverage)

        print(f"   Quality: {decision.quality:.2f}")
        print(
            f"   Team size: {decision.debug_info.get('team_size', '?')} "
            f"(min: {decision.debug_info.get('min_size', '?')}, "
            f"max: {decision.debug_info.get('max_size', '?')})"
        )
        print(f"   Coverage gaps: {len(decision.debug_info.get('gaps', []))} concepts below threshold")
        if decision.debug_info.get("gaps"):
            print(f"      → {', '.join(decision.debug_info['gaps'][:5])}")

        if decision.new_agent_specs:
            print(f"   🔔 Proposed new agents: {len(decision.new_agent_specs)}")
            for spec in decision.new_agent_specs:
                print(f"      - {spec.kind.upper()}: {spec.title}")
                print(f"        Focus: {', '.join(spec.focus_concepts[:3])}")

        if decision.agents_to_delete:
            print(f"   🔔 Proposed deletions: {len(decision.agents_to_delete)}")
            for agent in decision.agents_to_delete:
                print(f"      - {agent.title}")

        if not decision.new_agent_specs and not decision.agents_to_delete:
            print("   ℹ️  No evolution needed:")
            if decision.debug_info.get("note"):
                print(f"      → {decision.debug_info['note']}")
            else:
                print("      → No coverage gaps and no weak overlap detected")

        return bool(decision.new_agent_specs or decision.agents_to_delete)

    def _evolve_team(self, problem_statement: str, current_metrics: Dict[str, float]):
        """Execute the evolution decision"""
        if not self.evolution_decision:
            return
            
        print("\n🧬 Evolving team composition...\n")
        decision = self.evolution_decision
        changes_made = []
        
        # 1. Handle deletions
        for agent_to_remove in decision.agents_to_delete:
            # Don't remove the lead or coding agent!
            if agent_to_remove == self.team_lead or agent_to_remove == self.coding_agent:
                print(f"   ⚠️  Skipping removal of critical agent: {agent_to_remove.title}")
                continue
                
            if agent_to_remove in self.team_members:
                self.team_members.remove(agent_to_remove)
                changes_made.append(f"❌ Removed {agent_to_remove.title}")
                
        # 2. Handle new agents
        for spec in decision.new_agent_specs:
            new_agent = Agent(
                title=spec.title,
                expertise=spec.expertise,
                goal=f"optimize {list(current_metrics.keys())[0] if current_metrics else 'metrics'}",
                role=spec.role
            )

            # Seed KB using relevant past iterations
            if self.evolution_agent and self.iteration_records:
                self.evolution_agent.seed_agent_knowledge(
                    agent=new_agent,
                    iteration_records=self.iteration_records,
                    focus_concepts=spec.focus_concepts
                )

            self.team_members.append(new_agent)
            changes_made.append(f"✅ Added {new_agent.title} ({spec.kind})")
            
        # Update all_agents list
        self.all_agents = [self.team_lead] + self.team_members
        
        # Sync new team state to evolution agent
        self.evolution_agent.update_team(self.all_agents)
        
        print("\n🔄 Team Evolution Complete:")
        for change in changes_made:
            print(f"   {change}")

        print(f"\n👥 New team composition:")
        print(f"   - {self.team_lead.title} (Lead)")
        for agent in self.team_members:
            print(f"   - {agent.title}")
        print()

        # Save evolution record
        evolution_record = {
            'iteration': self.iteration,
            'decision': {
                'quality': decision.quality,
                'new_specs': [str(s) for s in decision.new_agent_specs],
                'deletions': [str(a) for a in decision.agents_to_delete],
                'debug': decision.debug_info
            },
            'changes': changes_made,
            'new_team': [{'title': a.title, 'expertise': a.expertise} for a in self.all_agents]
        }
        save_json(evolution_record, self.results_dir / f'evolution_iter_{self.iteration}.json')

        self.event_store.log_event(
            kind="evolution",
            iteration=self.iteration,
            agent=None,
            payload=evolution_record
        )

    def _update_agent_knowledge_from_iteration(
        self,
        iter_record: IterationRecord,
        target_metric: str,
        minimize: bool,
    ):
        """Update agent knowledge bases using structured methods from IterationRecord."""
        # Determine if this was an improvement
        prev_metric = None
        if len(self.iteration_records) > 1:
            prev_iter = self.iteration_records[-2]
            prev_metric = prev_iter.get_metric(target_metric)
        
        current_metric = iter_record.get_metric(target_metric)
        
        is_improvement = False
        if prev_metric is not None and current_metric is not None:
            is_improvement = (current_metric < prev_metric) if minimize else (current_metric > prev_metric)
        
        # Update all agents
        for agent in self.all_agents:
            # Add techniques from code analysis
            for technique in iter_record.code_analysis.techniques:
                agent.knowledge_base.add_technique(technique)
            
            # Track success/failure patterns
            if is_improvement and current_metric is not None and prev_metric is not None:
                improvement = abs(current_metric - prev_metric)
                technique = iter_record.code_analysis.techniques[0] if iter_record.code_analysis.techniques else "approach"
                agent.knowledge_base.add_success_pattern(
                    iteration=iter_record.iteration,
                    technique=technique,
                    metric=target_metric,
                    improvement=improvement
                )
            elif not is_improvement and prev_metric is not None:
                technique = iter_record.code_analysis.techniques[0] if iter_record.code_analysis.techniques else "approach"
                agent.knowledge_base.add_failure_pattern(
                    iteration=iter_record.iteration,
                    technique=technique,
                    metric=target_metric,
                    reason=iter_record.output_analysis.raw_summary
                )
            
            # Track errors
            for error in iter_record.output_analysis.errors:
                error_type = "ExecutionError"
                if "NameError" in error:
                    error_type = "NameError"
                elif "KeyError" in error:
                    error_type = "KeyError"
                elif "ValueError" in error:
                    error_type = "ValueError"
                
                agent.knowledge_base.add_error_insight(
                    iteration=iter_record.iteration,
                    error_type=error_type,
                    error_msg=error[:200]
                )
        
        # Log KB update event
        if self.event_store:
            self.event_store.log_event(
                kind="kb_update",
                iteration=iter_record.iteration,
                payload={
                    "techniques_added": iter_record.code_analysis.techniques,
                    "is_improvement": is_improvement
                }
            )
