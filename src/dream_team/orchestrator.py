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
from .evolution_policy import DefaultEvolutionPolicy, EvolutionPolicyConfig, AgentMeta
from .research import get_research_assistant
from .utils import save_json
from .context import Reflection, ReflectionMemory
from .event_store import EventStore
from .analyzers import OutputAnalyzer, CodeAnalyzer
from .semantic_state import IterationRecord, OutputAnalysis
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
        interactive_mode: bool = False,
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

        self.interactive_mode = interactive_mode
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
        # CodeAnalyzer removed as per user request - relying on output analysis
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
        interactive_mode: Optional[bool] = None,
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
        
        # Connect executor and artifacts_dir to context builder
        self.context_builder.executor = self.executor
        self.context_builder.artifacts_dir = artifacts_dir

        # Store for prompts & evolution
        self.problem_statement = problem_statement
        self.target_metric = target_metric
        self.minimize_metric = minimize_metric
        
        if interactive_mode is not None:
            self.interactive_mode = interactive_mode

        # Resume logic
        start_iteration = 1
        if resume:
            try:
                restored_iter = self._restore_state()
                if restored_iter > 0:
                    start_iteration = restored_iter + 1
                    print(f"✅ Resumed from iteration {restored_iter}")
                    # If we restored, bootstrap is definitely done
                    self.bootstrap_completed = True
            except Exception as e:
                print(f"⚠️  Resume failed: {e}")
                print("   Starting fresh...")
                start_iteration = 1

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
            
            # Update implementation with the code that was actually executed (in case of fixes)
            implementation = results.get('code', implementation)

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
            # code_analysis = self.code_analyzer.analyze(implementation) # Removed

            # Step 5.5: Reflexion - Reflect on iteration and extract learnings
            print(f"\n🤔 {self.team_lead.title} reflecting on iteration...\n")
            reflection = self._reflect_on_iteration(
                approach=approach,
                code=implementation,
                results=results,
                metrics=metrics,
                output_analysis=output_analysis,
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
                code_analysis=None,
                reflection=reflection,
            )

            self.iteration_records.append(iter_record)
            self.context_builder.set_iterations(self.iteration_records)

            # Step 6.5: Refine concept space based on iteration learnings
            if self.evolution_agent is not None and self.iteration > 0:
                # Determine if metric improved
                improved = False
                if len(self.iteration_records) > 1:
                    prev_metric = self.iteration_records[-2].metrics.get(target_metric)
                    curr_metric = metrics.get(target_metric)
                    if prev_metric is not None and curr_metric is not None:
                        if minimize_metric:
                            improved = curr_metric < prev_metric
                        else:
                            improved = curr_metric > prev_metric
                
                self.evolution_agent.refine_concept_space(
                    problem_statement=self.problem_statement,
                    target_metric=target_metric,
                    last_approach=approach,
                    last_metric=metrics.get(target_metric, 0.0),
                    improved=improved,
                    reflection_text=reflection,
                    iteration=self.iteration
                )

            # Step 7: update agents' KnowledgeBases from semantic state
            self._update_agent_knowledge_from_iteration(
                iter_record=iter_record,
                target_metric=target_metric,
                minimize=minimize_metric,
            )

            # Refine concept depths based on techniques actually used
            # (Technique tracking removed as per user request)
            if self.evolution_agent is not None:
                pass

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

        # Analyze exploration results to provide structured insights
        print("\n🔍 Analyzing exploration results...")
        exploration_analysis = self.output_analyzer.analyze(
            output=results['output'] if results['success'] else results.get('error', ''),
            error=results.get('error'),
            traceback=results.get('traceback')
        )

        # PI reviews results and recruits team using ReAct
        print(f"\n{self.team_lead.title} reviewing exploration results and recruiting team...\n")

        recruitment_task = f"""
Based on the problem and exploration results, decide what expertise you need on your team.

## Problem:
{problem_statement}

## Exploration Insights:
**Summary:**
{exploration_analysis.raw_summary}

**Key Observations:**
{chr(10).join([f"- {obs}" for obs in exploration_analysis.key_observations])}

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
        parse_task = f"""Parse this recruitment plan into structured JSON.

Recruitment Plan:
{recruitment_plan}

Output valid JSON only, no other text:
[
  {{"title": "...", "expertise": "...", "role": "..."}}
]

Rules:
- Extract all team members mentioned.
- All three fields required for each member.
- No markdown, no ```json blocks.
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
            columns_summary = "\n## AVAILABLE COLUMNS (CRITICAL: Use EXACT column names):\n"
            for df_name, cols in self.column_schemas.items():
                columns_summary += f"\n{df_name}: {cols}\n"

        agenda = f"""
**BE CONCISE.**

## Problem:
{problem_statement}

## Objective:
Decide on the next step to improve {self.target_metric}.

## Constraints:
- Be specific and actionable.
- You don't have to solve the entire problem in one iteration. Go step by step, adapt as new information comes in.

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

        task = f"""Implement the team's plan.

{impl_context}

## EXECUTION ENVIRONMENT
Pre-loaded variables (use directly, do NOT redefine):
{list(self.executor.data_context.keys())}

## CRITICAL RULES
1. Do NOT create, define, or overwrite the pre-loaded variables.
2. Do NOT generate dummy/mock data.
3. Use EXACT column names from the schema above.

## REQUIREMENTS
- Import needed libraries at the top.
- Use GPU when training models.
- Compute and print {self.target_metric}.
- Save expensive models (joblib.dump, torch.save).
- Suppress verbose output (warnings.filterwarnings('ignore')).

Output ONLY Python code in ```python blocks.
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
        execution_history = []

        while attempt <= max_retries:
            if attempt > 0:
                print(f"   🔄 Retry attempt {attempt}/{max_retries}\n")

            # Execute code
            result = self._execute_implementation(current_code)
            
            # Store attempt info
            attempt_info = {
                'attempt': attempt + 1,
                'code': current_code,
                'success': result['success'],
                'error': result.get('error'),
                'output': result.get('output', '')
            }
            execution_history.append(attempt_info)

            # If successful, return
            if result['success']:
                if attempt > 0:
                    print(f"   ✅ Fixed after {attempt} attempt(s)!\n")
                result['execution_history'] = execution_history
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

                # Interactive mode: Give user a choice
                if self.interactive_mode:
                    print("\n⚠️  Execution failed.")
                    print("Options:")
                    print("  [A]uto-fix (Agent tries to fix)")
                    print("  [M]anual fix (You edit the code)")
                    print("  [C]ontinue with failure (Skip)")
                    
                    try:
                        choice = input("Select option [A/m/c]: ").lower().strip()
                    except EOFError:
                        choice = 'a'
                    
                    if choice == 'm':
                        print("   📝 Opening code for manual edit...")
                        # Save to specific retry file
                        retry_count = attempt + 1
                        manual_file = self.results_dir / "code" / f"iteration_{self.iteration:02d}_retry_{retry_count}.py"
                        manual_file.parent.mkdir(exist_ok=True)
                        manual_file.write_text(current_code)
                        print(f"   Saved to: {manual_file}")
                        print(f"   Please edit the file and save it.")
                        try:
                            input("   Press Enter when ready...")
                        except EOFError:
                            pass
                        
                        if manual_file.exists():
                            current_code = manual_file.read_text()
                            print(f"   ✅ Loaded manual fix from {manual_file.name}. Retrying...")
                            attempt += 1
                            continue
                        else:
                            print("   ⚠️ File not found. Proceeding with auto-fix.")
                    
                    elif choice == 'c':
                        print("   Skipping retry...")
                        result['execution_history'] = execution_history
                        return result
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
                
                # Also save manual fix if it was manual
                if self.interactive_mode and choice == 'm':
                     # It's already saved to manual_fix.py, but let's ensure we track it as a retry
                     pass

            attempt += 1
        # Max retries exhausted, return last failed result
        print(f"   ⚠️ Max retries ({max_retries}) exhausted. Moving on with failure.\n")
        result['execution_history'] = execution_history
        return result

    def _restore_state(self) -> int:
        """
        Restore experiment state from event store.
        
        Replays events to:
        1. Restore team composition
        2. Restore execution state (variables) by re-executing code
        3. Restore semantic state (IterationRecords, Reflections)
        4. Restore EvolutionAgent state
        
        Returns:
            Last completed iteration number
        """
        print("\n🔄 Resuming experiment...")
        
        # 1. Load events
        if not self.event_store.events:
            # Try to load from disk
            event_file = self.results_dir / "events" / self.results_dir.name / "events.json"
            if event_file.exists():
                print(f"   Loading events from {event_file}...")
                self.event_store = EventStore.load(event_file, self.results_dir / "events")
            else:
                print("   No event history found.")
                return 0
                
        events = self.event_store.events
        if not events:
            return 0
            
        last_iteration = 0
        
        # 2. Replay Bootstrap (Iteration 0)
        print("   Replaying bootstrap...")
        bootstrap_events = [e for e in events if e.iteration == 0]
        
        # Restore team from bootstrap recruitment
        # We need to find the recruitment event or infer from subsequent events
        # Actually, let's look for the 'evolution' events or just rely on the fact 
        # that we save iteration_00_bootstrap.json which has the team
        bootstrap_file = self.results_dir / "iteration_00_bootstrap.json"
        if bootstrap_file.exists():
            import json
            with open(bootstrap_file, 'r') as f:
                data = json.load(f)
                
            # Restore team members
            if "recruited_agents" in data:
                self.team_members = []
                for agent_data in data["recruited_agents"]:
                    agent = Agent(
                        title=agent_data["title"],
                        expertise=agent_data["expertise"],
                        goal=agent_data.get("goal", "contribute to team success"),
                        role=agent_data.get("role", "team member")
                    )
                    self.team_members.append(agent)
                self.all_agents = [self.team_lead] + self.team_members
                print(f"   Restored {len(self.team_members)} team members.")
                
            # Restore column schemas if possible (usually re-execution handles this, 
            # but bootstrap re-execution is safer)
            
        # Re-execute bootstrap code to restore variables
        bootstrap_exec = [e for e in bootstrap_events if e.kind == "execution" and e.payload.get("success")]
        if bootstrap_exec:
            # Take the last successful bootstrap execution
            evt = bootstrap_exec[-1]
            code = self.event_store.get_blob_content(evt, "code")
            if code:
                print("   Re-executing bootstrap code...")
                self.executor.execute(code, description="Bootstrap Replay", silent=True)
                
                # Re-extract schemas
                for df_name in ["batches_train", "batches_test", "products", "sites", "regions"]:
                    df = self.executor.get_variable(df_name)
                    if df is not None and hasattr(df, "columns"):
                        self.column_schemas[df_name] = list(df.columns)
        
        # 3. Replay Iterations
        # Group events by iteration
        iter_events = {}
        for e in events:
            if e.iteration > 0:
                if e.iteration not in iter_events:
                    iter_events[e.iteration] = []
                iter_events[e.iteration].append(e)
                
        sorted_iters = sorted(iter_events.keys())
        
        for i in sorted_iters:
            print(f"   Replaying iteration {i}...")
            evts = iter_events[i]
            
            # A. Replay Execution (Restore Memory)
            # Find the LAST successful execution event for this iteration
            exec_evts = [e for e in evts if e.kind == "execution" and e.payload.get("success")]
            if exec_evts:
                last_exec = exec_evts[-1]
                code = self.event_store.get_blob_content(last_exec, "code")
                if code:
                    print(f"      Re-executing code (silent)...")
                    self.executor.execute(code, description=f"Iteration {i} Replay", silent=True)
            
            # B. Replay Evolution (Team Changes)
            evo_evts = [e for e in evts if e.kind == "evolution"]
            for evo in evo_evts:
                payload = evo.payload
                # Apply deletions
                if "decision" in payload and "deletions" in payload["decision"]:
                    # This is tricky because we stored string representations or objects
                    # Let's rely on the 'new_team' snapshot if available
                    pass
                
                # Simplest way: if 'new_team' is in payload, rebuild team from that
                if "new_team" in payload:
                    print(f"      Restoring team composition from evolution event...")
                    new_team_specs = payload["new_team"]
                    self.team_members = []
                    for spec in new_team_specs:
                        if spec["title"] == self.team_lead.title:
                            continue # Skip lead
                        agent = Agent(
                            title=spec["title"],
                            expertise=spec["expertise"],
                            goal="contribute", # Generic, will be updated if needed
                            role="member"
                        )
                        self.team_members.append(agent)
                    self.all_agents = [self.team_lead] + self.team_members

            # C. Reconstruct IterationRecord (Semantic History)
            # We need: approach, code, results, metrics, analysis, reflection
            # This is hard to fully reconstruct from raw events without a dedicated 'record' event
            # But we can load the JSON file!
            iter_file = self.results_dir / f"iteration_{i:02d}.json"
            if iter_file.exists():
                # This file has the summary. 
                # But we need the full IterationRecord for context builder.
                # Let's try to rebuild from what we have.
                
                # We need to find the components
                approach = ""
                meeting_evts = [e for e in evts if e.kind == "meeting" and e.payload.get("type") == "team"]
                # We don't store the full text in payload usually... 
                # But we saved meeting files!
                meeting_file = self.results_dir / "meetings" / f"iteration_{i:02d}_team_meeting.json"
                if meeting_file.exists():
                    with open(meeting_file, 'r') as f:
                        m_data = json.load(f)
                        if isinstance(m_data, dict):
                            approach = m_data.get("summary", "")
                        elif isinstance(m_data, list) and m_data:
                            # Fallback: if it's a list (transcript), try to find the last synthesis
                            # This is a best-effort recovery
                            last_msg = m_data[-1]
                            if isinstance(last_msg, dict):
                                approach = last_msg.get("message", "")
                            else:
                                approach = ""
                        else:
                            approach = ""

                # Code is already found above
                
                # Reflection
                reflection_text = ""
                ref_evts = [e for e in evts if e.kind == "reflection"]
                if ref_evts:
                    reflection_text = self.event_store.get_blob_content(ref_evts[-1], "reflection") or ""
                    # Add to reflection memory
                    self.reflection_memory.add_reflection(Reflection.from_text(i, reflection_text))
                
                # Metrics
                metrics = {}
                if exec_evts:
                    metrics = exec_evts[-1].payload.get("metrics", {})
                    
                # Re-run analysis (cheap) or try to load?
                # Let's re-run analysis to be safe and populate objects
                output_text = self.event_store.get_blob_content(exec_evts[-1], "output") if exec_evts else ""
                error_text = None # We assume success if we are replaying successful exec
                
                output_analysis = self.output_analyzer.analyze(output_text, None, None)
                # code_analysis = self.code_analyzer.analyze(code) # Removed
                
                # Create Record
                rec = IterationRecord(
                    iteration=i,
                    approach=approach,
                    code=code,
                    results={"success": True, "output": output_text}, # Minimal needed
                    metrics=metrics,
                    output_analysis=output_analysis,
                    code_analysis=None,
                    reflection=reflection_text
                )
                self.iteration_records.append(rec)
                
                # Update Evolution Agent Concepts
                # (Technique tracking removed as per user request)
                if self.evolution_agent:
                    pass
            
            last_iteration = i
            
        # Update context builder
        self.context_builder.set_iterations(self.iteration_records)
        
        return last_iteration

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
Your previous code failed. Produce a corrected version.

{fix_context}

EXECUTION CONTEXT (CRITICAL):
- These variables ALREADY EXIST with REAL DATA: {preloaded_vars}
- You MUST NOT create new DataFrames with these names.
- You MUST NOT assign any literal/dummy data to these names.
- If the old code contains such dummy definitions, REMOVE them.

Instructions:
- Return the FULL corrected Python script (no snippets or diffs).
- Keep changes as small as possible while fixing the error.

Output ONLY the complete Python code in ```python``` blocks.
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
        output_analysis: Optional[OutputAnalysis] = None,
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
            output_analysis: Semantic analysis of output (optional)

        Returns:
            Reflection text with structured sections
        """
        # Prepare output section
        # If we have analysis, we rely on it and skip raw output to save context/noise.
        # If no analysis, we fall back to raw output.
        output_section = ""
        if output_analysis:
            output_section = f"""
## Execution Analysis (Automated)
**Summary:** {output_analysis.raw_summary}

**Key Observations:**
{chr(10).join([f"- {obs}" for obs in output_analysis.key_observations])}
"""
        else:
            # Fallback: Show raw output if no analysis available
            output = results.get('output', '')
            if len(output) > 5000:
                output_preview = output[:5000] + "\n... (truncated)"
            else:
                output_preview = output
            
            output_section = f"""
## Execution Output (Raw)
{output_preview}
"""

        error_section = ""
        if not results.get('success'):
            error_section = f"""
## Error
{results.get('error', 'Unknown error')}

## Traceback
{results.get('traceback', 'No traceback available')[:1000]}
"""

        history_section = ""
        if 'execution_history' in results and len(results['execution_history']) > 1:
             history_section = "\n## Previous Failed Attempts\n"
             for i, attempt in enumerate(results['execution_history'][:-1]):
                 history_section += f"Attempt {attempt['attempt']}:\nError: {attempt['error']}\nCode snippet: {attempt['code'][:200]}...\n\n"

        # Get previous metric for comparison
        prev_metric = None
        improvement_text = ""
        if len(self.iteration_records) > 0:
            last_record = self.iteration_records[-1]
            prev_metric = last_record.metrics.get(list(metrics.keys())[0]) if metrics else None
            
        task = f"""
Reflect on this iteration and extract learnings.

## Approach
{approach}

{output_section}
{error_section}
{history_section}

## Metrics
{metrics}
"""
        if self.target_metric in last_record.metrics:
                prev_metric = last_record.metrics[self.target_metric]
                current_metric = metrics.get(self.target_metric)
                if current_metric is not None and prev_metric is not None:
                    if self.minimize_metric:
                        improved = current_metric < prev_metric
                    else:
                        improved = current_metric > prev_metric
                    
                    if improved:
                        improvement_text = f"(improved from {prev_metric:.4f})"
                    else:
                        improvement_text = f"(no improvement from {prev_metric:.4f})"
        
        if not improvement_text and prev_metric is None:
            improvement_text = "(baseline)"
        
        # Get clean summary from output analyzer if available
        output_summary = output_preview
        if hasattr(self, 'output_analyzer') and results.get('success'):
            # Use the last iteration record's output analysis if available
            if len(self.iteration_records) > 0:
                last_analysis = self.iteration_records[-1].output_analysis
                if last_analysis.raw_summary:
                    output_summary = last_analysis.raw_summary
        
        reflection_prompt = f"""You are the Principal Investigator reviewing Iteration {self.iteration} of this research experiment.

## Hypothesis (What we planned to test)
{approach}

## Experiment (What we implemented)
```python
{code}
```

## Observations (What happened)
Execution: {'Successful' if results.get('success') else 'Failed'}
Target metric ({self.target_metric}): {metrics.get(self.target_metric, 'N/A')} {improvement_text}
{output_summary}
{error_section}
{history_section}

---

## Reflection

As the research lead, analyze this iteration scientifically:

### 1. Understanding
What did this experiment teach us about the problem?
What assumptions were validated or invalidated?

### 2. Attribution  
If results improved: What specifically drove the improvement?
If results didn't improve or failed: What was the root cause? (Not symptoms - dig deep)

### 3. Constraints Discovered
What limitations of the data, approach, or problem did we encounter?

### 4. Dead Ends
What approaches can we now rule out and why?

Write as if preparing brief notes for tomorrow's team meeting. Focus on insights that will help the team make better decisions, not prescriptive recommendations.
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
        """Initialize the evolution agent with policy."""
        print("\n🧬 Initializing Evolution Agent with Policy...")
        
        # Create policy config
        config = EvolutionPolicyConfig(
            min_per_role={"lead": 1, "coder": 1},
            max_team_size=6,
            min_team_size=3,
            gap_threshold=0.3,
            max_concepts_per_new_agent=3
        )
        
        policy = DefaultEvolutionPolicy(config)
        
        # Register core agents
        # PI / Lead
        policy.register_agent(
            self.team_lead,
            AgentMeta(
                id=self.team_lead.title,
                title=self.team_lead.title,
                role="lead",
                core=True,
                tags=["leadership", "planning", "synthesis"]
            )
        )
        
        # Coder
        policy.register_agent(
            self.coding_agent,
            AgentMeta(
                id=self.coding_agent.title,
                title=self.coding_agent.title,
                role="coder",
                core=True,
                tags=["python", "implementation", "debugging"]
            )
        )
        
        # Register initial team members
        for agent in self.team_members:
            # Infer tags from expertise string
            tags = [t.strip() for t in agent.expertise.split(',')]
            policy.register_agent(
                agent,
                AgentMeta(
                    id=agent.title,
                    title=agent.title,
                    role="member",  # Generic role for now
                    core=False,
                    tags=tags
                )
            )

        self.evolution_agent = EvolutionAgent(
            llm=self.llm,
            target_team_size=(3, 6),
            gap_threshold=0.3,
            specialize_overlap_threshold=0.4,
            specialize_gini_threshold=0.5,
            policy=policy
        )
        
        self.evolution_agent.initialize(
            agents=self.all_agents,
            problem_statement=problem_statement,
            target_metric=target_metric,
            minimize_metric=minimize_metric,
            column_schemas=self.column_schemas,
            techniques=None
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
            if iter_record.code_analysis:
                for technique in iter_record.code_analysis.techniques:
                    agent.knowledge_base.add_technique(technique)
            
            # Track success/failure patterns
            if is_improvement and current_metric is not None and prev_metric is not None:
                improvement = abs(current_metric - prev_metric)
                technique = "approach"
                if iter_record.code_analysis and iter_record.code_analysis.techniques:
                    technique = iter_record.code_analysis.techniques[0]
                
                agent.knowledge_base.add_success_pattern(
                    iteration=iter_record.iteration,
                    technique=technique,
                    metric=target_metric,
                    improvement=improvement
                )
            elif not is_improvement and prev_metric is not None:
                technique = "approach"
                if iter_record.code_analysis and iter_record.code_analysis.techniques:
                    technique = iter_record.code_analysis.techniques[0]
                
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
            techniques_added = []
            if iter_record.code_analysis:
                techniques_added = iter_record.code_analysis.techniques
                
            self.event_store.log_event(
                kind="kb_update",
                iteration=iter_record.iteration,
                payload={
                    "techniques_added": techniques_added,
                    "is_improvement": is_improvement
                }
            )

        # === NEW: update concept space based on techniques ===
        if self.evolution_agent is not None and iter_record.code_analysis:
            self.evolution_agent.maybe_expand_concepts_from_techniques(
                iter_record.code_analysis.techniques
            )
