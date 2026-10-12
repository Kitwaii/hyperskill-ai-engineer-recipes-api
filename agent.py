import asyncio
import os
import warnings
from typing import Any

# Silence deprecation warnings raised inside third-party libraries (pydantic, workflows) on Python 3.14.
warnings.filterwarnings("ignore", category=DeprecationWarning)

import dotenv
from github import Auth, Github
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow, AgentOutput, ToolCall, ToolCallResult
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAIResponses

dotenv.load_dotenv()

# Provided by the GitHub Actions workflow (REPOSITORY is "username/repo-name").
repository = os.getenv("REPOSITORY")
pr_number = os.getenv("PR_NUMBER")

git = Github(auth=Auth.Token(os.getenv("GITHUB_TOKEN"))) if os.getenv("GITHUB_TOKEN") else Github()

repo = git.get_repo(repository)


def get_pr_details(pr_number: int) -> dict[str, Any]:
    """Useful for getting the details of a pull request given its number.
    Returns the author, title, body, diff URL, state, head SHA and the commit SHAs of the PR."""
    pull_request = repo.get_pull(pr_number)

    commit_SHAs = []
    commits = pull_request.get_commits()
    for c in commits:
        commit_SHAs.append(c.sha)

    return {
        "user": pull_request.user.login,
        "title": pull_request.title,
        "body": pull_request.body,
        "diff_url": pull_request.diff_url,
        "state": pull_request.state,
        "commit_SHAs": commit_SHAs,
        "head_sha": pull_request.head.sha,
    }


def get_file_contents(file_path: str) -> str:
    """Useful for fetching the contents of a file from the repository given its path."""
    file_content = repo.get_contents(file_path)
    return file_content.decoded_content.decode('utf-8')


def get_pr_commit_details(head_sha: str) -> list[dict[str, Any]]:
    """Useful for getting the details of a commit given its SHA.
    Returns the changed files with their filename, status, additions, deletions, changes and patch (diff)."""
    commit = repo.get_commit(head_sha)
    changed_files: list[dict[str, Any]] = []
    for f in commit.files:
        changed_files.append({
            "filename": f.filename,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "changes": f.changes,
            "patch": f.patch,
        })
    return changed_files


async def add_context_to_state(ctx: Context, gathered_contexts: str) -> str:
    """Useful for adding the gathered context (PR details, changed files, requested files) to the state."""
    current_state = await ctx.store.get("state")
    current_state["gathered_contexts"] = gathered_contexts
    await ctx.store.set("state", current_state)
    return "State updated with gathered contexts."


async def add_comment_to_state(ctx: Context, draft_comment: str) -> str:
    """Useful for adding the draft PR review comment to the state."""
    current_state = await ctx.store.get("state")
    current_state["draft_comment"] = draft_comment
    await ctx.store.set("state", current_state)
    return "State updated with draft comment."


async def add_final_review_to_state(ctx: Context, final_review: str) -> str:
    """Useful for adding the final, checked PR review comment to the state."""
    current_state = await ctx.store.get("state")
    current_state["final_review"] = final_review
    await ctx.store.set("state", current_state)
    return "State updated with final review."


def post_review_to_github(pr_number: int, comment: str) -> str:
    """Useful for posting the final review comment to a pull request on GitHub given the PR number and the comment."""
    pull_request = repo.get_pull(pr_number)
    pull_request.create_review(body=comment, event="COMMENT")
    return f"Review posted to PR #{pr_number} on GitHub."


pr_details_tool = FunctionTool.from_defaults(get_pr_details)
file_contents_tool = FunctionTool.from_defaults(get_file_contents)
pr_commit_details_tool = FunctionTool.from_defaults(get_pr_commit_details)
add_context_to_state_tool = FunctionTool.from_defaults(add_context_to_state)
add_comment_to_state_tool = FunctionTool.from_defaults(add_comment_to_state)
add_final_review_to_state_tool = FunctionTool.from_defaults(add_final_review_to_state)
post_review_to_github_tool = FunctionTool.from_defaults(post_review_to_github)

llm = OpenAIResponses(
    model=os.getenv("OPENAI_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
)

context_agent = FunctionAgent(
    llm=llm,
    name="ContextAgent",
    description="Gathers all the needed context from the GitHub repository: PR details (author, title, body, "
                "diff_url, state, head_sha), changed files and any requested repository files.",
    tools=[pr_details_tool, file_contents_tool, pr_commit_details_tool, add_context_to_state_tool],
    system_prompt=(
        "You are the context gathering agent. When gathering context, you MUST gather \n: \n"
        "  - The details: author, title, body, diff_url, state, and head_sha; \n\n"
        "  - Changed files; \n\n"
        "  - Any requested for files; \n\n"
        "Once you gather the requested info, you MUST hand control back to the Commentor Agent. "
    ),
    can_handoff_to=["CommentorAgent"],
)

commentor_agent = FunctionAgent(
    llm=llm,
    name="CommentorAgent",
    description="Uses the context gathered by the context agent to draft a pull review comment comment.",
    tools=[add_comment_to_state_tool],
    system_prompt=(
        "You are the commentor agent that writes review comments for pull requests as a human reviewer would. \n \n"
        "Ensure to do the following for a thorough review: \n"
        " - Request for the PR details, changed files, and any other repo files you may need from the ContextAgent. \n"
        " - Once you have asked for all the needed information, write a good ~200-300 word review in markdown format detailing: \n\n"
        "    - What is good about the PR? \n\n"
        "    - Did the author follow ALL contribution rules? What is missing? \n\n"
        "    - Are there tests for new functionality? If there are new models, are there migrations for them? - use the diff to determine this. \n\n"
        "    - Are new endpoints documented? - use the diff to determine this. \n \n"
        "    - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement. \n\n"
        " - If you need any additional details, you must hand off to the Commentor Agent. \n\n"
        " - You should directly address the author. So your comments should sound like: \n\n"
        " \"Thanks for fixing this. I think all places where we call quote should be fixed. Can you roll this fix out everywhere?\" \n\n"
        " - You must hand off to the ReviewAndPostingAgent once you are done drafting a review. "
    ),
    can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"],
)

review_and_posting_agent = FunctionAgent(
    llm=llm,
    name="ReviewAndPostingAgent",
    description="Checks the draft review written by the CommentorAgent, requests rewrites when it does not meet "
                "the review criteria, saves the final review to the state and posts it to the PR on GitHub.",
    tools=[add_final_review_to_state_tool, post_review_to_github_tool],
    system_prompt=(
        "You are the Review and Posting agent. You must use the CommentorAgent to create a review comment. \n"
        "Once a review is generated, you need to run a final check and post it to GitHub.\n"
        "   - The review must: \n"
        "   - Be a ~200-300 word review in markdown format. \n"
        "   - Specify what is good about the PR: \n"
        "   - Did the author follow ALL contribution rules? What is missing? \n"
        "   - Are there notes on test availability for new functionality? If there are new models, are there migrations for them? \n"
        "   - Are there notes on whether new endpoints were documented? \n"
        "   - Are there suggestions on which lines could be improved upon? Are these lines quoted? \n"
        " If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns. \n"
        " When you are satisfied, post the review to GitHub.  "
    ),
    can_handoff_to=["CommentorAgent"],
)

workflow_agent = AgentWorkflow(
    agents=[context_agent, commentor_agent, review_and_posting_agent],
    root_agent=review_and_posting_agent.name,
    initial_state={
        "gathered_contexts": "",
        "draft_comment": "",
        "final_review": "",
    },
)


async def main():
    query = f"Write a review for PR number {pr_number}"

    handler = workflow_agent.run(query)

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\n\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")


if __name__ == "__main__":
    asyncio.run(main())
    git.close()
