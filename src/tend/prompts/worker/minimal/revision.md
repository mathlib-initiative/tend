Resume the worker session for contribution {contribution_id}.
Previous feedback:
{feedback_message}
Revise the same worktree and leave the task file complete when the revision is done. If the feedback reports a merge conflict, run `git merge main` (or `git rebase main`) inside the worktree and resolve the conflicts however is correct for the task — `main` is a local branch in the same `.git` as your worktree, so no `git fetch` is needed and there may not even be a remote configured.
Commit your work yourself; anything left uncommitted at session end will not be published.
When done, call the final_result tool exactly once with the structured worker contribution summary.
