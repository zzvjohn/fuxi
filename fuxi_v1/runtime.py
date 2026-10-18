"""Optional workspace governance at the CLI boundary; absent in the pure research core."""

from contextlib import contextmanager
from pathlib import Path
from .contracts import require


def authority_location(output_root):
    """One workspace-wide exposure/authority ledger, independent of campaign output folders."""
    output = Path(output_root).resolve()
    workspace = next((p for p in (output, *output.parents)
                      if (p / "codex_automations/fuxi_runtime_policy.json").is_file()), None)
    return (workspace / "data/fuxi_v1/authority.sqlite") if workspace else output / "authority.sqlite"


@contextmanager
def controlled_writer(output_root):
    output = Path(output_root).resolve()
    workspace = next((p for p in (output, *output.parents)
                      if (p / "codex_automations/fuxi_runtime_policy.json").is_file()), None)
    if workspace is None:
        yield
        return
    from .governance import governed_writer, inherited_identity, native_cli_action, validate_native_budget
    parent = inherited_identity(workspace)
    trigger = parent['trigger'] if parent else 'manual'
    action = native_cli_action()
    if trigger == 'automatic':
        require(parent is not None and parent['action'] == action,
                'native CLI action differs from inherited controller phase')
        validate_native_budget(parent)
    with governed_writer(workspace, action, trigger=trigger) as identity:
        yield identity
