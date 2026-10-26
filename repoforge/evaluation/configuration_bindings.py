"""Frozen action implementation names and their public tool schemas."""
import importlib


def tool_bindings(manifest, job):
    from repoforge.actions.finish import Finish
    from repoforge.actions.read_observation import ReadObservation
    from repoforge.actions.task_state import UpdateTaskState
    from repoforge.cli.service import TOOLS

    managed = job['entrypoint'] == 'cli' or manifest['profiles'][job['variant']]['managed']
    if job['entrypoint'] == 'cli':
        actions = [TOOLS[name] for name in manifest['cli_config'].get('tools', list(TOOLS))] + [Finish]
    else:
        actions = []
        for config in manifest['agent_configs'][job['variant']]['actions']:
            module, name = config['action_class'].rsplit('.', 1)
            actions.append(getattr(importlib.import_module(module), name))
    if managed:
        actions.append(ReadObservation)
        if manifest['cli_config']['context_policy'].get('enable_task_state', True):
            actions.append(UpdateTaskState)
    return {action.__name__: action.args_schema.name for action in actions}

