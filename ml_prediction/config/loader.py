"""Compatibility wrapper to reuse shared platform config loader."""

from services.config.loader import (  # noqa: F401
    load_config,
    resolve_config_path,
    get_active_cluster,
    get_cluster,
    get_remote_server,
    get_remote_paths,
    get_queue_defaults,
    get_template_path,
    get_executor,
    get_commands,
    get_local_paths,
    get_templates,
    get_module_task,
)

__all__ = [
    "load_config",
    "resolve_config_path",
    "get_active_cluster",
    "get_cluster",
    "get_remote_server",
    "get_remote_paths",
    "get_queue_defaults",
    "get_template_path",
    "get_executor",
    "get_commands",
    "get_local_paths",
    "get_templates",
    "get_module_task",
]
