"""Compatibility wrapper to use shared SSH manager."""

from services.remote_server.ssh_manager import SSHManager, build_slurm_script

__all__ = ["SSHManager", "build_slurm_script"]
