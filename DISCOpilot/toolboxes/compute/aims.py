from typing import Dict, Optional, Tuple

from services.aims.workflow import AimsWorkflowManager


def prepare_aims_inputs(
    structure_dict: Dict,
    structure_name: str,
    aims_params: Optional[Dict] = None,
    slurm_params: Optional[Dict] = None,
    control_template: Optional[str] = None,
    slurm_template: Optional[str] = None,
    task_key: Optional[str] = None,
) -> Tuple[bool, Dict]:
    workflow = AimsWorkflowManager({})
    return workflow.prepare_calculation(
        structure_dict,
        structure_name,
        aims_params=aims_params,
        slurm_params=slurm_params,
        control_content=control_template,
        slurm_content=slurm_template,
        task_key=task_key,
    )


def run_aims_calculation(
    structure_dict: Dict,
    structure_name: str,
    config_dict: Dict,
    aims_params: Optional[Dict] = None,
    slurm_params: Optional[Dict] = None,
    control_template: Optional[str] = None,
    slurm_template: Optional[str] = None,
    task_key: Optional[str] = None,
    submit: bool = False,
) -> Dict:
    workflow = AimsWorkflowManager(config_dict)

    if not submit:
        ok, files = workflow.prepare_calculation(
            structure_dict,
            structure_name,
            aims_params=aims_params,
            slurm_params=slurm_params,
            control_content=control_template,
            slurm_content=slurm_template,
            task_key=task_key,
        )
        return {
            "status": "prepared" if ok else "error",
            "files": files,
        }

    ok, msg = workflow.connect_remote()
    if not ok:
        return {"status": "error", "error": f"SSH connect failed: {msg}"}

    ok, files = workflow.prepare_calculation(
        structure_dict,
        structure_name,
        aims_params=aims_params,
        slurm_params=slurm_params,
        control_content=control_template,
        slurm_content=slurm_template,
        task_key=task_key,
    )
    if not ok:
        workflow.disconnect_remote()
        return {"status": "error", "error": files.get("error", "prepare failed")}

    ok, result = workflow.submit_calculation(files)
    if ok:
        return {
            "status": "submitted",
            "job_id": result,
            "remote_dir": workflow.current_remote_dir,
        }

    workflow.disconnect_remote()
    return {"status": "error", "error": result}
