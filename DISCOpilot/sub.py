import os
import random
import time
import shutil
import getpass
import subprocess
from ase.db import connect

# ====== 并发监控参数 ======
MAX_JOBS = 1000          # 允许同时在队列中的最大作业数（运行+排队）
CHECK_INTERVAL = 30    # 达上限时的轮询间隔（秒）

def _cmd_exists(cmd: str) -> bool:
    return shutil.which(cmd) is not None

def _count_nonempty_lines(lines):
    lines = [ln for ln in lines.splitlines() if ln.strip()]
    # 常见的 djob 表头包含 "JOBID"；若有则去掉第一行
    if lines and "JOBID" in lines[0].upper():
        lines = lines[1:]
    return len(lines)

def infer_scheduler(queue: str) -> str:
    """根据队列脚本头自动判断调度器类型。"""
    header = (queue or "").upper()
    if "#PBS" in header:
        return 'pbs'
    if "#SBATCH" in header:
        return 'slurm'
    if "#DSUB" in header:
        return 'dsub'
    return 'unknown'

def get_submit_command(queue: str) -> str:
    scheduler = infer_scheduler(queue)
    if scheduler == 'pbs':
        return 'qsub'
    if scheduler == 'slurm':
        return 'sbatch'
    if scheduler == 'dsub':
        return 'dsub'
    raise ValueError('Unknown scheduler type; cannot determine submit command.')

def get_current_job_count(queue: str) -> int:
    """
    根据队列类型返回当前用户在队列中的作业数（运行+排队）。
    查询失败时返回一个大数以避免继续提交。
    """
    user = os.environ.get("USER") or getpass.getuser()
    scheduler = infer_scheduler(queue)
    try:
        if scheduler == 'slurm' and _cmd_exists('squeue'):
            out = subprocess.check_output(f"squeue -h -u {user}",
                                          shell=True, text=True, stderr=subprocess.STDOUT)
            return len([ln for ln in out.splitlines() if ln.strip()])
        elif scheduler == 'dsub' and _cmd_exists('djob'):
            try:
                out = subprocess.check_output(f"djob -u {user}",
                                              shell=True, text=True, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError:
                out = subprocess.check_output("djob", shell=True, text=True, stderr=subprocess.STDOUT)
            return _count_nonempty_lines(out)
        elif scheduler == 'pbs' and _cmd_exists('qstat'):
            try:
                out = subprocess.check_output(f"qstat -u {user}",
                                              shell=True, text=True, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError:
                out = subprocess.check_output("qstat", shell=True, text=True, stderr=subprocess.STDOUT)
            lines = [ln for ln in out.splitlines() if ln.strip()]
            filtered = []
            for ln in lines:
                upper = ln.upper()
                if upper.startswith('JOB') or upper.startswith('---'):
                    continue
                if user in ln or scheduler == 'pbs':
                    filtered.append(ln)
            return len(filtered)
        else:
            return 10**9
    except Exception:
        return 10**9

def wait_for_slot(queue: str, max_jobs: int = MAX_JOBS, check_interval: int = CHECK_INTERVAL):
    """如当前作业数已达上限，则等待直到有空位。"""
    while True:
        cur = get_current_job_count(queue)
        if cur < max_jobs:
            break
        print(f"[info] queue full: {cur}/{max_jobs}, sleep {check_interval}s")
        time.sleep(check_interval)

queue_huawei = """#!/bin/sh
#DSUB --job_type cosched
#DSUB -n {job_name}
#DSUB -A root.yinghsxtslwei
#DSUB -q root.default
#DSUB -R cpu=120
#DSUB -N 1
#DSUB -oo out.log
#DSUB -eo err.log
echo "PWD:    $(pwd)"    >> out
module purge
module use /home/HPCBase/workspace/public/software/modules
module load compilers/bisheng/bisheng2.5.0 
module load mpi/hmpi/1.3.0/hmpi1.3.0-bisheng2.5.0
module load libs/openblas/openblas0.3.6_bisheng2.5.0
module load libs/fftw/3.3.8/fftw3.3.8-bisheng2.5.0_hmpi1.3.0
export PATH=/home/yinghsxtslwei/cccs-share15/vasp5.4.4/vasp.5.4.4_bisheng2.5.0/bin:$PATH
# 构建 hostfile
if [ -n "${{CCS_ALLOC_FILE}}" ]; then
    cat ${{CCS_ALLOC_FILE}}
fi
export HOSTFILE=/tmp/hostfile.$$
awk '{{ if (NF>=3) print $1" slots="$2 >> ENVIRON["HOSTFILE"] }}' ${{CCS_ALLOC_FILE}}
"""

queue_zyy = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=out.%j
#SBATCH --error=err.%j
#SBATCH --partition=vasp
#SBATCH --nodes=1
#SBATCH --exclusive

#CURRENT_HOSTNAME=$(hostname)
#if [ "$CURRENT_HOSTNAME" == "node9" ]; then
#    num_cores=50
#fi

source /Public/home/zyy/intel/oneapi/setvars.sh intel64 > /dev/null 2>&1
export PATH=/Public/home/zyy/soft/vasp.6.3.2_c/bin:$PATH
"""

queue_jilin = """#!/bin/sh
#SBATCH --job-name=revision-{job_name}
#SBATCH --output=out.%j
#SBATCH --error=err.%j
#SBATCH --partition=computerPartiton
#SBATCH --nodes=1
#SBATCH --exclusive
export PATH=/public/home/inspur/liuw/zyy/soft/vasp.6.4.2/bin:$PATH
"""

queue_jilin_aims = """#!/bin/sh
#SBATCH --job-name=revision-{job_name}
#SBATCH --output=out.%j
#SBATCH --error=err.%j
#SBATCH --partition=computerPartiton
#SBATCH --nodes=1
#SBATCH --exclusive
"""

queue_group = """#!/bin/bash
#PBS -N {job_name}
#PBS -q wq_CPU
#PBS -l nodes=1:ppn=64,mem=125gb
#PBS -j oe
#PBS -o pbs.out

cd $PBS_O_WORKDIR

source /public/apps/oneapi/setvars.sh
module load vasp/5.4.4
"""

# 初始磁矩
element_magmoms = {
    'Sc': 3.0, 'Ti': 3.0, 'V' : 3.0, 'Cr': 3.0, 'Mn': 5.0, 'Fe': 3.0, 'Co': 3.0, 'Ni': 3.0, 'Cu': 1.0, 'Zn': 1.0,
    'Y' : 2.0, 'Zr': 2.0, 'Nb': 2.0, 'Mo': 3.0, 'Ru': 1.0, 'Rh': 1.0, 'Pd': 1.0, 'Ag': 1.0, 'Cd': 1.0, 'Hf': 1.0,
    'Ta': 1.0, 'W' : 3.0, 'Re': 3.0, 'Os': 3.0, 'Ir': 3.0, 'Pt': 3.0,}

# VASP计算参数
#opt_sp = {'xc': 'pbe', 'prec': 'Normal', 'encut': 450, 'potim': 0.5, 'ibrion': -1, 'nsw': 0, 'nelm': 1000, 'ediff': 5e-6, 'ediffg': -0.05, 'ismear': 0, 'sigma': 0.05, 'ivdw': 12, 'lwave': False, 'lcharg': False, 'istart': 0, 'icharg': 2, 'algo': 'N', 'npar': 4, 'isif': 2, 'ISYM':0, 'ispin': 1, 'kpts': [3, 3, 1], 'gamma': True, 'command': 'time -p mpirun --allow-run-as-root --mca plm_rsh_agent /opt/batch/agent/tools/dstart -x PATH=$PATH -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH -np 120 -x OMP_NUM_THREADS=1 vasp_std'}

opt_sp = {'xc': 'pbe', 'prec': 'Normal', 'encut': 400, 'potim': 0.5, 'ibrion': 2, 'nsw':500, 'nelm': 5000, 'ediff': 1e-5, 'ediffg': -0.05, 
          'ismear': 0, 'sigma': 0.05, 'ivdw': 20, 'lwave': False, 'lcharg': False, 'istart': 0, 'icharg': 2, 'algo': 'N', 'npar': 4, 
          'isif': 2, 'ispin': 1, 'kpts': [1, 1, 1], 'gamma': True, 'LREAL': 'Auto', 'command': 'time -p mpirun -np $(nproc) vasp_gam'}
#
# FHI-aims 低精度单点参数（与当前 benchmark 对齐）
AIMS_COMMAND = 'mpirun -np $(nproc) /public/home/inspur/liuw/software/fhi-aims.221103/bin/aims.221103.scalapack.mpi.x > aims.out'
AIMS_SPECIES_DIR = '/public/home/inspur/liuw/software/fhi-aims.221103/species_defaults/defaults_2020/light'
aims_sp = {
    'xc': 'pbe',
    'k_grid': '1 1 1',
    'spin': 'none',
    'occupation_type': 'gaussian 0.05',
    'sc_accuracy_rho': 1e-5,
    'sc_accuracy_etot': 1e-6,
    'sc_accuracy_eev': 1e-3,
    'd3': '',
}

def build_constraint_block(fix) -> str:
    """构造写入提交脚本的约束代码块。"""
    if fix is True:
        return """
fixed_indices = [i for i, tag in enumerate(atoms.get_tags()) if tag == 0]
atoms.set_constraint(FixAtoms(indices=fixed_indices))
"""

    if isinstance(fix, str) and fix.lower() in {'benzene_cz', 'pt_fix_cz'}:
        return """
pt_indices = [i for i, atom in enumerate(atoms) if atom.symbol == 'Pt']
c_indices = [i for i, atom in enumerate(atoms) if atom.symbol == 'C']
constraints = [FixAtoms(indices=pt_indices)]
constraints.extend(FixScaled(i, mask=(False, False, True), cell=atoms.cell) for i in c_indices)
atoms.set_constraint(constraints)
"""

    return """
# keep constraints from ASE DB as-is
"""

def submit_job(
    job_id,
    vasp_params,
    read=None,
    save=None,
    fix=False,
    queue=queue_huawei,
    check_queue=False,
    max_jobs: int = MAX_JOBS,
    check_interval: int = CHECK_INTERVAL,
):
    """
    读取 ASE 数据库中 id=job_id 的条目，使用条目里的 name 和 ads 生成工作目录 name/ads，
    如果没有 name 和 ads 字段，使用 job_id 生成文件夹（不加额外的路径）。
    """
    if check_queue:
        wait_for_slot(queue, max_jobs=max_jobs, check_interval=check_interval)

    # 绝对路径避免 cd 后相对路径出错
    read_abs = os.path.abspath(read)
    save_abs = os.path.abspath(save)

    # 取出该条目原始的 name / ads
    row = connect(read_abs).get(id=job_id)

    # 组装 VASP 参数与约束片段
    param_str = ', '.join(f'{k}={v!r}' for k, v in vasp_params.items())
    magmoms_str = f"{element_magmoms!r}"
    constraint_block = build_constraint_block(fix)

    header = queue.format(job_name=f"MgO_{job_id}")
    submit_cmd = get_submit_command(queue)

    workdir = str(job_id)
    os.makedirs(workdir, exist_ok=True)

    script = header + f"""python - << EOF
import os, time; from ase.db import connect
from ase.calculators.vasp import Vasp; from ase.constraints import FixAtoms, FixScaled

db_read = connect(r'{read_abs}')
db_save = connect(r'{save_abs}')

row = db_read.get(id={job_id})
atoms = row.toatoms()
atoms.set_pbc([True, True, True])
# 约束（如需）
{constraint_block.strip()}

# 计算
calc = Vasp({param_str})
atoms.calc = calc
t0 = time.time()
atoms.set_initial_magnetic_moments(magmoms=[{magmoms_str}.get(atom.symbol, 1.0) for atom in atoms])
energy = atoms.get_potential_energy()
runtime_hours = round((time.time() - t0) / 3600, 4)

db_save.write(
    atoms,
    id={job_id},
    node=os.uname().nodename,
    time=runtime_hours
)
EOF
"""

    script_path = os.path.join(workdir, 'job.sh')
    with open(script_path, 'w') as f:
        f.write(script)
    cwd = os.getcwd()
    os.chdir(workdir)
    os.system('chmod +x job.sh')
    os.system(f'{submit_cmd} job.sh')
    os.chdir(cwd)
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} submitted: MgO_{job_id} via {submit_cmd}")


def submit_job_aims(
    job_id,
    aims_params,
    read=None,
    save=None,
    fix=False,
    queue=queue_jilin_aims,
    aims_command=AIMS_COMMAND,
    z_vacuum=100.0,
    check_queue=False,
    max_jobs: int = MAX_JOBS,
    check_interval: int = CHECK_INTERVAL,
):
    """提交 FHI-aims 单点任务，并将 wall time 写回 ASE DB。"""
    if check_queue:
        wait_for_slot(queue, max_jobs=max_jobs, check_interval=check_interval)

    read_abs = os.path.abspath(read)
    save_abs = os.path.abspath(save)
    connect(read_abs).get(id=job_id)

    param_str = ', '.join(f'{k}={v!r}' for k, v in aims_params.items())
    constraint_block = build_constraint_block(fix)

    header = queue.format(job_name=f"AIMS_{job_id}")
    submit_cmd = get_submit_command(queue)
    workdir = str(job_id)
    os.makedirs(workdir, exist_ok=True)

    script = header + f"""python - << EOF
import os, time
from ase.db import connect
from ase.calculators.aims import Aims, AimsProfile
from ase.constraints import FixAtoms, FixScaled

db_read = connect(r'{read_abs}')
db_save = connect(r'{save_abs}')

row = db_read.get(id={job_id})
atoms = row.toatoms()
atoms.set_pbc([True, True, True])
if {float(z_vacuum)} > 0:
    atoms.center(vacuum={float(z_vacuum)}, axis=2)
{constraint_block.strip()}

profile = AimsProfile(
    command=r'{aims_command}',
    default_species_directory=r'{AIMS_SPECIES_DIR}',
)
calc = Aims(profile=profile, {param_str})
atoms.calc = calc

t0 = time.time()
energy = atoms.get_potential_energy()
runtime_hours = round((time.time() - t0) / 3600, 4)

db_save.write(
    atoms,
    id={job_id},
    node=os.uname().nodename,
    method='aims',
    time=runtime_hours,
    e_sp=float(energy),
)
EOF
"""

    script_path = os.path.join(workdir, 'job.sh')
    with open(script_path, 'w') as f:
        f.write(script)

    cwd = os.getcwd()
    os.chdir(workdir)
    os.system('chmod +x job.sh')
    os.system(f'{submit_cmd} job.sh')
    os.chdir(cwd)
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} submitted: AIMS_{job_id} via {submit_cmd}")

if __name__ == '__main__':
    for i in range(1, 2):
        submit_job(i, vasp_params=opt_sp, read='Pt_Ph_stretch_0.2A_15A.db', save='Pt_Ph_stretch_0.2A_15A_opt.db', fix='benzene_cz', queue=queue_group)
        #submit_job_aims(i, aims_params=aims_sp, read='oc20_easy150_preview.db', save='oc20_easy150_preview_aims_opt.db', fix=False, queue=queue_jilin_aims)
