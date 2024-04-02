import subprocess
import re

from cpuinfo import get_cpu_info
from GPUtil import GPUtil

def get_cuda_version():
    regex = re.compile(r"CUDA Version: (\d+)\.(\d+)")
    try:
        nvidia_smi_proc = subprocess.run("nvidia-smi", capture_output=True, text=True, check=True)

        match = re.search(regex, nvidia_smi_proc)
        if match is not None:
            # Return the major version
            return {"major_version": match.group(1), "minor_version": match.group(2)}

    except subprocess.CalledProcessError:
        pass

    return None


def retrieve_platform_specifics() -> dict:
    cpu_info = get_cpu_info()
    gpus = GPUtil.getGPUs()
    gpu_names = list({gpu.name for gpu in gpus})
    cuda_info = get_cuda_version()
    return {
        "python_version": cpu_info["python_version"],
        "architecture": cpu_info["arch_string_raw"],
        "bits": cpu_info["bits"],
        "cpu_count": cpu_info["count"],
        "cpu_model": cpu_info.get("brand_raw"),
        "gpu_count": len(gpu_names),
        "gpu_names": gpu_names,
        "cuda_major_version": cuda_info["major_version"] if cuda_info is not None else None,
        "cuda_minor_version": cuda_info["minor_version"] if cuda_info is not None else None,
    }
