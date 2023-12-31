from pynvml import *

def check_nvml_error(logger): 
    try:
        nvmlInit()
        logger.info(f"Driver Version: {nvmlSystemGetDriverVersion()}")
        deviceCount = nvmlDeviceGetCount()
        devices = []  
        for i in range(deviceCount):
            handle = nvmlDeviceGetHandleByIndex(i)
            logger.info(f"Device {i}: {nvmlDeviceGetName(handle)}")
            #gpu_device = GPUDevice(handle=handle, gpu_index=i) 
            devices.append(nvmlDeviceGetTotalEnergyConsumption(handle))
        return 0
    except NVMLError:  # if the input was invalid
        return 1
  