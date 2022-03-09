from tokens import Tokens
from qiskit_ibm_runtime import IBMRuntimeService

# Save account to disk.
IBMRuntimeService.save_account(auth="cloud",
                               token=Tokens.CLOUD_TOKEN,
                               # instance=Tokens.CLOUD_INSTANCE_LITE,
                               instance= Tokens.CLOUD_INSTANCE_STD,
                               overwrite=True)

service = IBMRuntimeService()
print(service)
from qiskit.test.reference_circuits import ReferenceCircuits
from qiskit_ibm_runtime import IBMRuntimeService

service = IBMRuntimeService()
program_inputs = {'iterations': 1}
# options = {"backend_name": "ibmq_qasm_simulator"}
options = {"backend_name": "ibm_canberra"}
job = service.run(program_id="hello-world",
                  options=options,
                  inputs=program_inputs
                 )
print(f"job id: {job.job_id}")
result = job.result()
print(result)

