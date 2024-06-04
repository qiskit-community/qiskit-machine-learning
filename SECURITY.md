# Security Policy

## Supported Versions

Qiskit Machine Learning supports one minor version release at a time, both for bug and security fixes.

> [!TIP]
> For example, if the most recent release is `0.7.2`, then the current major release series is `0.x` the current minor 
> release is `0.7.x`, with `0.7.2` being the current patch release.

As an additional resource, you can find more details on the release and support schedule of Qiskit in the [documentation](https://docs.quantum.ibm.com/start/install#release-schedule).

## Reporting a Vulnerability

You can privately report a potential vulnerability or security issue
via the GitHub security vulnerabilities feature, which can be accessed here:

https://github.com/qiskit-community/qiskit-machine-learning/security/advisories

> [!IMPORTANT]
> We kindly ask that you do not open a public GitHub issue about the vulnerability until we have had a chance to 
investigate and, if confirmed, address it. We are committed to working with you to coordinate a public disclosure 
timeline that allows us to release a fix and inform the users.

1. **Include Details**: In your report, please include as much information as possible to help us understand the 
nature and scope of the vulnerability. This might include:
   - The link to the filed issue stub.
   - A description of the vulnerability and its impact.
   - Steps to reproduce the issue or a proof-of-concept (PoC) for independent confirmation.
   - Any potential fixes or recommendations you might have.

2. **Response Time**: We will acknowledge your report within 3 business days and provide you with an estimated time frame for resolving the issue.


### Untrusted models
Models can be manipulated to produce undesired outputs and can be susceptible to 
backdoor triggers to expose confidential information[^data-poisoning-sources]. Be careful about using untrusted models 
and sharing models.

You can find more details on the security vulnerability feature in the GitHub documentation here:

https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing/privately-reporting-a-security-vulnerability

Thank you for helping keep our project secure! 

[^data-poisoning-sources]: To understand risks of utilization of data from unknown sources, read the following Cornell 
papers on data poisoning and model safety:
    https://arxiv.org/abs/2312.04748
    https://arxiv.org/abs/2401.05566
