# Dection workflow
This directory has been refactored to be completely standalone and does not rely on any other code (or files) in this
repo. Any previous depencies have been moved to S3. 

The idea is to formalize the detection workflow in a portable and reproducable framework. 


## Flyte 
Flyte is a "workflow automation platform for complex, mission-critical data and ML processes at scale". It's an open 
source project currently incubating with the Linux Foundation. 

In theory Flyte will allow us to run the workflow on both the command line, a local docker container, and the cloud.
We shall see....