from flytekit import LaunchPlan




lp = LaunchPlan.get_or_create(workflow="detect", default_inputs={
    "model"
})

