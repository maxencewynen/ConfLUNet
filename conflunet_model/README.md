# Bundled default inference model (Dataset321_WMLIS)

Not stored in git (weights are large). To rebuild this tree before `docker build`:

conflunet_model/
├── nnUNet_results/Dataset321_WMLIS/ConfLUNet_FLAIR_v24-12-11/fold_{0..4}/checkpoint_best_Panoptic_Quality_ConfLUNet.pth
├── nnUNet_results/Dataset321_WMLIS/{dataset.json,nnUNetPlans.json}
└── nnUNet_preprocessed/Dataset321_WMLIS/{dataset.json,nnUNetPlans.json}

Source of weights: <SAMBA share path / internal location>
Copy with the commands in docs/docker_instructions.md.
