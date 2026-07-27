# Short-Response Model — Next-Stage Package (Phase 0 completed)

This is a complete continuation of the earlier starter ZIP. It includes the original benchmark and model scripts plus populated train/dev/sealed datasets, validation tools, an audit workflow, and PowerShell commands.

## Run order

```powershell
.\.venv\Scripts\Activate.ps1
.\scripts\01_validate_data.ps1
```

Review `reports/HUMAN_DATA_AUDIT_PACKET.md`, repair or reject weak records, then run:

```powershell
.\scripts\02_evaluate_base.ps1
.\scripts\03_train_all_layer_qlora.ps1
.\scripts\04_evaluate_dense_sft.ps1
```

Run the layer scan only after the dense-SFT behavior gate passes:

```powershell
.\scripts\05_run_layer_scan.ps1
```

The sealed split must remain isolated from training and routine tuning.
